import os
import glob
import numpy as np
import tensorflow as tf
from pycochleagram import erbfilter as erb, cochleagram as cgram
import scipy.signal as signallib
import re

tf.compat.v1.disable_eager_execution()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_wav_file(wav_path):
    from scipy.io import wavfile
    sr, audio = wavfile.read(wav_path)
    if audio.ndim == 1: audio = np.stack([audio, audio], axis=1)
    if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
    return audio, sr

def make_downsample_filter(sr=48000, env_sr=8000, window_size=4097, beta=10.06):
    downsample_ratio = sr // env_sr
    downsample_filter_times = np.arange(-window_size/2, int(window_size/2))
    downsample_filter_response_orig = np.sinc(downsample_filter_times / downsample_ratio) / downsample_ratio
    try:
        downsample_filter_window = signallib.windows.kaiser(window_size, beta)
    except AttributeError:
        downsample_filter_window = signallib.kaiser(window_size, beta)
    downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    return downsample_filter_response.astype(np.float32), downsample_ratio

def wav_to_model_input(audio, sr=48000, target_sr=48000, n_channels=39, low_lim=30, hi_lim=20000, final_samples=8000):
    if sr != target_sr:
        from scipy import signal
        n_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, n_samples)
        sr = target_sr
        
    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
    cochleagrams = []
    
    # Silence the divide by zero warning from PyCochleagram plotting
    with np.errstate(divide='ignore'):
        for channel in range(2):
            coch = cgram.cochleagram(audio[:, channel], sr, n, low_lim, hi_lim, 1, 
                                     ret_mode='subband', strict=False)
            if coch.shape[0] > n_channels:
                start_idx = (coch.shape[0] - n_channels) // 2
                coch = coch[start_idx:start_idx + n_channels]
            cochleagrams.append(coch)
            
    coch_stereo = np.stack(cochleagrams, axis=2).astype(np.float32)
    
    downsample_filter, ratio = make_downsample_filter()
    downsampled = np.zeros((n_channels, coch_stereo.shape[1] // ratio, 2), dtype=np.float32)
    for ear in range(2):
        for ch in range(n_channels):
            filtered = np.convolve(coch_stereo[ch, :, ear], downsample_filter, mode='same')
            downsampled[ch, :, ear] = filtered[::ratio][:downsampled.shape[1]]
    
    downsampled = np.maximum(downsampled, 0)
    
    if downsampled.shape[1] < final_samples:
        pad_width = final_samples - downsampled.shape[1]
        downsampled = np.pad(downsampled, ((0, 0), (0, pad_width), (0, 0)))
    else:
        downsampled = downsampled[:, :final_samples, :]
        
    return downsampled

def build_tfrecords(data_dir, output_file_base):
    search_pattern = os.path.join(data_dir, "**", "*.wav")
    files = sorted(glob.glob(search_pattern, recursive=True))
    print(f"Found {len(files)} total wav files. Parsing metadata...")
    
    options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
    RECORDS_PER_SHARD = 500
    
    # Categorize and validate files
    valid_files = {0: [], 1: []}
    
    for f in files:
        filename = os.path.basename(f)
        az_match = re.search(r'az(-?\d+)', filename)
        el_match = re.search(r'el(-?\d+)', filename)
        
        if '0click' in filename:
            c_type = 0
        elif '1click' in filename:
            c_type = 1
        else:
            continue
            
        if not az_match:
            continue
            
        az = int(az_match.group(1))
        el = int(el_match.group(1)) if el_match else 0
            
        if el < 0:
            continue
            
        if az < 0: az += 360
        
        az_bin = az // 5
        el_bin = el // 10
        final_class = int(az_bin + (el_bin * 72))
        
        valid_files[c_type].append({
            'path': f,
            'az_bin': az_bin,
            'el_bin': el_bin,
            'final_class': final_class,
            'c_type': c_type
        })
        
    print(f"Validation complete. Found {len(valid_files[0])} 0click and {len(valid_files[1])} 1click valid files.")
    
    state_labels = {0: '0click', 1: '1click'}
    
    import random
    random.seed(42)  # Deterministic shuffle so we can recreate exact same shards if needed
    
    for c_type, label in state_labels.items():
        category_files = valid_files[c_type]
        random.shuffle(category_files)
        num_shards = (len(category_files) + RECORDS_PER_SHARD - 1) // RECORDS_PER_SHARD
        
        for shard_idx in range(num_shards):
            output_file = f"{output_file_base}_{label}-{shard_idx:05d}.tfrecords"
            tmp_file = output_file + ".tmp"
            
            if os.path.exists(output_file):
                print(f"Skipping {output_file} (already complete)...")
                continue
                
            print(f"--> Building shard: {output_file} ({min((shard_idx+1)*RECORDS_PER_SHARD, len(category_files))}/{len(category_files)} records)")
            shard_items = category_files[shard_idx * RECORDS_PER_SHARD : (shard_idx + 1) * RECORDS_PER_SHARD]
            
            try:
                with tf.io.TFRecordWriter(tmp_file, options=options) as writer:
                    for item in shard_items:
                        try:
                            audio, sr = load_wav_file(item['path'])
                            features = wav_to_model_input(audio, sr)
                            
                            feature = {
                                'train/azim': _int64_feature(item['az_bin']),
                                'train/elev': _int64_feature(item['el_bin']),
                                'train/class_num': _int64_feature(item['final_class']),
                                'train/image': _bytes_feature(tf.compat.as_bytes(features.tobytes())),
                                'train/image_height': _int64_feature(features.shape[0]),
                                'train/image_width': _int64_feature(features.shape[1]),
                                'train/click_type': _int64_feature(item['c_type'])
                            }
                            
                            example = tf.train.Example(features=tf.train.Features(feature=feature))
                            writer.write(example.SerializeToString())
                            
                        except Exception as e:
                            print(f"Error processing {item['path']}: {e}")
                
                # Commit the completed shard by stripping the .tmp extension
                os.rename(tmp_file, output_file)
                
            except Exception as e:
                print(f"Failed to build shard {output_file}: {e}")
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                    
    print("\nSuccess! Finished processing all shards.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Directory containing recursive wav files")
    parser.add_argument('--output_file', required=True, help="Base path for output TFRecords (e.g. data/finetune)")
    args = parser.parse_args()
    
    build_tfrecords(args.data_dir, args.output_file)

if __name__ == '__main__':
    main()
