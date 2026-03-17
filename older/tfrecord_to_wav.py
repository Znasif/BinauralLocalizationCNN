#!/usr/bin/env python3
"""
Convert TFRecords containing cochleagrams back to WAV files.
Reconstructs the waveform from the subbands using pycochleagram.
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

# Import pycochleagram components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycochleagram import erbfilter as erb
from pycochleagram import subband as sb

def parse_args():
    parser = argparse.ArgumentParser(description='Convert TFRecord cochleagrams to WAV')
    parser.add_argument('--tfrecord', required=True, help='Path to input .tfrecords file')
    parser.add_argument('--output', required=True, help='Path to output .wav file')
    parser.add_argument('--index', type=int, default=0, help='Index of sample to extract (default: 0)')
    parser.add_argument('--sr', type=int, default=48000, help='Sampling rate (default: 48000)')
    parser.add_argument('--low_lim', type=int, default=30, help='Low frequency limit (default: 30)')
    parser.add_argument('--hi_lim', type=int, default=20000, help='High frequency limit (default: 20000)')
    return parser.parse_args()

def load_tfrecord_sample(tfrecord_path, sample_index):
    """Load cochleagram subbands from TFRecord."""
    print(f"Loading sample {sample_index} from {tfrecord_path}...")
    
    # Use compat.v1 patterns to match environment
    options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    
    iterator = tf.python_io.tf_record_iterator(tfrecord_path, options=options)
    
    for i, serialized_example in enumerate(iterator):
        if i == sample_index:
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            features = example.features.feature
            
            if 'train/image' not in features:
                raise ValueError("Feature 'train/image' not found in TFRecord")
                
            image_bytes = features['train/image'].bytes_list.value[0]
            
            try:
                # Try float32 first
                data = np.frombuffer(image_bytes, dtype=np.float32)
            except:
                 data = np.frombuffer(image_bytes, dtype=np.float64).astype(np.float32)
            
            # Reshape logic matching inference script
            try:
                # Expecting [channels, time, ears] usually [39, 48000, 2]
                cochleagram = data.reshape(39, 48000, 2)
            except ValueError:
                try:
                    # Try stacked [78, 48000] and unstack
                    flat = data.reshape(78, 48000)
                    cochleagram = np.stack([flat[:39], flat[39:]], axis=2)
                except ValueError as e:
                    print(f"Data shape: {data.shape}")
                    raise ValueError(f"Could not reshape data to cochleagram: {e}")
            
            # Print metadata if available
            meta = {}
            if 'train/azim' in features: meta['Azimuth'] = features['train/azim'].int64_list.value[0]
            if 'train/elev' in features: meta['Elevation'] = features['train/elev'].int64_list.value[0]
            print(f"Found metadata: {meta}")
            
            return cochleagram
            
    raise ValueError(f"Index {sample_index} out of bounds")

def reconstruct_waveform(cochleagram, sr, low_lim, hi_lim):
    """
    Collapse subbands back to waveform.
    cochleagram: [n_channels, n_samples, n_ears]
    """
    n_channels, n_samples, n_ears = cochleagram.shape
    print(f"Reconstructing from shape {cochleagram.shape}...")
    
    # Generate the filterbank used to create these subbands
    # Note: The stored cochleagram likely has removed edge filters.
    # pycochleagram creates N + 2/4 filters usually (including low/high pass).
    # We need to recreate the exact filters and select the ones corresponding to the 39 channels.
    
    # Estimate n based on low/hi limits
    # n in generation was floor(erb(hi) - erb(lo)) - 1
    # For 30-20000Hz, freq2erb(30)=~0.8, freq2erb(20000)=~44.
    # Delta ~ 43. n ~ 42?
    # Actually, BinauralDataGen.txt creates cochleagram with n=None (auto).
    # pycochleagram default n calc: floor(erb(hi)-erb(lo)) - 1.
    
    # Let's try to regenerate filters with strict=False first to see what we get
    # But wait, collapse_subbands needs the exact filters.
    
    # We'll use cgm's helper logic.
    n_est = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
    print(f"Estimated n filters: {n_est}")
    
    # pycochleagram usually produces more filters than n.
    # BinauralDataGen uses 'human_cochleagram' with defaults (sample_factor=2).
    # Then it does:
    # subbands[:,:,0] = subbands_l[sample_factor:-sample_factor]
    # This means it strips the first 'sample_factor' and last 'sample_factor' filters.
    
    sample_factor = 1 # Matched to BinauralDataGen.txt (line 653)
    
    # Generate full filterbank
    # We need the signal length.
    # Note: Original signal processing used padding, but filters are defined by n_samples.
    # The reconstruction needs filters matching the current length.
    
    # filters must be [n_filters, n_freqs] for collapse_subbands (which uses full FFT)
    
    filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(
        n_samples,
        sr,
        n_est,
        low_lim,
        hi_lim,
        sample_factor,
        padding_size=None,
        full_filter=True, # Must use full filters (transposed in make_erb) for reconstruction
        strict=False
    )
    
    print(f"Generated filterbank shape: {filts.shape}")
    
    # The stored cochleagram has 39 channels.
    # BinauralDataGen slices [sample_factor : -sample_factor].
    
    expected_slice_start = sample_factor
    expected_slice_end = -sample_factor 
    
    if expected_slice_end == 0: # Handle -0 case if sample_factor was 0 (impossible but safe)
         expected_slice_end = None

    # Check if the sliced shape matches
    sliced_filts = filts[expected_slice_start:expected_slice_end]
    
    if sliced_filts.shape[0] == n_channels:
        print(f"Matched channel count {n_channels} with slice [{expected_slice_start}:{expected_slice_end}]")
        use_filts = sliced_filts
    else:
        print(f"WARNING: Filter count mismatch. Generated {filts.shape[0]}, Sliced {sliced_filts.shape[0]}, Expected {n_channels}.")
        # Fallback: Try to find a symmetric slice that matches
        diff = filts.shape[0] - n_channels
        if diff >= 0 and diff % 2 == 0:
            start = diff // 2
            end = - (diff // 2) if (diff // 2) > 0 else None
            use_filts = filts[start:end]
            print(f"Adjusted slice to [{start}:{end}]")
        else:
             print("Cannot match filter count exactly. Using first N filters (May be incorrect).")
             use_filts = filts[:n_channels]

    # Reconstruct separately for each ear
    reconstructed = []
    for ear in range(n_ears):
        # [Channels, Time]
        subbands = cochleagram[:, :, ear]
        
        # Collapse
        # Note: collapse_subbands expects full filterbank matching subband rows.
        # It performs: filters * fft(subbands).
        # We need to make sure 'use_filts' corresponds exactly to 'subbands'
        
        signal = sb.collapse_subbands(subbands, use_filts)
        reconstructed.append(signal)
        
    # Stack [Time, Ears]
    waveform = np.stack(reconstructed, axis=1)
    
    # Normalize for wav saving
    # Find max to avoid clipping, but maybe scaling is needed
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.9
        
    return waveform

def main():
    args = parse_args()
    
    cochleagram = load_tfrecord_sample(args.tfrecord, args.index)
    waveform = reconstruct_waveform(cochleagram, args.sr, args.low_lim, args.hi_lim)
    
    print(f"Saving to {args.output}...")
    wavfile.write(args.output, args.sr, waveform)
    print("Done.")

if __name__ == '__main__':
    main()
