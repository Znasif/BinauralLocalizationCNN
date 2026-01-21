#!/usr/bin/env python
"""
Minimal inference test - runs 1 batch then exits.
Tests that model loading and inference work correctly.
Forces CPU execution.

Supports two input modes:
  1. --wav_file: Load a stereo .wav file and convert to cochleagram using pycochleagram
  2. --tfrecord: Extract cochleagram data from a tfrecord file (e.g., data/train0.tfrecords)
"""
import os
# Disable GPU completely before importing TensorFlow
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import numpy as np
import tensorflow as tf
import scipy.signal as signallib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pycochleagram for wav to cochleagram conversion
from pycochleagram import erbfilter as erb
from pycochleagram import subband as sb
from pycochleagram import cochleagram as cgram

# Import matplotlib for visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def plot_prediction_distribution(probabilities, ground_truth=None, output_path='prediction_plot.png'):
    """
    Visualize the model's prediction distribution over azimuth and elevation.
    
    Args:
        probabilities: Array of shape (504,) with class probabilities
        ground_truth: Optional dict with 'azimuth' and 'elevation' keys
        output_path: Path to save the plot
    """
    # Reshape probabilities to 2D grid: elevation (14) x azimuth (36)
    print(probabilities.shape)
    prob_grid = probabilities.reshape(7, 72)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Heatmap of full probability distribution
    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(prob_grid, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax1.set_ylabel('Elevation (degrees)', fontsize=12)
    ax1.set_title('Probability Distribution (Heatmap)', fontsize=14, fontweight='bold')
    
    # Set ticks
    azim_ticks = np.arange(0, 72, 6)
    elev_ticks = np.arange(0, 7, 2)
    ax1.set_xticks(azim_ticks)
    ax1.set_xticklabels([f'{i*5}°' for i in azim_ticks])  # 5-degree bins for azimuth
    ax1.set_yticks(elev_ticks)
    ax1.set_yticklabels([f'{i*10}°' for i in elev_ticks])
    
    # Add ground truth marker if available
    if ground_truth and 'azimuth' in ground_truth:
        gt_azim_idx = ground_truth['azimuth'] / 5.0
        gt_elev_idx = ground_truth['elevation'] / 10.0
        ax1.plot(gt_azim_idx, gt_elev_idx, 'r*', markersize=20,
                label=f'Ground Truth ({ground_truth["azimuth"]}°, {ground_truth["elevation"]}°)')
        ax1.legend(loc='upper right')
    
    plt.colorbar(im, ax=ax1, label='Probability')
    
    # 2. Marginal distribution over azimuth
    ax2 = plt.subplot(2, 2, 2)
    azim_probs = np.sum(prob_grid, axis=0)
    
    azim_angles = np.arange(0, 360, 5)  # 5-degree bins, 36 bins total (0-175°)
    
    ax2.bar(azim_angles, azim_probs, width=4, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax2.set_ylabel('Marginal Probability', fontsize=12)
    ax2.set_title('Azimuth Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlim(-5, 360)
    ax2.grid(axis='y', alpha=0.3)

    if ground_truth and 'azimuth' in ground_truth:
        gt_azim = ground_truth['azimuth']
        ax2.axvline(gt_azim, color='red', linestyle='--', linewidth=2, label='Ground Truth')
        ax2.legend()
    
    # 3. Marginal distribution over elevation
    ax3 = plt.subplot(2, 2, 3)
    elev_probs = np.sum(prob_grid, axis=1)
    elev_angles = np.arange(0, 70, 10)
    ax3.barh(elev_angles, elev_probs, height=8, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Elevation (degrees)', fontsize=12)
    ax3.set_xlabel('Marginal Probability', fontsize=12)
    ax3.set_title('Elevation Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylim(-5, 145)
    ax3.grid(axis='x', alpha=0.3)
    
    if ground_truth and 'elevation' in ground_truth:
        gt_elev = ground_truth['elevation']
        ax3.axhline(gt_elev, color='red', linestyle='--', linewidth=2, label='Ground Truth')
        ax3.legend()
    
    # 4. Top predictions table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Get top 10 predictions
    top_indices = np.argsort(probabilities)[-10:][::-1]
    top_probs = probabilities[top_indices]
    
    # Create table data
    table_data = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        azim_idx = idx % 72
        elev_idx = idx // 72
        azim_deg = azim_idx * 5  # 5-degree bins
        elev_deg = elev_idx * 10  # 10-degree bins

        # Check if this matches ground truth
        marker = ''
        if ground_truth and 'azimuth' in ground_truth:
            if azim_idx == ground_truth['azimuth'] and elev_idx == ground_truth['elevation']:
                marker = '✓'

        table_data.append([f'{i}', f'{azim_deg}°', f'{elev_deg}°', f'{prob:.4f}', marker])
    
    # Create table
    table = ax4.table(cellText=table_data,
                      colLabels=['Rank', 'Azimuth', 'Elevation', 'Prob.', ''],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.15, 0.22, 0.22, 0.22, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight ground truth row if present
    if ground_truth and 'azimuth' in ground_truth:
        for i, row in enumerate(table_data, 1):
            if row[4] == '✓':
                for j in range(5):
                    table[(i, j)].set_facecolor('#FFEB3B')
    
    ax4.set_title('Top 10 Predictions', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def load_wav_file(wav_path):
    """Load a stereo wav file and return the signals and sample rate."""
    from scipy.io import wavfile
    sr, audio = wavfile.read(wav_path)
    
    if audio.ndim == 1:
        # Mono file - duplicate to stereo
        audio = np.stack([audio, audio], axis=1)
    elif audio.ndim == 2 and audio.shape[1] > 2:
        # Might be transposed
        audio = audio.T
    
    # Normalize to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    return audio, sr


def wav_to_cochleagram(audio, sr, target_sr=48000, n_channels=39, low_lim=30, hi_lim=20000):
    """
    Convert a stereo wav audio signal to cochleagram format expected by the model.
    
    Args:
        audio: Stereo audio array of shape (n_samples, 2)
        sr: Sample rate of the audio
        target_sr: Target sample rate (model expects 48000 Hz)
        n_channels: Number of frequency channels (model expects 39)
        low_lim: Lower frequency limit for ERB filterbank
        hi_lim: Upper frequency limit for ERB filterbank
    
    Returns:
        cochleagram: Array of shape (39, n_samples, 2) - stereo cochleagram
    """
    # Resample if necessary
    if sr != target_sr:
        from scipy import signal
        n_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, n_samples)
        sr = target_sr
    
    # Calculate number of filters based on ERB spacing
    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
    sample_factor = 1
    
    # Process each channel separately
    cochleagrams = []
    for channel in range(2):
        signal_data = audio[:, channel]
        
        # Generate cochleagram using pycochleagram
        # Use 'subband' mode to get subbands (not just envelopes)
        coch = cgram.cochleagram(
            signal_data, 
            sr, 
            n, 
            low_lim, 
            hi_lim, 
            sample_factor,
            padding_size=None,
            downsample=None,
            nonlinearity=None,
            fft_mode='np',
            ret_mode='subband',  # Get subbands (filtered waveforms) not envelopes
            strict=False
        )
        
        # Remove edge filters to get exactly n_channels
        # The filterbank produces n+2 filters (with high/low pass edge filters)
        if coch.shape[0] > n_channels:
            # Remove edge filters to match expected channels
            start_idx = (coch.shape[0] - n_channels) // 2
            coch = coch[start_idx:start_idx + n_channels]
        
        cochleagrams.append(coch)
    
    # Stack as (n_channels, n_samples, 2)
    cochleagram_stereo = np.stack(cochleagrams, axis=2)
    
    return cochleagram_stereo.astype(np.float32)


def load_tfrecord_sample(tfrecord_path, sample_index=None):
    """
    Load a sample cochleagram from a tfrecord file.
    
    Args:
        tfrecord_path: Path to the .tfrecords file
        sample_index: Which sample to extract (0-based)
    
    Returns:
        cochleagram: Array of shape (39, n_samples, 2)
        metadata: Dictionary with azimuth, elevation, etc.
    """
    options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    count = 0
    for serialized_example in tf.python_io.tf_record_iterator(tfrecord_path, options=options):
        if count==sample_index:
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            features = example.features.feature
            
            # Extract the image (cochleagram) data
            if 'train/image' in features:
                image_bytes = features['train/image'].bytes_list.value[0]
                # Decode as float32 or float64
                try:
                    image_data = np.frombuffer(image_bytes, dtype=np.float32)
                except:
                    image_data = np.frombuffer(image_bytes, dtype=np.float64).astype(np.float32)
                
                # Try to reshape - the tfrecord stores data in different formats
                # Based on tfrecords_iterator.py, stacked_channel format is [39, 48000, 2]
                try:
                    # Try stacked channel format first
                    cochleagram = image_data.reshape(39, 48000, 2)
                except ValueError:
                    # Try non-stacked format [78, 48000] then convert
                    try:
                        cochleagram = image_data.reshape(78, 48000)
                        # Convert to stacked format
                        cochleagram = np.stack([cochleagram[:39], cochleagram[39:]], axis=2)
                    except ValueError:
                        # Try other possible shapes
                        total_elements = len(image_data)
                        print(f"Total elements in image: {total_elements}")
                        raise ValueError(f"Cannot reshape image data with {total_elements} elements")
            else:
                raise ValueError("No 'train/image' found in tfrecord")
            
            # Extract metadata
            # Note: tfrecords store values in degrees (e.g., 175 for azimuth, 60 for elevation)
            # Model uses 5-degree bins for azimuth (0-35) and 10-degree bins for elevation (0-13)
            metadata = {}
            if 'train/azim' in features:
                azim_degrees = features['train/azim'].int64_list.value[0]
                # Convert to index in 5-degree bins (0-35 for azimuth: 0°, 5°, 10°, ..., 175°)
                metadata['azimuth'] = azim_degrees
            if 'train/elev' in features:
                elev_degrees = features['train/elev'].int64_list.value[0]
                # Convert to index in 10-degree bins (0-13 for elevation)
                metadata['elevation'] = elev_degrees
            
            return cochleagram, metadata
        
        count += 1
    
    raise ValueError(f"Sample index {sample_index} not found in tfrecord (only {count} samples)")


def make_downsample_filter(sr=48000, env_sr=8000, window_size=4097, beta=10.06):
    """
    Create a sinc-based lowpass filter for downsampling cochleagram.
    Matches the filter used in training.
    """
    downsample_ratio = sr // env_sr
    downsample_filter_times = np.arange(-window_size/2, int(window_size/2))
    downsample_filter_response_orig = np.sinc(downsample_filter_times / downsample_ratio) / downsample_ratio
    downsample_filter_window = signallib.kaiser(window_size, beta)
    downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    return downsample_filter_response.astype(np.float32), downsample_ratio


def downsample_cochleagram(cochleagram, sr=48000, target_sr=8000):
    """
    Downsample cochleagram from sr to target_sr.
    
    Args:
        cochleagram: Shape (39, n_samples, 2)
        sr: Current sample rate
        target_sr: Target sample rate
    
    Returns:
        downsampled: Shape (39, n_samples // ratio, 2)
    """
    downsample_filter, ratio = make_downsample_filter(sr, target_sr)
    
    # Apply filter and downsample each channel
    n_channels, n_samples, n_ears = cochleagram.shape
    n_output_samples = n_samples // ratio
    
    downsampled = np.zeros((n_channels, n_output_samples, n_ears), dtype=np.float32)
    
    for ear in range(n_ears):
        for channel in range(n_channels):
            # Apply lowpass filter
            filtered = np.convolve(cochleagram[channel, :, ear], downsample_filter, mode='same')
            # Downsample
            downsampled[channel, :, ear] = filtered[::ratio][:n_output_samples]

    # Apply ReLU (half-wave rectification) to match training
    # This is required before the power-law nonlinearity (x^0.3) which doesn't work with negatives
    downsampled = np.maximum(downsampled, 0)

    return downsampled


def prepare_input_for_model(cochleagram, target_samples=8000):
    """
    Prepare cochleagram input for the model.
    
    Args:
        cochleagram: Shape (39, n_samples, 2) - could be at 48kHz or already 8kHz
        target_samples: Expected number of samples after downsampling
    
    Returns:
        model_input: Shape (1, 39, 8000, 2) ready for inference
    """
    n_channels, n_samples, n_ears = cochleagram.shape
    
    # Check if we need to downsample
    if n_samples > target_samples * 2:
        # Assume 48kHz, need to downsample to 8kHz
        print(f"Downsampling cochleagram from {n_samples} to ~{target_samples} samples...")
        cochleagram = downsample_cochleagram(cochleagram, sr=48000, target_sr=8000)
        n_samples = cochleagram.shape[1]
    
    # Pad or truncate to target_samples
    if n_samples < target_samples:
        # Pad with zeros
        pad_width = target_samples - n_samples
        cochleagram = np.pad(cochleagram, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    elif n_samples > target_samples:
        # Truncate
        cochleagram = cochleagram[:, :target_samples, :]
    
    # Add batch dimension
    model_input = cochleagram[np.newaxis, ...]
    
    return model_input.astype(np.float32)


def generate_inputs(args, model_dir):
    """Generator that yields (cochleagram, metadata, png_file) for each input sample.

    Preserves original behavior: single wav file, tfrecord (single/indexed/bulk), wav folder,
    or dummy data when nothing provided.
    """
    # Single wav file
    if args.wav_file:
        print(f"Loading wav file: {args.wav_file}")
        png_file = args.wav_file.split("/")[-1].split(".")[0] + ".png"
        audio, sr = load_wav_file(args.wav_file)
        print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
        print("Converting wav to cochleagram using pycochleagram...")
        coch = wav_to_cochleagram(audio, sr)
        print(f"Cochleagram shape: {coch.shape}")
        metadata = {'azimuth': -1000, 'elevation': 0}
        yield coch, metadata, png_file
        return

    # TFRecord input
    if args.tfrecord:
        # If a specific sample index was provided
        if args.sample_index:
            print(f"Loading from tfrecord: {args.tfrecord}, sample {args.sample_index}")
            coch, metadata = load_tfrecord_sample(args.tfrecord, args.sample_index)
            png_file = ""
            yield coch, metadata, png_file
            return
        # Otherwise iterate through up to 200 samples
        for i in range(200):
            print(f"Loading from tfrecord: {args.tfrecord}, sample {i}")
            coch, metadata = load_tfrecord_sample(args.tfrecord, i)
            png_file = ""
            yield coch, metadata, png_file
        return

    # WAV folder
    if args.wav_folder:
        print(f"Loading wav files from folder: {args.wav_folder}")
        for filename in os.listdir(args.wav_folder):
            if filename.endswith(".wav"):
                file_path = os.path.join(args.wav_folder, filename)
                print(f"Loading wav file: {file_path}")
                audio, sr = load_wav_file(file_path)
                coch = wav_to_cochleagram(audio, sr)
                az = filename.split("az")[1].split("_")[0]
                az = int(az)
                if az < 0:
                    az += 360
                metadata = {'azimuth': az, 'elevation': 0}
                png_file = os.path.splitext(filename)[0] + ".png"
                yield coch, metadata, png_file
        return

    # Default: dummy input
    print("No input specified, using random dummy data for testing...")
    coch = np.abs(np.random.randn(39, 48000, 2).astype(np.float32))
    metadata = {'source': 'dummy'}
    png_file = ""
    yield coch, metadata, png_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Minimal inference test with wav or tfrecord input')
    parser.add_argument('--model_dir', required=True, help='Directory containing model checkpoint')
    parser.add_argument('--wav_file', default='', help='Path to stereo .wav file for inference')
    parser.add_argument('--wav_folder', default='', help='Path to folder containing .wav files for inference')
    parser.add_argument('--tfrecord', default='', help='Path to .tfrecords file')
    parser.add_argument('--sample_index', type=int, default=0, help='Sample index in tfrecord (default: 0)')
    parser.add_argument('--plot_output', default='', help='Path to save visualization plot (default: model_dir/inference_prediction.png)')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Enable GPU usage (default: False)')
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    # Use generator to produce (cochleagram, metadata, png_file)
    input_gen = generate_inputs(args, model_dir)
    
    # Load config and modify GPU references to CPU
    config_array = np.load(os.path.join(model_dir, 'config_array.npy'), allow_pickle=True)
    
    # Remap devices based on user preference
    def remap_devices(arr, use_gpu):
        """Recursively remap devices in config_array"""
        target_device = '/gpu:0' if use_gpu else '/cpu:0'
        
        if isinstance(arr, np.ndarray):
            return np.array([remap_devices(x, use_gpu) for x in arr], dtype=object)
        elif isinstance(arr, list):
            return [remap_devices(x, use_gpu) for x in arr]
        elif isinstance(arr, str) and '/gpu' in arr.lower():
            return target_device
        return arr
    
    config_array = remap_devices(config_array, args.use_gpu)
    device_mode = "GPU" if args.use_gpu else "CPU"
    print(f"Loaded config from {model_dir} (remapped to {device_mode})")
    
    from NetBuilder_valid_pad import NetBuilder
    
    n_classes_localization = 504
    batch_size = 1
    
    print("Building graph...")
    
    # Placeholder for input (after downsampling: 8000 samples)
    input_placeholder = tf.compat.v1.placeholder(tf.float32, [batch_size, 39, 8000, 2], name='input')
    nonlin = tf.pow(input_placeholder, 0.3)
    
    net = NetBuilder()
    out = net.build(config_array, nonlin, 
                   training_state=False,
                   dropout_training_state=False,
                   filter_dtype=tf.float32,
                   padding='VALID',
                   n_classes_localization=n_classes_localization,
                   n_classes_recognition=780,
                   branched=False,
                   regularizer=None)
    
    predictions = tf.argmax(out, axis=1)
    probabilities = tf.nn.softmax(out)
    
    print("Graph built. Loading checkpoint...")
    
    # Session config
    if args.use_gpu:
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
    else:
        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=True
        )
        
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, os.path.join(model_dir, 'model.ckpt-100000'))
    print("Checkpoint loaded!")

    if args.plot_output:
        import shutil
        shutil.rmtree(args.plot_output)
        os.makedirs(args.plot_output)
    
    print("Running inference...")
    for cochleagram, metadata, png_file in input_gen:
        model_input = prepare_input_for_model(cochleagram)
        pred, prob = sess.run([predictions, probabilities], feed_dict={input_placeholder: model_input})
        
        print(pred)
        # Get top 5 predictions
        top5_indices = np.argsort(prob[0])[-5:][::-1]
        top5_probs = prob[0][top5_indices]
        
        print("\n" + "="*50)
        print("INFERENCE TEST SUCCESSFUL!")
        print("="*50)
        if 'azimuth' in metadata:
            print(f"Ground truth: Azimuth={metadata['azimuth']}°, Elevation={metadata['elevation']}°")
        print(f"\nTop 5 Predictions:")
        print("-" * 50)

        for i, (pred_class, confidence) in enumerate(zip(top5_indices, top5_probs), 1):
            azim_idx = pred_class % 72
            elev_idx = pred_class // 72
            azim_deg = azim_idx*5  # 5-degree bins for azimuth
            elev_deg = elev_idx*10  # 10-degree bins for elevation
            print(f"{i}. Class {pred_class}: Azimuth={azim_deg}°, Elevation={elev_deg}° (confidence: {confidence:.4f})")
        
        print("="*50)
        
        # Generate visualization
        if args.plot_output:
            plot_output_path = args.plot_output+str(metadata['azimuth'])+".png"
        else:
            plot_output_path = png_file
        
        plot_prediction_distribution(prob[0], ground_truth=metadata, output_path=plot_output_path)
        
    sess.close()

if __name__ == '__main__':
    main()
