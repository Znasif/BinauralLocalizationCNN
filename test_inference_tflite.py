#!/usr/bin/env python
"""
Inference script using TFLite model.
Equivalent to test_inference_minimal.py but for .tflite models.
"""
import os
import sys
import argparse
import numpy as np
import scipy.signal as signallib
from scipy.io import wavfile

# Add current directory to path to find pycochleagram
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try importing from tflite_runtime first (lighter)
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        # Fallback to full tensorflow
        import tensorflow.lite as tflite
    except ImportError:
        try:
            # TF 1.x compat
            import tensorflow as tf
            tflite = tf.lite
        except:
            raise ImportError("Could not import tensorflow.lite or tflite_runtime. Please install one of them.")

# Import pycochleagram
try:
    from pycochleagram import erbfilter as erb
    from pycochleagram import subband as sb
    from pycochleagram import cochleagram as cgram
except ImportError:
    print("Error: pycochleagram not found. Make sure you are running this in the project root.")
    sys.exit(1)

# ==========================================
# Preprocessing Functions (Copied from test_inference_minimal.py)
# ==========================================

def load_wav_file(wav_path):
    """Load a stereo wav file and return the signals and sample rate."""
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
    """Convert a stereo wav audio signal to cochleagram format."""
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
            ret_mode='subband',
            strict=False
        )
        
        if coch.shape[0] > n_channels:
            start_idx = (coch.shape[0] - n_channels) // 2
            coch = coch[start_idx:start_idx + n_channels]
        
        cochleagrams.append(coch)
    
    cochleagram_stereo = np.stack(cochleagrams, axis=2)
    return cochleagram_stereo.astype(np.float32)

def make_downsample_filter(sr=48000, env_sr=8000, window_size=4097, beta=10.06):
    downsample_ratio = sr // env_sr
    downsample_filter_times = np.arange(-window_size/2, int(window_size/2))
    downsample_filter_response_orig = np.sinc(downsample_filter_times / downsample_ratio) / downsample_ratio
    
    # helper to get kaiser window
    if hasattr(signallib, 'windows') and hasattr(signallib.windows, 'kaiser'):
        kaiser_func = signallib.windows.kaiser
    elif hasattr(signallib, 'kaiser'):
        kaiser_func = signallib.kaiser
    else:
        # Fallback: try importing directly
        from scipy.signal.windows import kaiser as kaiser_func
        
    downsample_filter_window = kaiser_func(window_size, beta)
    downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    return downsample_filter_response.astype(np.float32), downsample_ratio

def downsample_cochleagram(cochleagram, sr=48000, target_sr=8000):
    downsample_filter, ratio = make_downsample_filter(sr, target_sr)
    n_channels, n_samples, n_ears = cochleagram.shape
    n_output_samples = n_samples // ratio
    
    downsampled = np.zeros((n_channels, n_output_samples, n_ears), dtype=np.float32)
    
    for ear in range(n_ears):
        for channel in range(n_channels):
            filtered = np.convolve(cochleagram[channel, :, ear], downsample_filter, mode='same')
            downsampled[channel, :, ear] = filtered[::ratio][:n_output_samples]

    # ReLU
    downsampled = np.maximum(downsampled, 0)
    return downsampled

def prepare_input_for_model(cochleagram, target_samples=8000):
    n_channels, n_samples, n_ears = cochleagram.shape
    
    if n_samples > target_samples * 2:
        print(f"Downsampling cochleagram from {n_samples} to ~{target_samples} samples...")
        cochleagram = downsample_cochleagram(cochleagram, sr=48000, target_sr=8000)
        n_samples = cochleagram.shape[1]
    
    if n_samples < target_samples:
        pad_width = target_samples - n_samples
        cochleagram = np.pad(cochleagram, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    elif n_samples > target_samples:
        cochleagram = cochleagram[:, :target_samples, :]
    
    # Ensure positivity (ReLU) - mandatory for x^0.3 nonlinearity
    # This acts as a safeguard if downsampling (which does ReLU) was skipped
    cochleagram = np.maximum(cochleagram, 0)

    # Add batch dimension: [1, 39, 8000, 2]
    model_input = cochleagram[np.newaxis, ...]
    return model_input.astype(np.float32)

# ==========================================
# Main Inference Logic
# ==========================================

def run_tflite_inference(model_path, wav_path):
    print(f"Loading TFLite model: {model_path}")
    
    # Load Interpreter
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
    except AttributeError:
        # Fallback for some TF versions
        interpreter = tflite.Interpreter(model_path=model_path)
        
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model Input Shape: {input_details[0]['shape']}")
    # print(f"Model Output Shape: {output_details[0]['shape']}")

    # Prepare Input
    if wav_path:
        print(f"Processing audio: {wav_path}")
        audio, sr = load_wav_file(wav_path)
        coch = wav_to_cochleagram(audio, sr)
    else:
        print("Using dummy input (random noise)")
        coch = np.abs(np.random.randn(39, 48000, 2).astype(np.float32))

    input_data = prepare_input_for_model(coch)
    
    print(f"Input Data Stats: Min={np.min(input_data)}, Max={np.max(input_data)}, Mean={np.mean(input_data)}")
    if np.isnan(input_data).any():
        print("WARNING: Input data contains NaNs!")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    print("Running inference...")
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0] # output shape is [1, 504] likely

    # Post-process results
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_probs = probabilities[top5_indices]

    print("\n" + "="*50)
    print("TFLITE INFERENCE RESULTS")
    print("="*50)
    print(f"Top 5 Predictions:")
    print("-" * 50)

    for i, (pred_class, confidence) in enumerate(zip(top5_indices, top5_probs), 1):
        # Decode class index (assuming same encoding as original: 72 azimuth bins * 7 elevation bins)
        # Azimuth bins: 5 degrees (0-355) -> 72 bins? Actually code says 0-175 in previous script...
        # Let's check test_inference_minimal.py again:
        # azim_idx = idx % 72
        # elev_idx = idx // 72
        # azim_deg = azim_idx * 5
        # elev_deg = elev_idx * 10
        
        azim_idx = pred_class % 72
        elev_idx = pred_class // 72
        azim_deg = azim_idx * 5
        elev_deg = elev_idx * 10
        
        print(f"{i}. Class {pred_class}: Azimuth={azim_deg}°, Elevation={elev_deg}° (confidence: {confidence:.4f})")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Run inference using TFLite model')
    parser.add_argument('--model_file', required=True, help='Path to .tflite model file')
    parser.add_argument('--wav_file', help='Path to .wav file for inference')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_file):
        print(f"Error: Model file not found at {args.model_file}")
        sys.exit(1)
        
    run_tflite_inference(args.model_file, args.wav_file)

if __name__ == '__main__':
    main()
