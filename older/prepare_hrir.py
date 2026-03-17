import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import argparse
import os

def prepare_hrir(input_path, output_path, target_sr=48000, target_duration=1.0):
    print(f"Processing {input_path}...")
    
    # 1. Load
    try:
        sr, data = wavfile.read(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Original: {sr} Hz, {data.shape}")

    # Ensure float
    if data.dtype != np.float32:
        # Normalize int to -1..1 float
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

    # 2. Resample if needed
    if sr != target_sr:
        print(f"Resampling {sr} -> {target_sr}...")
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
        
    # 3. Pad to Target Duration
    target_samples = int(target_sr * target_duration)
    current_samples = data.shape[0]
    
    if current_samples < target_samples:
        print(f"Padding {current_samples} -> {target_samples} samples...")
        pad_width = target_samples - current_samples
        # Pad at the end (post-impulse silence)
        # Assuming shape is [samples, channels]
        padding = np.zeros((pad_width, data.shape[1]), dtype=np.float32)
        data = np.vstack((data, padding))
    else:
        print("Signal is longer than target duration, cropping...")
        data = data[:target_samples, :]
        
    # 4. Save
    # Convert back to int16 for compatibility with some readers, or keep float
    # The inference script uses wavfile.read which handles float, but many external tools like int16
    data_int16 = (data * 32767).astype(np.int16)
    
    wavfile.write(output_path, target_sr, data_int16)
    print(f"Saved prepared file to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input HRIR WAV file')
    parser.add_argument('--output', help='Output WAV file (default: input_prepared.wav)')
    args = parser.parse_args()
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_prepared{ext}"
        
    prepare_hrir(args.input, args.output)
