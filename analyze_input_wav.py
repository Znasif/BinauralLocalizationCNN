#!/usr/bin/env python3
import scipy.io.wavfile as wavfile
import numpy as np
import argparse
import sys

def analyze_wav(filepath):
    print(f"Analyzing {filepath}...")
    try:
        sr, data = wavfile.read(filepath)
    except Exception as e:
        print(f"Error reading WAV: {e}")
        return

    # Normalize stats
    duration = data.shape[0] / sr
    n_channels = 1 if len(data.shape) == 1 else data.shape[1]
    
    print("\n--- Properties ---")
    print(f"Sampling Rate: {sr} Hz")
    print(f"Duration:      {duration:.6f} s")
    print(f"Samples:       {data.shape[0]}")
    print(f"Channels:      {n_channels}")
    print(f"Data Type:     {data.dtype}")
    print(f"Min Value:     {data.min()}")
    print(f"Max Value:     {data.max()}")
    
    # Analyze Content
    if n_channels > 1:
        rms_l = np.sqrt(np.mean(data[:, 0].astype(float)**2))
        rms_r = np.sqrt(np.mean(data[:, 1].astype(float)**2))
        print(f"RMS (L):       {rms_l:.4f}")
        print(f"RMS (R):       {rms_r:.4f}")
    else:
        rms = np.sqrt(np.mean(data.astype(float)**2))
        print(f"RMS:           {rms:.4f}")

    print("\n--- Comparison with Training Baseline ---")
    
    # 1. Sampling Rate
    TRAIN_SR = 48000
    if sr != TRAIN_SR:
        print(f"[!] SR Mismatch: File is {sr} Hz, Model expects {TRAIN_SR} Hz.")
        print(f"    -> You MUST resample this file to {TRAIN_SR} Hz.")
    else:
        print(f"[OK] SR matches {TRAIN_SR} Hz.")

    # 2. Duration
    # Training data length varies but cochleagram windows need some minimum size
    MIN_DURATION = 0.05 # arbitrary safety for filters
    if duration < MIN_DURATION:
        print(f"[!] Very Short Duration ({duration:.4f}s).")
        print(f"    -> This looks like an Impulse Response (HRIR) rather than a full audio scene.")
        print(f"    -> The cochleagram filters (esp. low freq) may have minimal support.")
        print("    -> 30Hz cycle = ~33ms. Your signal is 16ms. Low freqs will be garbage.")
    else:
        print(f"[OK] Duration seems sufficient ({duration:.2f}s).")

    # 3. Type
    if "int" in str(data.dtype):
        print(f"[i] Format is Integer PCM. Model preprocessing likely converts to Float32.")
    
    print("\n--- Recommendation ---")
    if duration < 0.02 and sr > 48000:
        print("This file appears to be a raw HRIR (Head Related Impulse Response) recorded at 96kHz.")
        print("To use this with the BinauralLocalizationCNN:")
        print("1. Convolve this HRIR with a larger audio signal (e.g. speech/noise).")
        print("2. Or, strictly pad it with zeros to at least 1-2 seconds (if testing pure impulse localization).")
        print("3. Resample to 48kHz.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wav_file', help='Path to wav file')
    args = parser.parse_args()
    
    analyze_wav(args.wav_file)
