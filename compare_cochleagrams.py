import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Import the identical functions used in our inference pipeline
from test_inference_minimal import load_tfrecord_sample, load_wav_file, wav_to_cochleagram, prepare_input_for_model

def plot_cochleagram(ax, coch, title):
    """Plot a single channel cochleagram."""
    # coch shape is [39, 8000]
    im = ax.imshow(coch, aspect='auto', origin='lower', cmap='inferno')
    ax.set_title(title)
    ax.set_ylabel("Frequency Channel")
    ax.set_xlabel("Time (samples)")
    plt.colorbar(im, ax=ax)

def main():
    parser = argparse.ArgumentParser(description="Compare original TFRecord cochleagram with new WAV cochleagram")
    parser.add_argument('--tfrecord', type=str, default="data/train0.tfrecords", help="Path to original TFRecord")
    parser.add_argument('--wav_file', type=str, required=True, help="Path to new finetuning WAV file")
    parser.add_argument('--output', type=str, default="cochleagram_comparison.png", help="Path to save the comparison plot")
    args = parser.parse_args()

    print(f"Loading original sample from {args.tfrecord} (index 0)...")
    # Load raw cochleagram from TFRecord
    # Returns [39, 8000, 2]
    orig_coch, orig_meta = load_tfrecord_sample(args.tfrecord, sample_index=0)
    
    # Run through the final prep stage just like the model sees it
    orig_input = prepare_input_for_model(orig_coch)[0] # Strip batch dim [39, 8000, 2]

    print(f"Loading new data from {args.wav_file}...")
    audio, sr = load_wav_file(args.wav_file)
    # Generate cochleagram using identical pipeline
    new_coch = wav_to_cochleagram(audio, sr=sr)
    # Run through the final prep stage (downsample, pad, relu) just like the model sees it
    new_input = prepare_input_for_model(new_coch)[0] # Strip batch dim [39, 8000, 2]

    # Print statistical comparisons
    print("\n--- Statistical Comparison ---")
    print(f"Original Shape: {orig_input.shape} | New Shape: {new_input.shape}")
    print(f"Original MIN: {orig_input.min():.5f} | MAX: {orig_input.max():.5f} | MEAN: {orig_input.mean():.5f}")
    print(f"New MIN:      {new_input.min():.5f} | MAX: {new_input.max():.5f} | MEAN: {new_input.mean():.5f}")

    # Plotting
    print(f"\nGenerating plots to {args.output}...")
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Top Row: Original Data
    plot_cochleagram(axs[0, 0], orig_input[:, :, 0], f"Original TFRecord - LEFT Ear\nMax: {orig_input[:, :, 0].max():.4f}")
    plot_cochleagram(axs[0, 1], orig_input[:, :, 1], f"Original TFRecord - RIGHT Ear\nMax: {orig_input[:, :, 1].max():.4f}")

    # Bottom Row: New Data
    plot_cochleagram(axs[1, 0], new_input[:, :, 0], f"New WAV Data - LEFT Ear\nMax: {new_input[:, :, 0].max():.4f}")
    plot_cochleagram(axs[1, 1], new_input[:, :, 1], f"New WAV Data - RIGHT Ear\nMax: {new_input[:, :, 1].max():.4f}")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print("Done!")

if __name__ == '__main__':
    main()
