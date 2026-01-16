#!/usr/bin/env python
"""
Simplified inference script for BinauralLocalizationCNN.
Runs testing on TFRecords data using a pretrained model.

Usage:
    python run_inference.py --model_dir models/net1 --data_pattern "data/train*.tfrecords"
"""

import os
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import sys
import glob
import json
import numpy as np
import tensorflow as tf

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tf_record_CNN_spherical_gradcheckpoint_valid_pad import tf_record_CNN_spherical


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run inference on TFRecords')
    parser.add_argument('--model_dir', required=True, help='Path to model directory (e.g., models/net1)')
    parser.add_argument('--data_pattern', required=True, help='Glob pattern for TFRecords (e.g., data/train*.tfrecords)')
    parser.add_argument('--model_version', type=int, default=100000, help='Checkpoint version')
    parser.add_argument('--snr', type=int, default=80, help='SNR for testing (dB)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference (default: 16)')
    args = parser.parse_args()

    # Resolve paths
    model_dir = os.path.abspath(args.model_dir)
    data_pattern = os.path.abspath(args.data_pattern) if not os.path.isabs(args.data_pattern) else args.data_pattern
    
    # For this simplified version, use data as both foreground and background
    bkgd_pattern = data_pattern
    
    # Check model files exist
    config_path = os.path.join(model_dir, 'config_array.npy')
    if not os.path.exists(config_path):
        print(f"ERROR: config_array.npy not found in {model_dir}")
        sys.exit(1)
    
    ckpt_path = os.path.join(model_dir, f'model.ckpt-{args.model_version}.index')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint model.ckpt-{args.model_version} not found in {model_dir}")
        sys.exit(1)
    
    # Check data files exist
    data_files = glob.glob(data_pattern)
    if not data_files:
        print(f"ERROR: No TFRecords found matching {data_pattern}")
        sys.exit(1)
    print(f"Found {len(data_files)} TFRecord files")
    
    # Load config
    config_array = np.load(config_path, allow_pickle=True)
    print(f"Loaded model config from {config_path}")
    
    # Get existing files in model dir (for num_files check)
    files = glob.glob(os.path.join(model_dir, '*'))
    files_filtered = [f for f in files if ".json" not in f]
    num_files = len(files_filtered)
    
    # Fixed parameters for testing
    tone_version = False
    itd_tones = False
    ild_tones = False
    branched = False
    zero_padded = True
    
    # Testing mode parameters
    testing = True
    manually_added = False
    freq_label = False
    sam_tones = False
    transposed_tones = False
    precedence_effect = False
    narrowband_noise = False
    stacked_channel = True  # The provided data uses stacked channel format
    all_positions_bkgd = False
    background_textures = False
    
    num_epochs = 1
    model_version = [args.model_version]
    regularizer = None
    SNR_max = args.snr
    SNR_min = args.snr
    
    # Dummy arch_ID (not used in testing)
    arch_ID = 0
    
    print(f"\n=== Starting Inference ===")
    print(f"Model: {model_dir}")
    print(f"Data: {data_pattern}")
    print(f"SNR: {args.snr} dB")
    print(f"Checkpoint: {args.model_version}")
    print(f"Batch Size: {args.batch_size}")
    print()
    
    # Run inference
    tf_record_CNN_spherical(
        tone_version, itd_tones, ild_tones, manually_added, freq_label,
        sam_tones, transposed_tones, precedence_effect, narrowband_noise,
        all_positions_bkgd, background_textures, testing, branched,
        zero_padded, stacked_channel, model_version, num_epochs,
        data_pattern, bkgd_pattern, arch_ID, config_array, files,
        num_files, model_dir, regularizer, SNR_max, SNR_min,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
