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
    
    # Metrics accumulation
    class MetricsTracker:
        def __init__(self):
            self.total_samples = 0
            self.correct_top1 = 0
            self.correct_top5 = 0
            self.azimuth_errors = []
            self.elevation_errors = []
        
        def update(self, data):
            gt_az_idx = data['gt_az_idx']
            gt_el_idx = data['gt_el_idx']
            top_k_indices = data['top_k_indices']
            
            batch_size = len(gt_az_idx)
            self.total_samples += batch_size
            
            for i in range(batch_size):
                # Reconstruct ground truth class
                # Class = elev_idx * 36 + azim_idx
                true_class = gt_el_idx[i] * 36 + gt_az_idx[i]
                
                # Check Top-1
                pred_class = top_k_indices[i][0]
                if pred_class == true_class:
                    self.correct_top1 += 1
                
                # Check Top-5
                if true_class in top_k_indices[i]:
                    self.correct_top5 += 1
                
                # Calculate Angular Errors
                # Convert indices to degrees (Azimuth: 5deg steps, Eleavation: 10deg steps)
                true_az = gt_az_idx[i] * 5
                true_el = gt_el_idx[i] * 10 
                # Note: Code uses 10 degree steps for elevation in display, but `label_div_const` logic in tfrecords_iterator 
                # uses localization_bin_resolution. Run_inference sets `label_div_const` logic via arguments.
                # Assuming 10 degree elevation steps based on `test_inference_minimal.py` logic.
                
                pred_az_idx = pred_class % 36
                pred_el_idx = pred_class // 36
                pred_az = pred_az_idx * 5
                pred_el = pred_el_idx * 10  # Assuming 10 degrees? tf_record uses `tf.scalar_mul(tf.constant(36...), elev)`
                # 36 azimuth bins -> 180 degrees range / 5 degrees step = 36 bins.
                # Elevation? If 504 classes / 36 = 14 elevation bins.
                # 14 bins * 10 degrees = 140 degrees (0-130). Checks out.
                
                self.azimuth_errors.append(abs(pred_az - true_az))
                self.elevation_errors.append(abs(pred_el - true_el))
                
            # print(f"Processed {self.total_samples} samples...", end='\r')

        def print_summary(self):
            if self.total_samples == 0:
                print("No samples processed.")
                return
                
            accuracy = self.correct_top1 / self.total_samples * 100
            top5_accuracy = self.correct_top5 / self.total_samples * 100
            mean_az_error = np.mean(self.azimuth_errors)
            mean_el_error = np.mean(self.elevation_errors)
            
            print("\n" + "="*40)
            print(f"Inference Summary ({self.total_samples} samples)")
            print("="*40)
            print(f"Accuracy (Top-1):       {accuracy:.2f}%")
            print(f"Accuracy (Top-5):       {top5_accuracy:.2f}%")
            print(f"Mean Azimuth Error:     {mean_az_error:.2f} degrees")
            print(f"Mean Elevation Error:   {mean_el_error:.2f} degrees")
            print("="*40)

    tracker = MetricsTracker()

    # Run inference
    try:
        tf_record_CNN_spherical(
            tone_version, itd_tones, ild_tones, manually_added, freq_label,
            sam_tones, transposed_tones, precedence_effect, narrowband_noise,
            all_positions_bkgd, background_textures, testing, branched,
            zero_padded, stacked_channel, model_version, num_epochs,
            data_pattern, bkgd_pattern, arch_ID, config_array, files,
            num_files, model_dir, regularizer, SNR_max, SNR_min,
            batch_size=args.batch_size,
            batch_callback=tracker.update
        )
    except Exception as e:
        print(f"An error occurred during inference: {e}")
    finally:
        tracker.print_summary()


if __name__ == '__main__':
    main()
