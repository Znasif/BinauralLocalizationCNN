#!/usr/bin/env python
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.conversion import convert_to_tflite


def main():
    parser = argparse.ArgumentParser(
        description='Convert BinauralLocalizationCNN checkpoint to TFLite')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing checkpoint and config_array.npy')
    parser.add_argument('--output', default=None,
                        help='Output .tflite path (default: model_dir/model.tflite)')
    parser.add_argument('--config', default=None,
                        help='Path to config_array.npy (default: model_dir/config_array.npy)')
    args = parser.parse_args()

    model_dir   = os.path.abspath(args.model_dir)
    output_file = args.output or os.path.join(model_dir, 'model.tflite')
    config_path = os.path.abspath(args.config) if args.config else None

    try:
        convert_to_tflite(model_dir, output_file, config_path=config_path)
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
