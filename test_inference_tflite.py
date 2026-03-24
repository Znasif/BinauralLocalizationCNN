#!/usr/bin/env python
"""
TFLite inference script.
Equivalent to test_inference_minimal.py but uses the .tflite model directly,
requiring only tflite_runtime (no full TensorFlow install needed).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_io import load_wav_file, wav_to_cochleagram, prepare_input_for_model
from src.inference import load_tflite_interpreter, run_tflite_inference_on_input
from src.visualization import decode_class_index


def run_tflite_inference(model_path, wav_path):
    print(f'Loading TFLite model: {model_path}')
    interpreter   = load_tflite_interpreter(model_path)
    input_details = interpreter.get_input_details()
    print(f'Model Input Shape: {input_details[0]["shape"]}')

    if wav_path:
        print(f'Processing audio: {wav_path}')
        audio, sr = load_wav_file(wav_path)
        coch = wav_to_cochleagram(audio, sr)
    else:
        print('Using dummy input (random noise)')
        coch = np.abs(np.random.randn(39, 48000, 2).astype(np.float32))

    input_data = prepare_input_for_model(coch)
    print(f'Input stats: min={np.min(input_data):.4f}  '
          f'max={np.max(input_data):.4f}  mean={np.mean(input_data):.4f}')
    if np.isnan(input_data).any():
        print('WARNING: Input data contains NaNs!')

    probabilities = run_tflite_inference_on_input(interpreter, input_data)
    top5_indices  = np.argsort(probabilities)[-5:][::-1]
    top5_probs    = probabilities[top5_indices]

    print('\n' + '=' * 50)
    print('TFLITE INFERENCE RESULTS')
    print('=' * 50)
    print('Top 5 Predictions:')
    print('-' * 50)
    for i, (pred_class, confidence) in enumerate(zip(top5_indices, top5_probs), 1):
        azim_deg, elev_deg = decode_class_index(pred_class)
        print(f'{i}. Class {pred_class}: '
              f'Azimuth={azim_deg}°, Elevation={elev_deg}°  '
              f'(confidence: {confidence:.4f})')
    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(description='Run inference using TFLite model')
    parser.add_argument('--model_file', required=True, help='Path to .tflite model file')
    parser.add_argument('--wav_file', default=None, help='Path to .wav file for inference')
    args = parser.parse_args()

    if not os.path.exists(args.model_file):
        print(f'Error: Model file not found at {args.model_file}')
        sys.exit(1)

    run_tflite_inference(args.model_file, args.wav_file)


if __name__ == '__main__':
    main()
