#!/usr/bin/env python
"""
Minimal inference test — runs 1+ batches then exits.
Supports: --wav_file, --wav_folder, --tfrecord, --tfrecords_dir, or dummy data.
Forces CPU by default; use --use_gpu to enable GPU.
"""
import os
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import argparse
import shutil
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph_builder import load_config, build_inference_graph
from src.data_io       import generate_inputs, prepare_input_for_model
from src.inference     import make_session_config, restore_checkpoint, run_ckpt_inference
from src.visualization import (plot_prediction_distribution, plot_aggregate_errors,
                                decode_class_index)


def main():
    parser = argparse.ArgumentParser(
        description='Minimal inference test with wav or tfrecord input')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing model checkpoint and config_array.npy')
    parser.add_argument('--wav_file',      default='',
                        help='Path to stereo .wav file')
    parser.add_argument('--wav_folder',    default='',
                        help='Path to folder containing .wav files (recursive)')
    parser.add_argument('--tfrecords_dir', default='',
                        help='Path to directory containing *.tfrecords files '
                             '(evaluates up to 200 samples per shard)')
    parser.add_argument('--tfrecord',      default='',
                        help='Path to a single .tfrecords file')
    parser.add_argument('--sample_index',  type=int, default=0,
                        help='Sample index within tfrecord (default: 0)')
    parser.add_argument('--plot_output',   default='',
                        help='Directory to save per-sample prediction plots')
    parser.add_argument('--aggregate',     default='',
                        help='Path to save aggregate error-distribution plot')
    parser.add_argument('--use_gpu',       action='store_true', default=False,
                        help='Enable GPU (default: CPU-only)')
    parser.add_argument('--quiet',         action='store_true', default=False,
                        help='Suppress output; emit only PROGRESS: lines')
    parser.add_argument('--checkpoint',    default='',
                        help='Explicit checkpoint path '
                             '(default: model_dir/model.ckpt-100000)')
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)

    # ── Build graph ───────────────────────────────────────
    config_array = load_config(
        model_dir, target_device='/gpu:0' if args.use_gpu else '/cpu:0')
    if not args.quiet:
        print(f'Loaded config from {model_dir} '
              f'(device: {"GPU" if args.use_gpu else "CPU"})')

    print('Building graph...')
    input_placeholder, _, probabilities, predictions = build_inference_graph(
        config_array, batch_size=1, n_classes=504)
    print('Graph built.')

    # ── Session + restore ─────────────────────────────────
    sess = tf.compat.v1.Session(config=make_session_config(args.use_gpu))
    sess.run(tf.compat.v1.global_variables_initializer())
    restore_checkpoint(sess, model_dir,
                       checkpoint_path=args.checkpoint or None)

    # ── Input generator ───────────────────────────────────
    input_gen     = generate_inputs(args, model_dir)
    total_samples = 0

    first_item = next(input_gen, None)
    if first_item and isinstance(first_item[0], str) and first_item[0] == 'INIT_TOTAL':
        total_samples = first_item[1]
    elif first_item:
        from itertools import chain
        input_gen = chain([first_item], input_gen)

    if args.plot_output:
        shutil.rmtree(args.plot_output, ignore_errors=True)
        os.makedirs(args.plot_output, exist_ok=True)

    # Aggregate error storage
    agg_data = {
        '0click': {'top1_az_errs': [], 'top1_el_errs': [],
                   'top5_az_errs': [], 'top5_el_errs': []},
        '1click': {'top1_az_errs': [], 'top1_el_errs': [],
                   'top5_az_errs': [], 'top5_el_errs': []},
    }

    if not args.quiet:
        print('Running inference...')

    processed_count = 0
    for cochleagram, metadata, png_file in input_gen:
        processed_count += 1
        model_input = prepare_input_for_model(cochleagram)
        pred, prob  = run_ckpt_inference(
            sess, input_placeholder, predictions, probabilities, model_input)

        top5_indices = np.argsort(prob[0])[-5:][::-1]
        top5_probs   = prob[0][top5_indices]

        if args.quiet:
            if total_samples > 0:
                print(f'PROGRESS: {processed_count}/{total_samples}', flush=True)
        else:
            print(pred)
            print('\n' + '=' * 50)
            print('INFERENCE TEST SUCCESSFUL!')
            print('=' * 50)
            if 'azimuth' in metadata:
                print(f'Ground truth: Azimuth={metadata["azimuth"]}°, '
                      f'Elevation={metadata["elevation"]}°')
            print('\nTop 5 Predictions:')
            print('-' * 50)
            for i, (pred_class, confidence) in enumerate(zip(top5_indices, top5_probs), 1):
                azim_deg, elev_deg = decode_class_index(pred_class)
                print(f'{i}. Class {pred_class}: '
                      f'Azimuth={azim_deg}°, Elevation={elev_deg}°  '
                      f'(confidence: {confidence:.4f})')
            print('=' * 50)

        if args.plot_output:
            plot_output_path = os.path.join(args.plot_output, png_file) \
                if png_file else os.path.join(args.plot_output, f'sample_{processed_count}.png')
            plot_prediction_distribution(
                prob[0], ground_truth=metadata, output_path=plot_output_path)

        if args.aggregate:
            condition = metadata.get('condition',
                                     'tfrecord' if args.tfrecord else 'unknown')
            if condition not in agg_data:
                agg_data[condition] = {
                    'top1_az_errs': [], 'top1_el_errs': [],
                    'top5_az_errs': [], 'top5_el_errs': []}

            gt_azim = metadata['azimuth']
            gt_elev = metadata['elevation']

            # Top-1 signed error
            top1_az, top1_el = decode_class_index(top5_indices[0])
            agg_data[condition]['top1_az_errs'].append(
                ((top1_az - gt_azim + 180) % 360) - 180)
            agg_data[condition]['top1_el_errs'].append(top1_el - gt_elev)

            # Top-5: closest prediction by Euclidean distance in angle space
            best_az_err = best_el_err = best_dist = float('inf')
            for pc in top5_indices:
                pred_az, pred_el = decode_class_index(pc)
                err_az = ((pred_az - gt_azim + 180) % 360) - 180
                err_el = pred_el - gt_elev
                dist   = (err_az ** 2 + err_el ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_az_err, best_el_err = dist, err_az, err_el
            agg_data[condition]['top5_az_errs'].append(best_az_err)
            agg_data[condition]['top5_el_errs'].append(best_el_err)

    sess.close()

    if args.aggregate:
        plot_aggregate_errors(agg_data, args.aggregate)


if __name__ == '__main__':
    main()
