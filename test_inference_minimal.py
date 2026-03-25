#!/usr/bin/env python
"""
Minimal inference test — runs 1+ batches then exits.
Supports: --wav_file, --wav_folder, --tfrecord, --tfrecords_dir, or dummy data.
Forces CPU by default; use --use_gpu to enable GPU.

Optional analysis flags:
  --activation_output DIR   Save per-layer activation maps (first sample only)
  --tuning_output DIR       Accumulate class-conditional tuning maps across all samples
  --generate_report         Also write an HTML report after --tuning_output
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
                                plot_activation_maps, decode_class_index)
from src.tuning import (
    DEFAULT_LAYERS,
    build_layer_tensor_map, list_available_layers,
    make_accumulators, reduce_activation, accumulate,
    compute_tuning_maps, compute_selectivity,
    save_npz, save_all_plots, save_topk_mosaics,
    prepare_spatial_cache_for_render, print_coverage,
    _is_conv,
)


def main():
    parser = argparse.ArgumentParser(
        description='Minimal inference test with wav or tfrecord input')

    # ── Input ─────────────────────────────────────────────
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing model checkpoint and config_array.npy')
    parser.add_argument('--wav_file',      default='')
    parser.add_argument('--wav_folder',    default='')
    parser.add_argument('--tfrecords_dir', default='',
                        help='Directory of *.tfrecords files (up to 200 samples/shard)')
    parser.add_argument('--tfrecord',      default='',
                        help='Path to a single .tfrecords file')
    parser.add_argument('--sample_index',  type=int, default=0)
    parser.add_argument('--checkpoint',    default='',
                        help='Explicit checkpoint path (default: model_dir/model.ckpt-100000)')
    parser.add_argument('--use_gpu',       action='store_true', default=False)

    # ── Per-sample output ──────────────────────────────────
    parser.add_argument('--plot_output',   default='',
                        help='Directory to save per-sample prediction plots')
    parser.add_argument('--aggregate',     default='',
                        help='Path to save aggregate error-distribution plot')
    parser.add_argument('--quiet',         action='store_true', default=False)

    # ── Activation maps (first sample only) ───────────────
    parser.add_argument('--activation_output', default='',
                        help='Directory to save intermediate layer activation maps '
                             '(runs on first sample only)')

    # ── Tuning analysis (all samples) ─────────────────────
    parser.add_argument('--tuning_output', default='',
                        help='Directory to save class-conditional tuning maps '
                             'accumulated across all samples')
    parser.add_argument('--layers', nargs='+', default=DEFAULT_LAYERS,
                        help='Layers to analyse for --tuning_output '
                             '(friendly names: relu_N, pool_N)')
    parser.add_argument('--spatial_reduction', default='max',
                        choices=['max', 'mean'],
                        help='How to reduce H×W spatial dims for conv layers')
    parser.add_argument('--max_units_in_grid', type=int, default=64)
    parser.add_argument('--save_individual_units', action='store_true',
                        help='Save one PNG per top-selective unit')
    parser.add_argument('--top_k_individual', type=int, default=20,
                        help='How many top-selective units to save individually')
    parser.add_argument('--top_k',        type=int, default=8,
                        help='Top-k samples to track per unit for mosaics')
    parser.add_argument('--save_topk_mosaics', action='store_true',
                        help='Render cochleagram mosaics for top-k activating samples')
    parser.add_argument('--generate_report', action='store_true',
                        help='Write an HTML report after --tuning_output completes')

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

    graph = tf.compat.v1.get_default_graph()

    # ── Activation-map tensors (first sample only) ────────
    _ACT_OP_TYPES = {'Relu', 'MaxPool', 'Reshape', 'Softmax'}
    act_tensor_map = {
        op.name: op.outputs[0]
        for op in graph.get_operations()
        if op.type in _ACT_OP_TYPES and op.outputs
    }

    # ── Tuning-analysis setup ─────────────────────────────
    tuning_enabled = bool(args.tuning_output)
    tuning_layer_map = {}
    sum_acc = count_acc = topk_acc = {}
    sample_cache  = {}
    spatial_cache = {}
    tuning_fetch_names   = []
    tuning_fetch_tensors = []

    if tuning_enabled:
        os.makedirs(args.tuning_output, exist_ok=True)
        tuning_layer_map = build_layer_tensor_map(
            graph, layer_names_filter=args.layers)
        if not tuning_layer_map:
            print(f'WARNING: none of {args.layers} found in graph.')
            print('Available:', list_available_layers(graph))
            tuning_enabled = False
        else:
            print(f'Tuning layers ({len(tuning_layer_map)}): '
                  f'{list(tuning_layer_map)}')
            top_k = args.top_k if args.save_topk_mosaics else 0
            sum_acc, count_acc, topk_acc = make_accumulators(
                tuning_layer_map, n_classes=504, top_k=top_k)
            tuning_fetch_names   = list(tuning_layer_map.keys())
            tuning_fetch_tensors = list(tuning_layer_map.values())

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

    if args.activation_output:
        os.makedirs(args.activation_output, exist_ok=True)

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
    tuning_processed = 0

    for cochleagram, metadata, png_file in input_gen:
        processed_count += 1
        model_input = prepare_input_for_model(cochleagram)

        # ── Single forward pass ───────────────────────────
        # When tuning is enabled we extend the fetch list so that layer
        # activations are collected in the same forward pass as predictions.
        if tuning_enabled:
            all_tensors = ([predictions, probabilities]
                           + tuning_fetch_tensors)
            all_results = sess.run(all_tensors,
                                   feed_dict={input_placeholder: model_input})
            pred               = all_results[0]
            prob               = all_results[1:2]   # shape (1, 504) wrapped in list
            raw_layer_outputs  = all_results[2:]
        else:
            pred, prob = run_ckpt_inference(
                sess, input_placeholder, predictions, probabilities, model_input)
            raw_layer_outputs = []

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
            for i, (pred_class, confidence) in enumerate(
                    zip(top5_indices, top5_probs), 1):
                azim_deg, elev_deg = decode_class_index(pred_class)
                print(f'{i}. Class {pred_class}: '
                      f'Azimuth={azim_deg}°, Elevation={elev_deg}°  '
                      f'(confidence: {confidence:.4f})')
            print('=' * 50)

        # ── Activation maps (first sample only) ───────────
        if args.activation_output and processed_count == 1:
            all_tensors = [predictions, probabilities] + list(act_tensor_map.values())
            all_results = sess.run(all_tensors,
                                   feed_dict={input_placeholder: model_input})
            layer_outputs = dict(zip(act_tensor_map.keys(), all_results[2:]))
            base_path = os.path.join(args.activation_output, 'activation.png')
            print(f'\nSaving activation maps for {len(layer_outputs)} layers '
                  f'→ {args.activation_output}/')
            plot_activation_maps(layer_outputs, list(act_tensor_map.keys()), base_path)

        # ── Per-sample prediction plot ─────────────────────
        if args.plot_output:
            plot_output_path = (
                os.path.join(args.plot_output, png_file)
                if png_file else
                os.path.join(args.plot_output, f'sample_{processed_count}.png'))
            plot_prediction_distribution(
                prob[0], ground_truth=metadata, output_path=plot_output_path)

        # ── Aggregate error accumulation ───────────────────
        if args.aggregate:
            condition = metadata.get(
                'condition', 'tfrecord' if args.tfrecord else 'unknown')
            if condition not in agg_data:
                agg_data[condition] = {
                    'top1_az_errs': [], 'top1_el_errs': [],
                    'top5_az_errs': [], 'top5_el_errs': []}

            gt_azim = metadata['azimuth']
            gt_elev = metadata['elevation']

            top1_az, top1_el = decode_class_index(top5_indices[0])
            agg_data[condition]['top1_az_errs'].append(
                ((top1_az - gt_azim + 180) % 360) - 180)
            agg_data[condition]['top1_el_errs'].append(top1_el - gt_elev)

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

        # ── Tuning accumulation ────────────────────────────
        if tuning_enabled and raw_layer_outputs:
            if 'azimuth' in metadata and 'elevation' in metadata:
                az_bin    = metadata['azimuth']  // 5
                el_bin    = metadata['elevation'] // 10
                class_idx = el_bin * 72 + az_bin

                layer_activations = {}
                for name, raw in zip(tuning_fetch_names, raw_layer_outputs):
                    is_conv = _is_conv(tuning_layer_map[name])
                    layer_activations[name] = reduce_activation(
                        raw, is_conv, args.spatial_reduction)

                accumulate(sum_acc, count_acc, topk_acc,
                           layer_activations, class_idx, tuning_processed)

                if args.save_topk_mosaics:
                    sample_cache[tuning_processed] = cochleagram
                    for name, raw in zip(tuning_fetch_names, raw_layer_outputs):
                        if _is_conv(tuning_layer_map[name]):
                            spatial_cache.setdefault(name, {})[tuning_processed] = raw[0]

                tuning_processed += 1

    sess.close()

    # ── Aggregate plot ─────────────────────────────────────
    if args.aggregate:
        plot_aggregate_errors(agg_data, args.aggregate)

    # ── Tuning finalisation ────────────────────────────────
    if tuning_enabled and tuning_processed > 0:
        print(f'\nTuning: {tuning_processed} labelled samples accumulated.')
        print('Computing tuning maps...')
        tuning_maps = compute_tuning_maps(sum_acc, count_acc)
        selectivity = compute_selectivity(tuning_maps)
        print_coverage(tuning_maps, count_acc)

        npz_path = save_npz(tuning_maps, selectivity, args.tuning_output)
        print(f'Saved: {npz_path}')

        print('Rendering tuning plots...')
        save_all_plots(tuning_maps, selectivity, args.tuning_output,
                       max_units_in_grid=args.max_units_in_grid,
                       top_k_individual=args.top_k_individual,
                       save_individual_units=args.save_individual_units,
                       total_samples=tuning_processed)

        if args.save_topk_mosaics:
            print('Rendering top-k mosaics...')
            prepare_spatial_cache_for_render(
                topk_acc, selectivity, spatial_cache, args.top_k_individual)
            save_topk_mosaics(topk_acc, sample_cache, spatial_cache,
                              args.tuning_output, args.top_k_individual,
                              selectivity, tuning_layer_map)

        if args.generate_report:
            from src.visualization import generate_tuning_report
            report_path = generate_tuning_report(args.tuning_output)
            print(f'Report: {report_path}')
        else:
            print(f'All tuning outputs written to: {args.tuning_output}')


if __name__ == '__main__':
    main()
