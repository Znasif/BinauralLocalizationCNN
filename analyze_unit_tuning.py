#!/usr/bin/env python
"""
analyze_unit_tuning.py  —  standalone CLI
─────────────────────────────────────────
Class-conditional activation aggregation for BinauralLocalizationCNN.

For each unit in selected layers, computes the mean activation per
azimuth/elevation class across many tfrecord samples, producing a 7×72
"tuning map" per unit. Optionally tracks the top-k activating samples
per unit and renders cochleagram mosaics.

Usage:
    # Quick pilot — verify shapes with 50 samples
    python analyze_unit_tuning.py \\
        --model_dir models/net1 \\
        --tfrecord data/train0.tfrecords \\
        --output_dir tuning_analysis/ \\
        --max_samples 50

    # Full run with top-k mosaics and per-unit PNGs
    python analyze_unit_tuning.py \\
        --model_dir models/net1 \\
        --tfrecords_dir data/ \\
        --output_dir tuning_analysis/ \\
        --save_individual_units \\
        --save_topk_mosaics \\
        --top_k 8 \\
        --use_gpu
"""
import os
import sys
import glob
import argparse

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

from src.graph_builder import load_config, build_inference_graph
from src.data_io       import load_tfrecord_samples_generator, prepare_input_for_model
from src.inference     import make_session_config, restore_checkpoint
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
        description='Class-conditional unit tuning analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input
    parser.add_argument('--model_dir',    required=True,
                        help='Checkpoint directory (contains config_array.npy)')
    parser.add_argument('--tfrecord',     default='',
                        help='Single .tfrecords file')
    parser.add_argument('--tfrecords_dir', default='',
                        help='Directory of .tfrecords shards')
    parser.add_argument('--max_samples',  type=int, default=0,
                        help='Stop after this many samples (0 = all)')
    # Model
    parser.add_argument('--checkpoint',   default='',
                        help='Explicit checkpoint path override')
    parser.add_argument('--use_gpu',      action='store_true', default=False)
    # Layer selection
    parser.add_argument('--layers',       nargs='+', default=DEFAULT_LAYERS,
                        help='Friendly layer names (relu_N, pool_N)')
    # Spatial reduction
    parser.add_argument('--spatial_reduction', default='max',
                        choices=['max', 'mean'],
                        help='How to reduce H×W for conv layers')
    # Output
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--max_units_in_grid', type=int, default=64)
    parser.add_argument('--save_individual_units', action='store_true',
                        help='Save one PNG per unit for top-k selective units')
    parser.add_argument('--top_k_individual', type=int, default=20,
                        help='How many top-selective units to save individually')
    # Top-k samples
    parser.add_argument('--top_k',        type=int, default=8,
                        help='Samples to track per unit for mosaics')
    parser.add_argument('--save_topk_mosaics', action='store_true',
                        help='Render cochleagram mosaics for top-k samples')
    # HTML report
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate index.html report after analysis')
    # Misc
    parser.add_argument('--quiet',        action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.abspath(args.model_dir)

    if not args.tfrecord and not args.tfrecords_dir:
        parser.error('Provide --tfrecord or --tfrecords_dir')

    # ── Build graph ───────────────────────────────────────
    config_array = load_config(
        model_dir, target_device='/gpu:0' if args.use_gpu else '/cpu:0')
    print('Building graph...')
    input_ph, _, _, _ = build_inference_graph(config_array, batch_size=1)
    print('Graph built.')

    graph            = tf.compat.v1.get_default_graph()
    layer_tensor_map = build_layer_tensor_map(graph, layer_names_filter=args.layers)

    if not layer_tensor_map:
        print(f'ERROR: none of the requested layers {args.layers} found in graph.')
        print('Available layers:', list_available_layers(graph))
        return

    print(f'Target layers ({len(layer_tensor_map)}): {list(layer_tensor_map)}')

    # ── Session + restore ─────────────────────────────────
    sess = tf.compat.v1.Session(config=make_session_config(args.use_gpu))
    sess.run(tf.compat.v1.global_variables_initializer())
    restore_checkpoint(sess, model_dir,
                       checkpoint_path=args.checkpoint or None)

    # ── Accumulators ──────────────────────────────────────
    top_k = args.top_k if args.save_topk_mosaics else 0
    sum_acc, count_acc, topk_acc = make_accumulators(
        layer_tensor_map, n_classes=504, top_k=top_k)

    sample_cache  = {}
    spatial_cache = {}

    fetch_names   = list(layer_tensor_map.keys())
    fetch_tensors = list(layer_tensor_map.values())

    # ── Data source ───────────────────────────────────────
    if args.tfrecords_dir:
        shards = sorted(glob.glob(os.path.join(args.tfrecords_dir, '*.tfrecords')))
        if not shards:
            print(f'ERROR: no .tfrecords found in {args.tfrecords_dir}')
            return
        def _iter():
            for shard in shards:
                yield from load_tfrecord_samples_generator(shard)
        sample_iter = _iter()
    else:
        sample_iter = load_tfrecord_samples_generator(args.tfrecord)

    # ── Main accumulation loop ────────────────────────────
    print('Accumulating activations...')
    processed = skipped = 0

    for cochleagram, metadata in sample_iter:
        if args.max_samples and processed >= args.max_samples:
            break

        if 'azimuth' not in metadata or 'elevation' not in metadata:
            skipped += 1
            continue

        az_bin    = metadata['azimuth']  // 5
        el_bin    = metadata['elevation'] // 10
        class_idx = el_bin * 72 + az_bin

        model_input = prepare_input_for_model(cochleagram)
        raw_outputs = sess.run(fetch_tensors, feed_dict={input_ph: model_input})

        layer_activations = {}
        for name, raw in zip(fetch_names, raw_outputs):
            is_conv = _is_conv(layer_tensor_map[name])
            layer_activations[name] = reduce_activation(
                raw, is_conv, args.spatial_reduction)

        accumulate(sum_acc, count_acc, topk_acc,
                   layer_activations, class_idx, processed)

        if args.save_topk_mosaics:
            sample_cache[processed] = cochleagram
            for name, raw in zip(fetch_names, raw_outputs):
                if _is_conv(layer_tensor_map[name]):
                    spatial_cache.setdefault(name, {})[processed] = raw[0]

        processed += 1
        if not args.quiet:
            limit_str = f'/{args.max_samples}' if args.max_samples else ''
            print(f'\r  {processed}{limit_str} samples', end='', flush=True)
        else:
            print(f'PROGRESS: {processed}', flush=True)

    sess.close()
    print(f'\nDone. Processed: {processed}  Skipped (no label): {skipped}'  if not args.quiet else '')

    # ── Compute + save ─────────────────────────────────────
    print('Computing tuning maps...')
    tuning_maps = compute_tuning_maps(sum_acc, count_acc)
    selectivity = compute_selectivity(tuning_maps)
    print_coverage(tuning_maps, count_acc)

    npz_path = save_npz(tuning_maps, selectivity, args.output_dir)
    print(f'Saved: {npz_path}')

    print('Rendering plots...')
    save_all_plots(tuning_maps, selectivity, args.output_dir,
                   max_units_in_grid=args.max_units_in_grid,
                   top_k_individual=args.top_k_individual,
                   save_individual_units=args.save_individual_units,
                   total_samples=processed)

    if args.save_topk_mosaics:
        print('Rendering top-k mosaics...')
        prepare_spatial_cache_for_render(
            topk_acc, selectivity, spatial_cache, args.top_k_individual)
        save_topk_mosaics(topk_acc, sample_cache, spatial_cache,
                          args.output_dir, args.top_k_individual,
                          selectivity, layer_tensor_map)

    if args.generate_report:
        from src.visualization import generate_tuning_report
        report_path = generate_tuning_report(args.output_dir)
        print(f'Report: {report_path}')
    else:
        print(f'\nAll outputs written to: {args.output_dir}')
        print(f'  python generate_tuning_report.py --output_dir {args.output_dir}')


if __name__ == '__main__':
    main()
