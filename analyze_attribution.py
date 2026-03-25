#!/usr/bin/env python
"""
analyze_attribution.py — sample-level interpretability analysis
───────────────────────────────────────────────────────────────
Connects model predictions back to input cochleagram regions using
gradient-based attribution techniques from the DNN interpretability
literature (Simonyan et al. [57], Zeiler & Fergus [55], Selvaraju Grad-CAM).

For each input sample, produces:
  1. Saliency maps — which freq bins and time steps drove the prediction
  2. Grad-CAM ladder — per-layer attention overlaid on the cochleagram
  3. FC→output attribution — which FC units drove the predicted class

Optionally aggregates across samples to reveal stable per-class patterns.

Usage:
    # Quick smoke test — 3 samples, vanilla saliency only
    python analyze_attribution.py \\
        --model_dir models/net1 \\
        --tfrecords_dir data/ \\
        --output_dir attribution_analysis/ \\
        --max_samples 3 \\
        --methods vanilla

    # Full run with all methods
    python analyze_attribution.py \\
        --model_dir models/net1 \\
        --tfrecords_dir data/ \\
        --output_dir attribution_analysis/ \\
        --max_samples 20 \\
        --methods vanilla grad_x_input smoothgrad \\
        --smoothgrad_n 30
"""
import os
import sys
import glob
import argparse

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf

from src.graph_builder import load_config, build_inference_graph
from src.data_io import load_tfrecord_samples_generator, prepare_input_for_model
from src.inference import make_session_config, restore_checkpoint
from src.tuning import build_layer_tensor_map, _is_conv


# ─────────────────────────────────────────────────────────
#  Gradient attribution helpers
# ─────────────────────────────────────────────────────────

def build_gradient_ops(graph, input_ph, logits, conv_tensors):
    """Build TF ops for saliency maps and Grad-CAM.

    The input goes through tf.pow(x, 0.3) before the network.  The
    derivative 0.3 * x^(-0.7) produces inf/NaN at zero-valued bins,
    so we compute saliency w.r.t. the POST-nonlinearity tensor
    (after tf.pow).  This gives clean gradients that show which
    parts of the *transformed* input the network is sensitive to.

    Args:
        graph:        tf.Graph
        input_ph:     input placeholder (batch, 39, 8000, 2)
        logits:       output logits tensor (batch, 504)
        conv_tensors: OrderedDict {layer_name: tf.Tensor} for conv layers only

    Returns:
        class_ph:        placeholder for target class index
        saliency_op:     gradient of logit[class] w.r.t. post-nonlinearity
        saliency_target: the tensor saliency is computed w.r.t.
        gradcam_ops:     dict {layer_name: (activation, gradient)} tensors
    """
    # Placeholder for which class to compute gradients for
    class_ph = tf.compat.v1.placeholder(tf.int32, shape=(), name='target_class')

    # Extract the logit for the target class
    target_logit = logits[0, class_ph]

    # Find the post-nonlinearity tensor (output of tf.pow).
    # In graph_builder.py: nonlin = tf.pow(input_placeholder, 0.3)
    # This creates a 'Pow' op in the graph.
    pow_tensor = None
    for op in graph.get_operations():
        if op.type == 'Pow' and op.inputs[0] is input_ph:
            pow_tensor = op.outputs[0]
            break

    if pow_tensor is not None:
        print(f'  Found post-nonlinearity tensor: {pow_tensor.name}')
        saliency_target = pow_tensor
    else:
        # Fallback: use input_ph directly (may produce NaN)
        print('  [WARN] Could not find Pow tensor, using raw input (may have NaN gradients)')
        saliency_target = input_ph

    # Saliency: ∂logit/∂post_nonlinearity
    saliency_op = tf.gradients(target_logit, saliency_target)[0]

    # Grad-CAM: ∂logit/∂conv_activation for each conv layer
    gradcam_ops = {}
    for lname, tensor in conv_tensors.items():
        grad = tf.gradients(target_logit, tensor)[0]
        if grad is not None:
            gradcam_ops[lname] = (tensor, grad)
        else:
            print(f'  [WARN] No gradient path for {lname}, skipping Grad-CAM')

    return class_ph, saliency_op, saliency_target, gradcam_ops


def compute_gradcam(activation, gradient, target_h=39, target_w=8000):
    """Compute Grad-CAM heatmap from activation and gradient arrays.

    Args:
        activation: np.ndarray (1, H, W, C)
        gradient:   np.ndarray (1, H, W, C)
        target_h:   target height for upsampling
        target_w:   target width for upsampling

    Returns:
        np.ndarray (target_h, target_w) — class-discriminative heatmap
    """
    # Channel importance weights: global average pool of gradients
    weights = gradient[0].mean(axis=(0, 1))  # (C,)

    # Weighted combination of activation channels
    cam = np.sum(weights[np.newaxis, np.newaxis, :] * activation[0], axis=2)  # (H, W)

    # ReLU — only keep positive influence
    cam = np.maximum(cam, 0)

    # Upsample to input resolution
    if cam.shape[0] != target_h or cam.shape[1] != target_w:
        from scipy.ndimage import zoom as nd_zoom
        scale_h = target_h / cam.shape[0]
        scale_w = target_w / cam.shape[1]
        cam = nd_zoom(cam, (scale_h, scale_w), order=1)

    return cam


def compute_smoothgrad(sess, input_ph, saliency_op, class_ph,
                       model_input, target_class, n=30, sigma=0.15):
    """SmoothGrad: average saliency over noisy copies of the input.

    Args:
        sess:          TF session
        input_ph:      input placeholder
        saliency_op:   gradient op
        class_ph:      target class placeholder
        model_input:   np.ndarray (1, 39, 8000, 2)
        target_class:  int
        n:             number of noise samples
        sigma:         noise standard deviation (fraction of input range)

    Returns:
        np.ndarray (39, 8000, 2) — smoothed saliency map
    """
    stdev = sigma * (model_input.max() - model_input.min())
    total_grad = np.zeros_like(model_input[0])

    for i in range(n):
        noise = np.random.normal(0, stdev, model_input.shape).astype(np.float32)
        noisy_input = model_input + noise
        grad = sess.run(saliency_op, feed_dict={
            input_ph: noisy_input, class_ph: target_class})
        total_grad += grad[0]

    return total_grad / n


def azimuth_to_sector(az_deg):
    """Map azimuth degrees to sector name."""
    az = az_deg % 360
    if az > 315 or az <= 45:
        return 'Front (±45°)'
    if az <= 135:
        return 'Right (45-135°)'
    if az <= 225:
        return 'Back (135-225°)'
    return 'Left (225-315°)'


SECTOR_NAMES = ['Front (±45°)', 'Right (45-135°)',
                'Back (135-225°)', 'Left (225-315°)']


def compute_attention_profile(gradcam_maps_by_layer, layer_names):
    """Compute per-layer freq/time attention fractions from Grad-CAM maps.

    Args:
        gradcam_maps_by_layer: dict {layer_name: list of np.ndarray (39, 8000)}
        layer_names:           list of str

    Returns:
        dict with 'freq_bands' and 'time_regions', each np.ndarray (n_layers, 3)
    """
    n_layers = len(layer_names)
    freq_bands = np.zeros((n_layers, 3))
    time_regions = np.zeros((n_layers, 3))

    freq_edges = [0, 13, 26, 39]
    time_thirds = 3

    for li, lname in enumerate(layer_names):
        maps = gradcam_maps_by_layer.get(lname, [])
        if not maps:
            continue

        # Average Grad-CAM map across samples
        avg_map = np.mean(np.stack(maps), axis=0)  # (39, 8000)
        total = avg_map.sum() + 1e-12

        # Frequency bands
        for bi in range(3):
            freq_bands[li, bi] = avg_map[freq_edges[bi]:freq_edges[bi+1], :].sum() / total

        # Time regions (thirds)
        w = avg_map.shape[1]
        for ti in range(3):
            t0 = ti * w // time_thirds
            t1 = (ti + 1) * w // time_thirds
            time_regions[li, ti] = avg_map[:, t0:t1].sum() / total

    return {'freq_bands': freq_bands, 'time_regions': time_regions}


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sample-level input attribution analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_dir', required=True,
                        help='Checkpoint directory (contains config_array.npy)')
    parser.add_argument('--tfrecord', default='',
                        help='Single .tfrecords file')
    parser.add_argument('--tfrecords_dir', default='',
                        help='Directory of .tfrecords shards')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Number of samples to analyze')
    parser.add_argument('--checkpoint', default='',
                        help='Explicit checkpoint path override')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--methods', nargs='+',
                        default=['vanilla', 'grad_x_input'],
                        choices=['vanilla', 'grad_x_input', 'smoothgrad'],
                        help='Saliency methods to compute')
    parser.add_argument('--smoothgrad_n', type=int, default=30,
                        help='Number of noise samples for SmoothGrad')
    parser.add_argument('--smoothgrad_sigma', type=float, default=0.15,
                        help='Noise standard deviation for SmoothGrad')
    parser.add_argument('--gradcam_layers', nargs='+',
                        default=['pool_0', 'pool_1', 'pool_2', 'pool_3'],
                        help='Conv layers for Grad-CAM')
    parser.add_argument('--target_class', default='predicted',
                        choices=['predicted', 'ground_truth', 'both'],
                        help='Which class to compute gradients for')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.abspath(args.model_dir)

    if not args.tfrecord and not args.tfrecords_dir:
        parser.error('Provide --tfrecord or --tfrecords_dir')

    # ── Build graph ───────────────────────────────────────
    config_array = load_config(
        model_dir, target_device='/gpu:0' if args.use_gpu else '/cpu:0')
    print('Building graph...')
    input_ph, logits, probs, preds = build_inference_graph(config_array, batch_size=1)
    print('Graph built.')

    graph = tf.compat.v1.get_default_graph()

    # Find conv layers for Grad-CAM
    all_layers = build_layer_tensor_map(graph)
    conv_layers = {name: tensor for name, tensor in all_layers.items()
                   if _is_conv(tensor) and name in args.gradcam_layers}

    # Find FC ReLU layer for FC attribution
    fc_layers = {name: tensor for name, tensor in all_layers.items()
                 if not _is_conv(tensor)}
    fc_layer_name = list(fc_layers.keys())[-1] if fc_layers else None
    fc_tensor = fc_layers[fc_layer_name] if fc_layer_name else None

    print(f'Grad-CAM layers: {list(conv_layers.keys())}')
    print(f'FC layer for attribution: {fc_layer_name}')

    # Report spatial dims of conv layers
    for lname, tensor in conv_layers.items():
        shape = tensor.get_shape().as_list()
        print(f'  {lname}: shape={shape}')
        if shape[1] == 1 and shape[2] == 1:
            print(f'    [WARN] 1×1 spatial dims — Grad-CAM will be uninformative')

    # Build gradient ops
    class_ph, saliency_op, saliency_target, gradcam_ops = build_gradient_ops(
        graph, input_ph, logits, conv_layers)

    # Find output weight variable for FC attribution
    output_weight_tensor = None
    for var in tf.compat.v1.global_variables():
        if 'wc_out_0' in var.name:
            output_weight_tensor = var
            break

    # ── Session + restore ─────────────────────────────────
    sess = tf.compat.v1.Session(config=make_session_config(args.use_gpu))
    sess.run(tf.compat.v1.global_variables_initializer())
    restore_checkpoint(sess, model_dir,
                       checkpoint_path=args.checkpoint or None)

    # Get output weights for FC attribution
    output_weights = None
    if output_weight_tensor is not None:
        output_weights = sess.run(output_weight_tensor)
        print(f'Output weight matrix: {output_weights.shape}')

    # Fetch tensors for Grad-CAM activations + FC layer
    gradcam_layer_names = list(gradcam_ops.keys())
    fetch_gradcam = {}
    for lname in gradcam_layer_names:
        act_tensor, grad_tensor = gradcam_ops[lname]
        fetch_gradcam[lname] = (act_tensor, grad_tensor)

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

    # ── Aggregation accumulators ──────────────────────────
    sector_saliency_sums = {s: np.zeros((39, 8000, 2), dtype=np.float64)
                            for s in SECTOR_NAMES}
    sector_counts = {s: 0 for s in SECTOR_NAMES}
    gradcam_maps_by_layer = {lname: [] for lname in gradcam_layer_names}
    all_saliency_data = []

    # ── Main loop ─────────────────────────────────────────
    print(f'\nProcessing up to {args.max_samples} samples...')
    print(f'Methods: {args.methods}')
    processed = 0

    for cochleagram, metadata in sample_iter:
        if processed >= args.max_samples:
            break

        has_gt = 'azimuth' in metadata and 'elevation' in metadata
        ground_truth = None
        if has_gt:
            ground_truth = {
                'azimuth': metadata['azimuth'],
                'elevation': metadata['elevation']
            }

        model_input = prepare_input_for_model(cochleagram)

        # Forward pass — get prediction
        pred_class, prob_vals = sess.run(
            [preds, probs],
            feed_dict={input_ph: model_input})
        pred_class = int(pred_class[0])
        confidence = float(prob_vals[0, pred_class])

        # Determine target classes for gradient computation
        target_classes = []
        if args.target_class in ('predicted', 'both'):
            target_classes.append(('predicted', pred_class))
        if args.target_class in ('ground_truth', 'both') and has_gt:
            gt_class = (metadata['elevation'] // 10) * 72 + (metadata['azimuth'] // 5)
            target_classes.append(('ground_truth', gt_class))

        # Create sample output directory
        sample_dir = os.path.join(args.output_dir, f'sample_{processed:03d}')
        os.makedirs(sample_dir, exist_ok=True)

        # Save metadata
        meta_info = {
            'predicted_class': pred_class,
            'confidence': confidence,
        }
        if ground_truth:
            meta_info['gt_azimuth'] = ground_truth['azimuth']
            meta_info['gt_elevation'] = ground_truth['elevation']

        for target_label, target_class in target_classes:
            suffix = f'_{target_label}' if len(target_classes) > 1 else ''

            # ── Saliency maps ─────────────────────────────
            if 'vanilla' in args.methods:
                grad_val = sess.run(saliency_op, feed_dict={
                    input_ph: model_input, class_ph: target_class})
                vanilla_sal = grad_val[0]  # (39, 8000, 2)

                # Diagnostic: print gradient statistics
                if processed == 0 and target_label == target_classes[0][0]:
                    print(f'  [DIAG] Saliency gradient stats:')
                    print(f'    min={vanilla_sal.min():.6e}  max={vanilla_sal.max():.6e}')
                    print(f'    mean={vanilla_sal.mean():.6e}  std={vanilla_sal.std():.6e}')
                    print(f'    nonzero={np.count_nonzero(vanilla_sal)}/{vanilla_sal.size}')
                    print(f'    nan={np.isnan(vanilla_sal).sum()}  inf={np.isinf(vanilla_sal).sum()}')

                # Replace NaN/inf with zero
                vanilla_sal = np.nan_to_num(vanilla_sal, nan=0.0, posinf=0.0, neginf=0.0)

                from src.visualization import plot_saliency_overlay
                plot_saliency_overlay(
                    cochleagram, np.abs(vanilla_sal),
                    os.path.join(sample_dir, f'saliency_vanilla{suffix}.png'),
                    predicted_class=pred_class,
                    ground_truth=ground_truth,
                    confidence=confidence,
                    method_name='Vanilla Saliency')

                # Aggregate
                if has_gt:
                    sector = azimuth_to_sector(metadata['azimuth'])
                    sector_saliency_sums[sector] += np.abs(vanilla_sal).astype(np.float64)
                    sector_counts[sector] += 1

            if 'grad_x_input' in args.methods:
                grad_val = sess.run(saliency_op, feed_dict={
                    input_ph: model_input, class_ph: target_class})
                grad_clean = np.nan_to_num(grad_val[0], nan=0.0, posinf=0.0, neginf=0.0)
                # Use the post-nonlinearity values for grad×input
                post_nonlin = sess.run(saliency_target, feed_dict={input_ph: model_input})
                gxi_sal = post_nonlin[0] * grad_clean

                from src.visualization import plot_saliency_overlay
                plot_saliency_overlay(
                    cochleagram, gxi_sal,
                    os.path.join(sample_dir, f'saliency_grad_x_input{suffix}.png'),
                    predicted_class=pred_class,
                    ground_truth=ground_truth,
                    confidence=confidence,
                    method_name='Gradient × Input')

            if 'smoothgrad' in args.methods:
                sg_sal = compute_smoothgrad(
                    sess, input_ph, saliency_op, class_ph,
                    model_input, target_class,
                    n=args.smoothgrad_n, sigma=args.smoothgrad_sigma)

                from src.visualization import plot_saliency_overlay
                plot_saliency_overlay(
                    cochleagram, np.abs(sg_sal),
                    os.path.join(sample_dir, f'saliency_smoothgrad{suffix}.png'),
                    predicted_class=pred_class,
                    ground_truth=ground_truth,
                    confidence=confidence,
                    method_name=f'SmoothGrad (n={args.smoothgrad_n})')

            # ── Grad-CAM ──────────────────────────────────
            if gradcam_ops:
                gcam_maps = []
                gcam_names = []

                for lname in gradcam_layer_names:
                    act_tensor, grad_tensor = gradcam_ops[lname]
                    act_val, grad_val = sess.run(
                        [act_tensor, grad_tensor],
                        feed_dict={input_ph: model_input, class_ph: target_class})

                    cam = compute_gradcam(act_val, grad_val)
                    gcam_maps.append(cam)
                    gcam_names.append(lname)

                    # Accumulate for attention profile
                    if target_label == 'predicted':
                        gradcam_maps_by_layer[lname].append(cam)

                from src.visualization import plot_gradcam_ladder
                plot_gradcam_ladder(
                    cochleagram, gcam_maps, gcam_names,
                    os.path.join(sample_dir, f'gradcam_ladder{suffix}.png'),
                    predicted_class=pred_class,
                    ground_truth=ground_truth)

            # ── FC attribution ────────────────────────────
            if fc_tensor is not None and output_weights is not None:
                fc_act = sess.run(fc_tensor, feed_dict={input_ph: model_input})
                fc_act = fc_act[0]  # (512,)

                from src.visualization import plot_fc_output_attribution
                plot_fc_output_attribution(
                    fc_act, output_weights, pred_class,
                    os.path.join(sample_dir, f'fc_attribution{suffix}.png'),
                    ground_truth=ground_truth)

        processed += 1
        from src.visualization import decode_class_index
        pred_az, pred_el = decode_class_index(pred_class)
        gt_str = ''
        if ground_truth:
            gt_str = f' | GT: az={ground_truth["azimuth"]}° el={ground_truth["elevation"]}°'
        print(f'  [{processed}/{args.max_samples}] '
              f'Pred: az={pred_az}° el={pred_el}° conf={confidence:.4f}{gt_str}')

    sess.close()
    print(f'\nDone. Processed {processed} samples.')

    # ── Aggregate visualizations ──────────────────────────
    agg_dir = os.path.join(args.output_dir, 'aggregate')
    os.makedirs(agg_dir, exist_ok=True)

    # Mean saliency by sector
    if 'vanilla' in args.methods:
        mean_saliency = {}
        active_sectors = []
        for sname in SECTOR_NAMES:
            if sector_counts[sname] > 0:
                mean_saliency[sname] = (
                    sector_saliency_sums[sname] / sector_counts[sname]
                ).astype(np.float32)
                active_sectors.append(sname)
                print(f'  Sector {sname}: {sector_counts[sname]} samples')

        if active_sectors:
            from src.visualization import plot_aggregate_saliency
            plot_aggregate_saliency(
                mean_saliency, active_sectors,
                os.path.join(agg_dir, 'mean_saliency_by_sector.png'))

    # Layer attention profile
    if gradcam_ops:
        has_maps = any(len(v) > 0 for v in gradcam_maps_by_layer.values())
        if has_maps:
            profile = compute_attention_profile(
                gradcam_maps_by_layer, gradcam_layer_names)

            from src.visualization import plot_layer_attention_profile
            plot_layer_attention_profile(
                profile, gradcam_layer_names,
                os.path.join(agg_dir, 'layer_attention_profile.png'))

            # Save as CSV too
            import csv
            csv_path = os.path.join(agg_dir, 'layer_attention_table.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['layer', 'low_freq', 'mid_freq', 'high_freq',
                                 'onset', 'middle', 'sustain'])
                for i, lname in enumerate(gradcam_layer_names):
                    row = [lname]
                    row += [f'{v:.4f}' for v in profile['freq_bands'][i]]
                    row += [f'{v:.4f}' for v in profile['time_regions'][i]]
                    writer.writerow(row)
            print(f'  Saved attention table: {csv_path}')

    # Save raw attribution data
    save_dict = {}
    for sname in SECTOR_NAMES:
        if sector_counts[sname] > 0:
            safe_name = sname.replace(' ', '_').replace('(', '').replace(')', '') \
                             .replace('°', 'deg').replace('±', 'pm')
            save_dict[f'mean_saliency_{safe_name}'] = mean_saliency.get(sname)
    if save_dict:
        npz_path = os.path.join(args.output_dir, 'attribution_data.npz')
        np.savez_compressed(npz_path, **save_dict)
        print(f'  Saved attribution data: {npz_path}')

    print(f'\nAll outputs in: {args.output_dir}')
    print(f'  Per-sample: {args.output_dir}/sample_XXX/')
    print(f'  Aggregate:  {args.output_dir}/aggregate/')


if __name__ == '__main__':
    main()
