"""
src/tuning.py
─────────────
Class-conditional activation aggregation for BinauralLocalizationCNN.

Provides building blocks for computing per-unit 7×72 tuning maps and
tracking the top-k highest-activating samples per unit.

Importable by both analyze_unit_tuning.py (standalone CLI) and
test_inference_minimal.py (integrated --tuning_output flag).
"""
import os
import numpy as np
from collections import OrderedDict


# Default layers to analyse (friendly names produced by build_layer_tensor_map)
DEFAULT_LAYERS = ['pool_0', 'pool_1', 'pool_2', 'pool_3', 'relu_8']


# ─────────────────────────────────────────────────────────
#  RunningTopK — streaming top-k tracker
# ─────────────────────────────────────────────────────────

class RunningTopK:
    """Track the k highest-valued samples seen so far for a single unit.

    Memory: O(k) regardless of how many samples are processed.
    """

    def __init__(self, k=8):
        self.k       = k
        self.values  = np.full(k, -np.inf, dtype=np.float32)
        self.indices = np.full(k, -1,      dtype=np.int64)

    def update(self, value: float, sample_idx: int):
        min_pos = int(np.argmin(self.values))
        if value > self.values[min_pos]:
            self.values[min_pos]  = value
            self.indices[min_pos] = sample_idx

    def top_indices(self):
        """Return (sample_indices, values) sorted by activation descending."""
        order = np.argsort(self.values)[::-1]
        valid = self.indices[order] >= 0
        return self.indices[order][valid], self.values[order][valid]


# ─────────────────────────────────────────────────────────
#  Graph utilities
# ─────────────────────────────────────────────────────────

def build_layer_tensor_map(graph, layer_names_filter=None):
    """Scan a TF1 graph for Relu and MaxPool ops; assign friendly names.

    Naming convention:  relu_0, relu_1, …   pool_0, pool_1, …
    in order of appearance during graph traversal.

    Args:
        graph:              tf.compat.v1.Graph
        layer_names_filter: list of friendly names to keep, or None for all

    Returns:
        OrderedDict {friendly_name: tf.Tensor}
    """
    relu_count = pool_count = 0
    result = OrderedDict()
    for op in graph.get_operations():
        if op.type == 'Relu' and op.outputs:
            name = f'relu_{relu_count}'
            relu_count += 1
        elif op.type == 'MaxPool' and op.outputs:
            name = f'pool_{pool_count}'
            pool_count += 1
        else:
            continue
        if layer_names_filter is None or name in layer_names_filter:
            result[name] = op.outputs[0]
    return result


def list_available_layers(graph):
    """Return all friendly layer names in graph order (for --layers help)."""
    return list(build_layer_tensor_map(graph).keys())


def _n_units(tensor):
    """Number of units (channels) for a layer tensor.

    Works for both conv (batch, H, W, C) and FC (batch, N).
    """
    return tensor.get_shape().as_list()[-1]


def _is_conv(tensor):
    return len(tensor.get_shape().as_list()) == 4


# ─────────────────────────────────────────────────────────
#  Accumulators
# ─────────────────────────────────────────────────────────

def make_accumulators(layer_tensor_map, n_classes=504, top_k=0):
    """Allocate per-layer online accumulation arrays.

    Returns:
        sum_acc:   {name: np.zeros((n_classes, n_units), float64)}
        count_acc: {name: np.zeros(n_classes, int64)}
        topk_acc:  {name: [RunningTopK, …]}  — empty dict when top_k == 0
    """
    sum_acc = count_acc = {}
    sum_acc   = {}
    count_acc = {}
    topk_acc  = {}
    for name, tensor in layer_tensor_map.items():
        n = _n_units(tensor)
        sum_acc[name]   = np.zeros((n_classes, n), dtype=np.float64)
        count_acc[name] = np.zeros(n_classes,       dtype=np.int64)
        if top_k > 0:
            topk_acc[name] = [RunningTopK(top_k) for _ in range(n)]
    return sum_acc, count_acc, topk_acc


def reduce_activation(raw, is_conv_layer, spatial_reduction='max'):
    """Reduce a raw activation tensor to a 1-D unit vector.

    Conv  (1, H, W, C) → max/mean over spatial dims → (C,)
    FC    (1, N)        → strip batch dim            → (N,)
    """
    if is_conv_layer:
        if spatial_reduction == 'mean':
            return raw[0].mean(axis=(0, 1))
        else:
            return raw[0].max(axis=(0, 1))
    else:
        return raw[0]


def accumulate(sum_acc, count_acc, topk_acc, layer_activations,
               class_idx, sample_idx):
    """Update all accumulators for one sample.

    Args:
        layer_activations: {name: 1-D np.ndarray (n_units,)}
        class_idx:         flat 0-503 class index for this sample
        sample_idx:        sequential sample counter (used by RunningTopK)
    """
    for name, vec in layer_activations.items():
        sum_acc[name][class_idx]   += vec.astype(np.float64)
        count_acc[name][class_idx] += 1
        if name in topk_acc:
            for unit_idx, tracker in enumerate(topk_acc[name]):
                tracker.update(float(vec[unit_idx]), sample_idx)


# ─────────────────────────────────────────────────────────
#  Tuning map computation
# ─────────────────────────────────────────────────────────

def compute_tuning_maps(sum_acc, count_acc):
    """Divide accumulated sums by counts; reshape to (N_units, 7, 72).

    Classes with zero samples → NaN.

    Returns:
        {layer_name: np.ndarray (N_units, 7, 72, float32)}
    """
    result = {}
    for name, s in sum_acc.items():
        c = count_acc[name][:, np.newaxis].astype(np.float64)
        with np.errstate(invalid='ignore'):
            mean = np.where(c > 0, s / c, np.nan).astype(np.float32)
        # mean: (504, N_units) → (N_units, 7, 72)
        result[name] = mean.T.reshape(-1, 7, 72)
    return result


def compute_selectivity(tuning_maps_dict):
    """Per-unit variance across 504 class means (NaN-safe).

    Returns:
        {layer_name: np.ndarray (N_units,) float32}
    """
    result = {}
    for name, tmap in tuning_maps_dict.items():
        flat = tmap.reshape(tmap.shape[0], -1).astype(np.float64)
        result[name] = np.nanvar(flat, axis=1).astype(np.float32)
    return result


def save_npz(tuning_maps, selectivity, output_dir):
    """Save tuning_maps.npz into output_dir. Returns the path."""
    npz_path = os.path.join(output_dir, 'tuning_maps.npz')
    save_dict = {}
    for name, tmap in tuning_maps.items():
        save_dict[name]                  = tmap
        save_dict[f'{name}_selectivity'] = selectivity[name]
    np.savez_compressed(npz_path, **save_dict)
    return npz_path


# ─────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────

def _safe_layer_dir(output_dir, sub, layer_name):
    d = os.path.join(output_dir, sub, layer_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_all_plots(tuning_maps, selectivity, output_dir,
                   max_units_in_grid=64, top_k_individual=20,
                   save_individual_units=False, total_samples=0):
    """Render attribution summary, tuning grids, selectivity rankings, and optional per-unit PNGs."""
    from src.visualization import (plot_layer_attribution_map,
                                   plot_unit_tuning_grid, plot_selectivity_ranking)

    # ── Primary cross-layer summary (the main overview figure) ──
    plot_layer_attribution_map(
        tuning_maps,
        output_path=os.path.join(output_dir, 'layer_attribution_map.png'),
        total_samples=total_samples)

    for layer_name, tmap in tuning_maps.items():
        sel = selectivity[layer_name]

        plot_unit_tuning_grid(
            tmap, layer_name,
            output_path=os.path.join(output_dir, f'tuning_grid_{layer_name}.png'),
            selectivity_scores=sel,
            max_units=max_units_in_grid,
            total_samples=total_samples)

        plot_selectivity_ranking(
            sel, layer_name,
            output_path=os.path.join(output_dir, f'selectivity_{layer_name}.png'))

        if save_individual_units:
            top_unit_indices = np.argsort(sel)[::-1][:top_k_individual]
            unit_dir = _safe_layer_dir(output_dir, 'units', layer_name)
            for unit_idx in top_unit_indices:
                _plot_single_unit(
                    tmap[unit_idx], sel[unit_idx], unit_idx, layer_name,
                    os.path.join(unit_dir, f'unit_{unit_idx:04d}.png'))


def _plot_single_unit(tuning_map, sel_score, unit_idx, layer_name, output_path):
    """3-panel: full tuning map + azimuth marginal + elevation marginal."""
    import matplotlib.pyplot as plt
    from src.visualization import _apply_spatial_axes

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.suptitle(f'Unit {unit_idx} — {layer_name} — selectivity {sel_score:.4f}',
                 fontsize=11, fontweight='bold')

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'plasma_nan', plt.cm.plasma(np.linspace(0, 1, 256)))
    cmap.set_bad('#c8c8c8')
    ax = axes[0]
    im = ax.imshow(tuning_map, aspect='auto', cmap=cmap,
                   origin='lower', interpolation='nearest')
    _apply_spatial_axes(ax, fontsize=8)
    ax.set_title('Tuning map (7 × 72)', fontsize=9)
    plt.colorbar(im, ax=ax, label='Mean activation')

    ax = axes[1]
    az_marginal = np.nanmean(tuning_map, axis=0)
    ax.bar(np.arange(72) * 5, az_marginal, width=4, color='steelblue', alpha=0.8)
    ax.set_xlabel('Azimuth (°)', fontsize=9)
    ax.set_ylabel('Mean activation', fontsize=9)
    ax.set_title('Azimuth marginal', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    el_marginal = np.nanmean(tuning_map, axis=1)
    ax.barh(np.arange(7) * 10, el_marginal, height=8, color='coral', alpha=0.8)
    ax.set_ylabel('Elevation (°)', fontsize=9)
    ax.set_xlabel('Mean activation', fontsize=9)
    ax.set_title('Elevation marginal', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def save_topk_mosaics(topk_acc, sample_cache, spatial_cache,
                      output_dir, top_k_individual, selectivity,
                      layer_tensor_map):
    """Render top-k cochleagram mosaics for the most selective units."""
    from src.visualization import plot_topk_mosaic

    for layer_name, trackers in topk_acc.items():
        sel = selectivity[layer_name]
        top_unit_indices = np.argsort(sel)[::-1][:top_k_individual]
        mosaic_dir = _safe_layer_dir(output_dir, 'mosaics', layer_name)
        is_conv    = _is_conv(layer_tensor_map[layer_name])

        for unit_idx in top_unit_indices:
            tracker = trackers[unit_idx]
            indices, values = tracker.top_indices()
            if len(indices) == 0:
                continue

            cochleagrams = []
            act_vals     = []
            spatial_maps = [] if is_conv else None

            for sidx, val in zip(indices, values):
                sidx = int(sidx)
                if sidx not in sample_cache:
                    continue
                cochleagrams.append(sample_cache[sidx])
                act_vals.append(float(val))
                if is_conv:
                    sm = spatial_cache.get(layer_name, {}).get(sidx)
                    spatial_maps.append(sm)

            if not cochleagrams:
                continue

            plot_topk_mosaic(
                cochleagrams, act_vals, unit_idx, layer_name,
                output_path=os.path.join(
                    mosaic_dir, f'unit_{unit_idx:04d}_topk.png'),
                activation_maps=spatial_maps if is_conv else None)


def prepare_spatial_cache_for_render(topk_acc, selectivity, spatial_cache,
                                     top_k_individual):
    """Slice (H, W, C) maps down to per-unit (H, W) to save memory.

    Mutates spatial_cache in-place and returns it.
    """
    for layer_name, trackers in topk_acc.items():
        if layer_name not in spatial_cache:
            continue
        sel = selectivity[layer_name]
        top_units = set(np.argsort(sel)[::-1][:top_k_individual].tolist())
        unit_spatial = {}
        for unit_idx in top_units:
            tracker = trackers[unit_idx]
            indices, _ = tracker.top_indices()
            for sidx in indices:
                sidx = int(sidx)
                if sidx in spatial_cache.get(layer_name, {}):
                    full_map = spatial_cache[layer_name][sidx]
                    if unit_idx not in unit_spatial:
                        unit_spatial[unit_idx] = {}
                    unit_spatial[unit_idx][sidx] = full_map[:, :, unit_idx]
        spatial_cache[layer_name] = unit_spatial
    return spatial_cache


def print_coverage(tuning_maps, count_acc):
    """Print per-layer class coverage statistics."""
    for name in tuning_maps:
        counts  = count_acc[name]
        covered = int((counts > 0).sum())
        min_c   = int(counts[counts > 0].min()) if covered else 0
        max_c   = int(counts.max())
        print(f'  {name}: {covered}/504 classes covered, '
              f'min_count={min_c}, max_count={max_c}')
