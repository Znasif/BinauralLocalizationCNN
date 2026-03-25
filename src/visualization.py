import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec
import matplotlib.patches
import seaborn as sns


# ─────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────

def decode_class_index(class_idx):
    """Convert a flat 504-class index to (azimuth_deg, elevation_deg).

    Layout: 7 elevations × 72 azimuths
      azimuth   = (class_idx % 72) * 5   degrees  (0–355, 5-deg bins)
      elevation = (class_idx // 72) * 10 degrees  (0–60,  10-deg bins)
    """
    azim_deg = (class_idx % 72) * 5
    elev_deg = (class_idx // 72) * 10
    return int(azim_deg), int(elev_deg)


# ─────────────────────────────────────────────────────────
#  Per-sample prediction plot
# ─────────────────────────────────────────────────────────

def plot_prediction_distribution(probabilities, ground_truth=None,
                                 output_path='prediction_plot.png'):
    """2×2 subplot: heatmap, azimuth bar, elevation bar, top-10 table.

    Args:
        probabilities: array of shape (504,)
        ground_truth:  optional dict with 'azimuth' and 'elevation' keys (degrees)
        output_path:   file path to save the PNG
    """
    print(probabilities.shape)
    prob_grid = probabilities.reshape(7, 72)

    fig = plt.figure(figsize=(16, 10))

    # 1. Heatmap
    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(prob_grid, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax1.set_ylabel('Elevation (degrees)', fontsize=12)
    ax1.set_title('Probability Distribution (Heatmap)', fontsize=14, fontweight='bold')
    azim_ticks = np.arange(0, 72, 6)
    elev_ticks = np.arange(0, 7, 2)
    ax1.set_xticks(azim_ticks)
    ax1.set_xticklabels([f'{i*5}°' for i in azim_ticks])
    ax1.set_yticks(elev_ticks)
    ax1.set_yticklabels([f'{i*10}°' for i in elev_ticks])
    if ground_truth and 'azimuth' in ground_truth:
        gt_azim_idx = ground_truth['azimuth'] / 5.0
        gt_elev_idx = ground_truth['elevation'] / 10.0
        ax1.plot(gt_azim_idx, gt_elev_idx, 'r*', markersize=20,
                 label=f'Ground Truth ({ground_truth["azimuth"]}°, {ground_truth["elevation"]}°)')
        ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1, label='Probability')

    # 2. Azimuth marginal
    ax2 = plt.subplot(2, 2, 2)
    azim_probs  = np.sum(prob_grid, axis=0)
    azim_angles = np.arange(0, 360, 5)
    ax2.bar(azim_angles, azim_probs, width=4, color='steelblue', alpha=0.7,
            edgecolor='black')
    ax2.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax2.set_ylabel('Marginal Probability', fontsize=12)
    ax2.set_title('Azimuth Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlim(-5, 360)
    ax2.grid(axis='y', alpha=0.3)
    if ground_truth and 'azimuth' in ground_truth:
        ax2.axvline(ground_truth['azimuth'], color='red', linestyle='--',
                    linewidth=2, label='Ground Truth')
        ax2.legend()

    # 3. Elevation marginal
    ax3 = plt.subplot(2, 2, 3)
    elev_probs  = np.sum(prob_grid, axis=1)
    elev_angles = np.arange(0, 70, 10)
    ax3.barh(elev_angles, elev_probs, height=8, color='coral', alpha=0.7,
             edgecolor='black')
    ax3.set_ylabel('Elevation (degrees)', fontsize=12)
    ax3.set_xlabel('Marginal Probability', fontsize=12)
    ax3.set_title('Elevation Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylim(-5, 145)
    ax3.grid(axis='x', alpha=0.3)
    if ground_truth and 'elevation' in ground_truth:
        ax3.axhline(ground_truth['elevation'], color='red', linestyle='--',
                    linewidth=2, label='Ground Truth')
        ax3.legend()

    # 4. Top-10 table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    top_indices = np.argsort(probabilities)[-10:][::-1]
    top_probs   = probabilities[top_indices]
    table_data  = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        azim_deg, elev_deg = decode_class_index(idx)
        marker = ''
        if ground_truth and 'azimuth' in ground_truth:
            if (idx % 72) == ground_truth['azimuth'] and (idx // 72) == ground_truth['elevation']:
                marker = '✓'
        table_data.append([f'{i}', f'{azim_deg}°', f'{elev_deg}°', f'{prob:.4f}', marker])
    table = ax4.table(cellText=table_data,
                      colLabels=['Rank', 'Azimuth', 'Elevation', 'Prob.', ''],
                      cellLoc='center', loc='center',
                      colWidths=[0.15, 0.22, 0.22, 0.22, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    if ground_truth and 'azimuth' in ground_truth:
        for i, row in enumerate(table_data, 1):
            if row[4] == '✓':
                for j in range(5):
                    table[(i, j)].set_facecolor('#FFEB3B')
    ax4.set_title('Top 10 Predictions', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────
#  Aggregate error plots
# ─────────────────────────────────────────────────────────

def plot_top5_heatmap(ax, az_errs, el_errs, title):
    """2D histogram pcolormesh of top-5 relative errors."""
    h, xedges, yedges = np.histogram2d(
        az_errs, el_errs,
        bins=[np.arange(-180, 185, 5), np.arange(-90, 95, 10)])
    h = h.T
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, h, cmap='viridis')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Azimuth Error', fontsize=12)
    ax.set_ylabel('Elevation Error', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=3, label='Ground Truth Origin')
    ax.legend()
    return im


def plot_aggregate_errors(agg_data, output_path):
    """4-subplot figure: KDE for top-1 az/el errors + 2D heatmaps for top-5.

    Args:
        agg_data: dict keyed by condition ('0click', '1click', …), each value
                  is a dict with keys 'top1_az_errs', 'top1_el_errs',
                  'top5_az_errs', 'top5_el_errs'.
        output_path: file path to save the PNG.
    """
    active = [c for c in agg_data if len(agg_data[c]['top1_az_errs']) > 0]
    if not active:
        print('No valid data collected for aggregation.')
        return

    fig = plt.figure(figsize=(20, 16))
    colors = plt.cm.tab10(np.linspace(0, 1, len(active)))

    ax1 = plt.subplot(2, 2, 1)
    for i, cond in enumerate(active):
        sns.kdeplot(agg_data[cond]['top1_az_errs'], fill=True, color=colors[i],
                    alpha=0.5, label=cond, ax=ax1)
    ax1.set_title('Top-1 Relative Azimuth Error Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Error in Degrees (Predicted − Ground Truth)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    for i, cond in enumerate(active):
        sns.kdeplot(agg_data[cond]['top1_el_errs'], fill=True, color=colors[i],
                    alpha=0.5, label=cond, ax=ax2)
    ax2.set_title('Top-1 Relative Elevation Error Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Error in Degrees (Predicted − Ground Truth)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.legend()

    if len(active) >= 1:
        ax3 = plt.subplot(2, 2, 3)
        cond1 = active[0]
        im3 = plot_top5_heatmap(ax3, agg_data[cond1]['top5_az_errs'],
                                 agg_data[cond1]['top5_el_errs'],
                                 f'{cond1} Top-5 Relative Error')
        plt.colorbar(im3, ax=ax3, label='Count')

    if len(active) >= 2:
        ax4 = plt.subplot(2, 2, 4)
        cond2 = active[1]
        im4 = plot_top5_heatmap(ax4, agg_data[cond2]['top5_az_errs'],
                                 agg_data[cond2]['top5_el_errs'],
                                 f'{cond2} Top-5 Relative Error')
        plt.colorbar(im4, ax=ax4, label='Count')

    plt.tight_layout()
    agg_dir = os.path.dirname(output_path)
    if agg_dir and not os.path.exists(agg_dir):
        os.makedirs(agg_dir)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\n{"="*46}')
    print(f'Aggregate visualization saved to: {output_path}')
    print(f'{"="*46}')
    plt.close()


# ─────────────────────────────────────────────────────────
#  Activation map visualization
# ─────────────────────────────────────────────────────────

def plot_activation_maps(layer_outputs, layer_names, output_path, max_channels=16):
    """Visualize intermediate CNN activation maps; one PNG per layer.

    For convolutional layers (4-D tensors shaped [1, freq, time, channels] or
    [freq, time, channels]):
        Displays a grid of up to *max_channels* feature maps.  Each map is a
        freq × time image coloured with the 'viridis' colormap.  Channels are
        sampled evenly when the layer has more than *max_channels* channels.

    For dense / FC layers (2-D tensors shaped [1, N] or 1-D shaped [N]):
        Left panel — bar chart of all unit activations.
        Right panel — histogram of the activation distribution with mean marked.

    Output files are written as:
        <base>_<safe_layer_name><ext>
    where <base> and <ext> are derived from *output_path*
    (e.g. output_path='plots/act.png' → 'plots/act_conv1.png').

    Args:
        layer_outputs: dict mapping layer name → numpy array (batch dim optional).
        layer_names:   list of layer names to visualise (subset of layer_outputs).
        output_path:   base file path; extension sets the image format.
        max_channels:  maximum number of channels to show for conv layers.

    Returns:
        List of file paths that were saved.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = '.png'

    saved_paths = []

    for layer_name in layer_names:
        if layer_name not in layer_outputs:
            print(f'  [activation] Warning: layer "{layer_name}" not in layer_outputs, skipping.')
            continue

        activation = np.array(layer_outputs[layer_name])

        # Strip batch dimension
        if activation.ndim == 4:
            activation = activation[0]   # → (freq, time, channels)
        elif activation.ndim == 2:
            activation = activation[0]   # → (N,)

        safe_name = layer_name.replace('/', '_').replace(':', '_').replace(' ', '_')
        out_path  = f'{base}_{safe_name}{ext}'

        # ── Convolutional feature maps ──────────────────
        if activation.ndim == 3:
            H, W, C = activation.shape
            n_show  = min(C, max_channels)
            indices = np.linspace(0, C - 1, n_show, dtype=int)

            ncols = min(n_show, 8)
            nrows = (n_show + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 2.8, nrows * 2.8 + 1.2),
                                     squeeze=False)
            fig.suptitle(
                f'Activations — {layer_name}\n'
                f'Shape: ({H} freq bins, {W} time steps, {C} channels)'
                + (f'  [showing {n_show}/{C}]' if n_show < C else ''),
                fontsize=11, fontweight='bold')

            flat_axes = axes.flatten()
            for i, ch_idx in enumerate(indices):
                ax   = flat_axes[i]
                fmap = activation[:, :, ch_idx]
                vmin, vmax = fmap.min(), fmap.max()
                im = ax.imshow(fmap, aspect='auto', cmap='viridis',
                               vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(f'Ch {ch_idx}', fontsize=8)
                ax.set_xlabel('Time', fontsize=6)
                ax.set_ylabel('Freq', fontsize=6)
                ax.tick_params(labelsize=5)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for i in range(n_show, len(flat_axes)):
                flat_axes[i].set_visible(False)

        # ── Dense / FC activations ──────────────────────
        elif activation.ndim == 1:
            N = activation.shape[0]
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            fig.suptitle(
                f'Activations — {layer_name}\nShape: ({N} units)',
                fontsize=11, fontweight='bold')

            ax1 = axes[0]
            ax1.bar(np.arange(N), activation, color='steelblue', alpha=0.7, width=1.0)
            ax1.set_xlabel('Unit Index', fontsize=10)
            ax1.set_ylabel('Activation', fontsize=10)
            ax1.set_title('All Unit Activations', fontsize=11)
            ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax1.grid(axis='y', alpha=0.3)

            ax2 = axes[1]
            ax2.hist(activation, bins=min(50, N // 4 + 1),
                     color='coral', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Activation Value', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.set_title('Activation Distribution', fontsize=11)
            ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='zero')
            ax2.axvline(activation.mean(), color='blue', linestyle='--', alpha=0.8,
                        label=f'mean = {activation.mean():.3f}')
            frac_active = (activation > 0).mean()
            ax2.text(0.97, 0.95, f'Active: {frac_active*100:.1f}%',
                     transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

        else:
            print(f'  [activation] Unexpected shape {activation.shape} for "{layer_name}", skipping.')
            continue

        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_paths.append(out_path)
        print(f'  Saved activation map: {out_path}')

    print(f'\nActivation maps saved: {len(saved_paths)} file(s)')
    return saved_paths


# ─────────────────────────────────────────────────────────
#  Unit tuning maps  (class-conditional activation aggregation)
# ─────────────────────────────────────────────────────────

# Shared axis helpers
_AZIM_TICKS       = np.arange(0, 72, 12)          # every 60 degrees
_AZIM_TICK_LABELS = [f'{int(t*5)}°' for t in _AZIM_TICKS]
_ELEV_TICKS       = np.arange(7)                   # all 7 bins
_ELEV_TICK_LABELS = [f'{int(t*10)}°' for t in _ELEV_TICKS]


def _apply_spatial_axes(ax, fontsize=6):
    ax.set_xticks(_AZIM_TICKS)
    ax.set_xticklabels(_AZIM_TICK_LABELS, fontsize=fontsize)
    ax.set_yticks(_ELEV_TICKS)
    ax.set_yticklabels(_ELEV_TICK_LABELS, fontsize=fontsize)
    ax.set_xlabel('Azimuth', fontsize=fontsize + 1)
    ax.set_ylabel('Elevation', fontsize=fontsize + 1)


def plot_unit_tuning_grid(tuning_maps, layer_name, output_path,
                          selectivity_scores=None, max_units=64,
                          total_samples=0):
    """Grid of dual-panel unit cards: azimuth tuning curve + 7×72 heatmap inset.

    Each card (like dissect's image-patch grid) shows:
      - Top panel: 1-D azimuth tuning curve (mean over elevation) as a bar chart
        → readable at small sizes, immediately shows spatial selectivity
      - Bottom panel: 7×72 heatmap inset showing elevation dependence

    Args:
        tuning_maps:        np.ndarray (N_units, 7, 72) — mean activation per class
        layer_name:         str, used in the figure supertitle
        output_path:        file path to save PNG
        selectivity_scores: optional np.ndarray (N_units,) — variance across classes
        max_units:          cap on units shown; sampled evenly when N_units > max_units
        total_samples:      int, shown in supertitle for reference
    """
    N_units = tuning_maps.shape[0]
    n_show  = min(N_units, max_units)
    indices = np.linspace(0, N_units - 1, n_show, dtype=int)

    ncols = min(n_show, 8)
    nrows = math.ceil(n_show / ncols)

    # Each card is 2 sub-rows: bar chart (top) + heatmap (bottom)
    fig = plt.figure(figsize=(ncols * 2.5, nrows * 3.4 + 1.2))
    shown = f'showing {n_show}/{N_units}' if n_show < N_units else f'{N_units} units'
    samp  = f', {total_samples} samples' if total_samples else ''
    fig.suptitle(f'Unit Tuning — {layer_name}  ({shown}{samp})',
                 fontsize=11, fontweight='bold', y=1.01)

    # Shared colour scale for heatmaps (NaN → light grey)
    shown_maps = tuning_maps[indices]
    vmin = float(np.nanmin(shown_maps)) if not np.all(np.isnan(shown_maps)) else 0.0
    vmax = float(np.nanmax(shown_maps)) if not np.all(np.isnan(shown_maps)) else 1.0
    cmap_heat = mcolors.LinearSegmentedColormap.from_list(
        'plasma_nan', plt.cm.plasma(np.linspace(0, 1, 256)))
    cmap_heat.set_bad('#c8c8c8')          # NaN → light grey

    az_bins = np.arange(72) * 5           # 0, 5, …, 355 degrees

    outer = matplotlib.gridspec.GridSpec(nrows, ncols, figure=fig,
                                         hspace=0.55, wspace=0.35)

    for i, unit_idx in enumerate(indices):
        row, col = divmod(i, ncols)
        inner = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col], hspace=0.15,
            height_ratios=[2.2, 1])

        tmap      = tuning_maps[unit_idx]          # (7, 72)
        az_curve  = np.nanmean(tmap, axis=0)       # (72,)  — mean over elevation
        peak_az   = int(az_bins[np.nanargmax(az_curve)]) if not np.all(np.isnan(az_curve)) else -1

        # ── Top: azimuth tuning bar chart ────────────────
        ax_bar = fig.add_subplot(inner[0])
        colors = plt.cm.plasma(
            (az_curve - vmin) / max(vmax - vmin, 1e-9))
        ax_bar.bar(az_bins, np.nan_to_num(az_curve), width=4,
                   color=colors, edgecolor='none')
        ax_bar.set_xlim(-5, 360)
        ax_bar.set_xticks([0, 90, 180, 270, 355])
        ax_bar.set_xticklabels(['0', '90', '180', '270', '355'], fontsize=4)
        ax_bar.tick_params(axis='y', labelsize=4)
        ax_bar.set_xlabel('Az (°)', fontsize=4, labelpad=1)
        ax_bar.grid(axis='y', alpha=0.25, linewidth=0.5)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        title = f'U{unit_idx}'
        if selectivity_scores is not None:
            title += f'  sel={selectivity_scores[unit_idx]:.3f}'
        if peak_az >= 0:
            title += f'  peak={peak_az}°'
        ax_bar.set_title(title, fontsize=6, pad=2)

        # ── Bottom: 7×72 heatmap inset ───────────────────
        ax_map = fig.add_subplot(inner[1])
        ax_map.imshow(tmap, aspect='auto', cmap=cmap_heat,
                      vmin=vmin, vmax=vmax,
                      origin='lower', interpolation='nearest')
        ax_map.set_xticks([0, 18, 36, 54, 71])
        ax_map.set_xticklabels(['0°', '90°', '180°', '270°', '355°'], fontsize=3.5)
        ax_map.set_yticks([0, 3, 6])
        ax_map.set_yticklabels(['0°', '30°', '60°'], fontsize=3.5)
        ax_map.tick_params(length=2, pad=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved tuning grid: {output_path}')


def plot_layer_attribution_map(tuning_maps_dict, output_path, total_samples=0):
    """Primary cross-layer summary: for each spatial position, which layer responds most.

    Produces a 3-panel figure:
      Left:   7×72 grid coloured by the layer index whose units have the highest
              max activation for that position  ("winner-takes-all by layer")
      Centre: 7×72 grid showing the actual peak activation value at each position
              (max over all units and all layers)
      Right:  Per-layer stacked bar chart of unit spatial-preference distribution
              (how many units in each layer prefer each azimuth sector — the
              direct analogue of dissect's object/part/color concept histogram)

    Args:
        tuning_maps_dict: {layer_name: np.ndarray (N_units, 7, 72)}
        output_path:      file path to save PNG
        total_samples:    shown in title
    """
    layer_names = list(tuning_maps_dict.keys())
    n_layers    = len(layer_names)

    # ── Per-layer max over units: (n_layers, 7, 72) ──────
    layer_max = np.stack(
        [np.nanmax(tuning_maps_dict[l], axis=0) for l in layer_names], axis=0)
    # nan-safe argmax: winner layer per spatial position
    valid     = ~np.all(np.isnan(layer_max), axis=0)        # (7, 72) bool
    winner    = np.full((7, 72), -1, dtype=int)
    if valid.any():
        winner[valid] = np.nanargmax(layer_max[:, valid], axis=0)
    peak_act  = np.nanmax(layer_max, axis=0)                # (7, 72)

    # ── Colour map: one colour per layer ─────────────────
    layer_cmap  = plt.cm.get_cmap('tab10', n_layers)
    layer_colors = [layer_cmap(i) for i in range(n_layers)]

    # ── Spatial-preference distribution per layer ────────
    # Sectors: Front(0±45°→315-45), Right(45-135), Back(135-225), Left(225-315)
    sector_edges = [(-45, 45), (45, 135), (135, 225), (225, 315)]
    sector_names = ['Front\n(±45°)', 'Right\n(45-135°)',
                    'Back\n(135-225°)', 'Left\n(225-315°)']
    sector_colors = ['#4e9af1', '#f4a261', '#2ecc71', '#e74c3c']

    def _az_to_sector(az_deg):
        az = az_deg % 360
        if az > 315 or az <= 45:  return 0   # front
        if az <= 135:             return 1   # right
        if az <= 225:             return 2   # back
        return 3                             # left

    # For each layer: count units with clear peak in each sector
    sector_counts = np.zeros((n_layers, 4), dtype=int)
    for li, lname in enumerate(layer_names):
        tmap = tuning_maps_dict[lname]       # (N_units, 7, 72)
        az_mean = np.nanmean(tmap, axis=1)   # (N_units, 72) — mean over elevation
        for unit_az in az_mean:
            if np.all(np.isnan(unit_az)):
                continue
            peak_bin = int(np.nanargmax(unit_az))
            sector_counts[li, _az_to_sector(peak_bin * 5)] += 1

    # ── Figure ───────────────────────────────────────────
    samp = f'  ({total_samples} samples)' if total_samples else ''
    fig  = plt.figure(figsize=(16, 5.5))
    fig.suptitle(f'Cross-layer spatial attribution{samp}',
                 fontsize=13, fontweight='bold')
    gs   = matplotlib.gridspec.GridSpec(1, 3, figure=fig,
                                        width_ratios=[1.4, 1.4, 2.2],
                                        wspace=0.38)

    # ── Panel 1: winner-layer map ─────────────────────────
    ax1 = fig.add_subplot(gs[0])
    # Build RGBA image
    rgba = np.ones((7, 72, 4))
    for row in range(7):
        for col in range(72):
            li = winner[row, col]
            rgba[row, col] = layer_colors[li] if li >= 0 else (0.82, 0.82, 0.82, 1)
    ax1.imshow(rgba, aspect='auto', origin='lower', interpolation='nearest')
    _apply_spatial_axes(ax1, fontsize=7)
    ax1.set_title('Dominant layer per position\n(winner-takes-all)', fontsize=9)
    # Legend
    handles = [matplotlib.patches.Patch(color=layer_colors[i], label=layer_names[i])
               for i in range(n_layers)]
    handles.append(matplotlib.patches.Patch(color='#d1d1d1', label='(no data)'))
    ax1.legend(handles=handles, fontsize=6, loc='upper right',
               framealpha=0.7, bbox_to_anchor=(1.0, -0.14), ncol=2)

    # ── Panel 2: peak activation magnitude map ───────────
    ax2 = fig.add_subplot(gs[1])
    cmap2 = mcolors.LinearSegmentedColormap.from_list(
        'plasma_nan', plt.cm.plasma(np.linspace(0, 1, 256)))
    cmap2.set_bad('#d1d1d1')
    im2 = ax2.imshow(peak_act, aspect='auto', cmap=cmap2,
                     origin='lower', interpolation='nearest')
    _apply_spatial_axes(ax2, fontsize=7)
    ax2.set_title('Peak activation magnitude\n(max over all units & layers)', fontsize=9)
    plt.colorbar(im2, ax=ax2, shrink=0.7, label='Max mean activation')

    # ── Panel 3: spatial-preference distribution per layer ─
    ax3 = fig.add_subplot(gs[2])
    bar_h   = 0.65
    y_pos   = np.arange(n_layers)
    lefts   = np.zeros(n_layers)
    total_u = sector_counts.sum(axis=1).clip(min=1)
    for si, (sname, scol) in enumerate(zip(sector_names, sector_colors)):
        fracs = sector_counts[:, si] / total_u
        bars  = ax3.barh(y_pos, fracs, left=lefts, height=bar_h,
                         color=scol, label=sname, edgecolor='none')
        # Annotate non-trivial fractions
        for bar, frac in zip(bars, fracs):
            if frac > 0.08:
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_y() + bar.get_height() / 2,
                         f'{frac:.0%}', ha='center', va='center',
                         fontsize=7, color='white', fontweight='bold')
        lefts += fracs

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(layer_names, fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel('Fraction of units', fontsize=9)
    ax3.set_title('Unit spatial-preference distribution per layer\n'
                  '(analogue of dissect concept histogram)', fontsize=9)
    ax3.legend(loc='lower right', fontsize=7, framealpha=0.7,
               ncol=2, bbox_to_anchor=(1.0, -0.22))
    ax3.grid(axis='x', alpha=0.25)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved attribution map: {output_path}')


def plot_selectivity_ranking(selectivity_scores, layer_name, output_path,
                             top_k=20):
    """Horizontal bar chart of units ranked by selectivity (variance across classes).

    Args:
        selectivity_scores: np.ndarray (N_units,)
        layer_name:         str
        output_path:        file path to save PNG
        top_k:              top-k bars highlighted in darkorange
    """
    N = len(selectivity_scores)
    order  = np.argsort(selectivity_scores)[::-1]   # descending
    scores = selectivity_scores[order]
    colors = ['darkorange' if i < top_k else 'steelblue' for i in range(N)]

    fig_h = max(4, N * 0.18)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    bars = ax.barh(np.arange(N), scores, color=colors, edgecolor='none', height=0.8)
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels([f'U{i}' for i in order], fontsize=max(4, 8 - N // 40))
    ax.invert_yaxis()
    ax.set_xlabel('Variance of mean activation across 504 classes', fontsize=10)
    ax.set_title(f'Unit Selectivity Ranking — {layer_name}\n'
                 f'Top {top_k} highlighted', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Annotate top-k values
    for i in range(min(top_k, N)):
        ax.text(scores[i] + scores[0] * 0.01, i, f'{scores[i]:.4f}',
                va='center', fontsize=max(5, 7 - N // 60))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved selectivity ranking: {output_path}')


def plot_topk_mosaic(cochleagrams, activations, unit_idx, layer_name,
                     output_path, activation_maps=None):
    """k-panel figure: cochleagram thumbnails for the top-k activating samples.

    Each panel shows the left-ear cochleagram (log-scaled for display).
    If activation_maps is provided, overlays the unit's spatial activation as
    a semi-transparent heatmap (only meaningful for conv layers).

    Args:
        cochleagrams:    list of np.ndarray (39, 8000, 2), length k
        activations:     list of float — max-pooled activation for each sample
        unit_idx:        int
        layer_name:      str
        output_path:     file path to save PNG
        activation_maps: optional list of np.ndarray (H, W) — per-sample
                         spatial activation maps before max-pooling;
                         None for FC/flatten layers
    """
    k     = len(cochleagrams)
    ncols = min(k, 4)
    nrows = math.ceil(k / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 2.8 + 0.8),
                             squeeze=False)
    fig.suptitle(f'Top-{k} activating samples — Unit {unit_idx} ({layer_name})',
                 fontsize=11, fontweight='bold')

    flat_axes = axes.flatten()
    for i, (coch, act_val) in enumerate(zip(cochleagrams, activations)):
        ax = flat_axes[i]
        # Left ear, log-compressed for display
        left_ear = coch[:, :, 0]
        display  = np.log1p(np.maximum(left_ear, 0))
        ax.imshow(display, aspect='auto', cmap='inferno',
                  origin='lower', interpolation='nearest')

        if activation_maps is not None and activation_maps[i] is not None:
            amap = activation_maps[i]
            # Upsample spatial map to cochleagram resolution
            from scipy.ndimage import zoom as nd_zoom
            scale_h = left_ear.shape[0] / amap.shape[0]
            scale_w = left_ear.shape[1] / amap.shape[1]
            upsampled = nd_zoom(amap, (scale_h, scale_w), order=1)
            # Normalise to [0, 1] and overlay
            u_norm = (upsampled - upsampled.min()) / (upsampled.ptp() + 1e-9)
            ax.imshow(u_norm, aspect='auto', cmap='hot', alpha=0.45,
                      origin='lower', interpolation='nearest')

        ax.set_title(f'Rank {i+1}  act={act_val:.3f}', fontsize=8)
        ax.set_xlabel('Time steps', fontsize=6)
        ax.set_ylabel('Freq bin', fontsize=6)
        ax.tick_params(labelsize=5)

    for i in range(k, len(flat_axes)):
        flat_axes[i].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved top-k mosaic: {output_path}')


# ─────────────────────────────────────────────────────────
#  HTML report generator
# ─────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
  header {{ padding: 16px 24px; background: #1a1a2e; border-bottom: 1px solid #333; }}
  header h1 {{ margin: 0 0 4px; font-size: 1.3em; }}
  header p  {{ margin: 0; font-size: 0.85em; color: #aaa; }}
  #controls {{ display: flex; gap: 12px; align-items: center; padding: 10px 24px;
               background: #16213e; flex-wrap: wrap; }}
  #controls label {{ font-size: 0.85em; color: #ccc; }}
  #controls select, #controls input {{ background: #222; color: #eee;
               border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }}
  #layer-tabs {{ display: flex; gap: 6px; padding: 10px 24px 0; flex-wrap: wrap; }}
  .tab {{ padding: 6px 14px; border-radius: 6px 6px 0 0; cursor: pointer;
          background: #222; border: 1px solid #444; border-bottom: none;
          font-size: 0.85em; }}
  .tab.active {{ background: #0f3460; color: #fff; border-color: #0f3460; }}
  #grid {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 14px 24px; }}
  .card {{ background: #1e1e2e; border: 1px solid #333; border-radius: 8px;
           width: 180px; cursor: pointer; transition: border-color .15s; }}
  .card:hover {{ border-color: #7b68ee; }}
  .card img {{ width: 100%; border-radius: 8px 8px 0 0; display: block; }}
  .card-info {{ padding: 6px 8px; font-size: 0.78em; }}
  .card-info .unit  {{ font-weight: bold; color: #a9d4ff; }}
  .card-info .score {{ color: #f4a261; }}
  .detail {{ display: none; background: #12122a; border: 1px solid #444;
             border-radius: 8px; padding: 16px; margin: 0 24px 16px;
             width: calc(100% - 96px); }}
  .detail.open {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .detail img {{ max-width: 480px; border-radius: 6px; }}
  .detail h3  {{ margin: 0 0 10px; font-size: 1em; color: #a9d4ff; width: 100%; }}
  .topk-row   {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
  .topk-row img {{ width: 140px; border-radius: 4px; }}
  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
<div id="controls">
  <label>Sort by:
    <select id="sort-sel" onchange="applySort()">
      <option value="sel-desc">Selectivity ↓</option>
      <option value="sel-asc">Selectivity ↑</option>
      <option value="idx-asc">Unit index ↑</option>
    </select>
  </label>
  <label>Filter unit:
    <input id="filter-input" type="text" placeholder="e.g. 42"
           oninput="applyFilter()" style="width:80px">
  </label>
  <span id="count-label" style="font-size:0.82em;color:#aaa;"></span>
</div>
<div id="layer-tabs">{layer_tabs}</div>
<div id="grid"></div>
<div id="detail-panel"></div>

<script>
const UNITS = {units_json};
let activeLayer = UNITS.length ? UNITS[0].layer : '';

function renderTabs() {{
  const layers = [...new Set(UNITS.map(u => u.layer))];
  document.getElementById('layer-tabs').innerHTML = layers.map(l =>
    `<div class="tab${{l===activeLayer?' active':''}}" onclick="switchLayer('${{l}}')">${{l}}</div>`
  ).join('');
}}

function switchLayer(l) {{
  activeLayer = l;
  renderTabs();
  renderGrid();
  document.getElementById('detail-panel').innerHTML = '';
}}

function applySort() {{ renderGrid(); }}
function applyFilter() {{ renderGrid(); }}

function visibleUnits() {{
  const filterVal = document.getElementById('filter-input').value.trim();
  const sortVal   = document.getElementById('sort-sel').value;
  let units = UNITS.filter(u => u.layer === activeLayer);
  if (filterVal !== '') units = units.filter(u => String(u.idx).includes(filterVal));
  if (sortVal === 'sel-desc') units.sort((a,b) => b.sel - a.sel);
  else if (sortVal === 'sel-asc') units.sort((a,b) => a.sel - b.sel);
  else units.sort((a,b) => a.idx - b.idx);
  return units;
}}

function renderGrid() {{
  const units = visibleUnits();
  document.getElementById('count-label').textContent =
    `Showing ${{units.length}} unit(s)`;
  document.getElementById('grid').innerHTML = units.map(u => `
    <div class="card" data-uid="${{u.uid}}" onclick="toggleDetail('${{u.uid}}')">
      <img src="${{u.grid_thumb}}" alt="unit ${{u.idx}}" loading="lazy">
      <div class="card-info">
        <span class="unit">U${{u.idx}}</span>
        <span class="score"> sel=${{u.sel.toFixed(4)}}</span>
      </div>
    </div>`).join('');
}}

function toggleDetail(uid) {{
  const panel = document.getElementById('detail-panel');
  if (panel.dataset.uid === uid && panel.querySelector('.detail.open')) {{
    panel.innerHTML = '';
    panel.dataset.uid = '';
    return;
  }}
  panel.dataset.uid = uid;
  const u = UNITS.find(x => x.uid === uid);
  if (!u) return;
  let topkHtml = '';
  if (u.topk_imgs && u.topk_imgs.length) {{
    topkHtml = `<div style="width:100%"><strong>Top-${{u.topk_imgs.length}} activating samples:</strong>
      <div class="topk-row">${{u.topk_imgs.map(src =>
        `<img src="${{src}}" loading="lazy">`).join('')}}</div></div>`;
  }}
  let indivHtml = u.indiv_img
    ? `<img src="${{u.indiv_img}}" alt="full tuning map">`
    : '';
  panel.innerHTML = `
    <div class="detail open">
      <h3>Unit ${{u.idx}} &mdash; ${{u.layer}} &mdash; selectivity ${{u.sel.toFixed(4)}}</h3>
      ${{indivHtml}}
      ${{topkHtml}}
    </div>`;
  panel.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}

renderTabs();
renderGrid();
</script>
</body>
</html>
"""


def generate_tuning_report(output_dir, title=None):
    """Read an analyze_unit_tuning.py output directory and write index.html.

    Reads (all relative to output_dir):
        tuning_maps.npz            — selectivity scores, layer names, unit counts
        tuning_grid_<layer>.png    — used as card thumbnails
        units/<layer>/unit_NNN.png — linked from expanded card view (optional)
        mosaics/<layer>/unit_NNN_topk.png — top-k cochleagram mosaics (optional)

    Args:
        output_dir: str — directory produced by analyze_unit_tuning.py
        title:      str or None — report title; defaults to basename of output_dir
    """
    npz_path = os.path.join(output_dir, 'tuning_maps.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f'tuning_maps.npz not found in {output_dir}. '
            'Run analyze_unit_tuning.py first.')

    data  = np.load(npz_path, allow_pickle=True)
    title = title or os.path.basename(os.path.abspath(output_dir))

    # Collect layer names present in the npz
    layer_names = [k for k in data.files if not k.endswith('_selectivity')]

    all_units = []
    for layer in layer_names:
        tmap   = data[layer]               # (N_units, 7, 72)
        sel_key = f'{layer}_selectivity'
        if sel_key in data.files:
            sel = data[sel_key]
        else:
            # Recompute from tuning maps if not stored
            flat = tmap.reshape(tmap.shape[0], -1).astype(np.float64)
            sel  = np.nanvar(flat, axis=1)

        N_units = tmap.shape[0]
        safe_layer = layer.replace('/', '_').replace(':', '_')

        for unit_idx in range(N_units):
            uid = f'{safe_layer}_u{unit_idx:04d}'

            # Card thumbnail: the per-layer grid PNG (same for all units in layer)
            grid_thumb = f'tuning_grid_{safe_layer}.png'

            # Individual unit PNG (optional)
            indiv_path = os.path.join(
                output_dir, 'units', safe_layer, f'unit_{unit_idx:04d}.png')
            indiv_img = (f'units/{safe_layer}/unit_{unit_idx:04d}.png'
                         if os.path.exists(indiv_path) else '')

            # Top-k mosaic PNG (optional)
            topk_path = os.path.join(
                output_dir, 'mosaics', safe_layer,
                f'unit_{unit_idx:04d}_topk.png')
            topk_imgs = ([f'mosaics/{safe_layer}/unit_{unit_idx:04d}_topk.png']
                         if os.path.exists(topk_path) else [])

            all_units.append({
                'uid':        uid,
                'layer':      safe_layer,
                'idx':        int(unit_idx),
                'sel':        float(sel[unit_idx]),
                'grid_thumb': grid_thumb,
                'indiv_img':  indiv_img,
                'topk_imgs':  topk_imgs,
            })

    n_layers = len(layer_names)
    n_units  = len(all_units)
    subtitle = (f'{n_layers} layer(s), {n_units} unit(s) total — '
                f'generated by analyze_unit_tuning.py')

    layer_tabs_html = ''.join(
        '<div class="tab" onclick="switchLayer(\'{}\')">'.format(
            l.replace('/', '_').replace(':', '_')) + l + '</div>'
        for l in layer_names)

    html = _HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        layer_tabs=layer_tabs_html,
        units_json=json.dumps(all_units, separators=(',', ':')),
    )

    out_path = os.path.join(output_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report written → {out_path}')
    return out_path


# ─────────────────────────────────────────────────────────
#  Input-level attribution visualizations (Simonyan [57],
#  Zeiler & Fergus [55], Selvaraju Grad-CAM)
# ─────────────────────────────────────────────────────────

def plot_saliency_overlay(cochleagram, saliency_map, output_path,
                          predicted_class=None, ground_truth=None,
                          confidence=None, method_name='Vanilla Saliency'):
    """Overlay a saliency map on the input cochleagram.

    Produces a 2×2 figure:
      Top-left:   left-ear cochleagram with saliency overlay
      Top-right:  right-ear cochleagram with saliency overlay
      Bottom-left:  saliency magnitude by frequency band (marginal)
      Bottom-right: saliency magnitude by time region (marginal)

    Args:
        cochleagram:     np.ndarray (39, 8000, 2)
        saliency_map:    np.ndarray (39, 8000, 2) — same shape as input
        output_path:     file path to save PNG
        predicted_class: optional int (0–503)
        ground_truth:    optional dict with 'azimuth', 'elevation'
        confidence:      optional float
        method_name:     str for the title
    """
    from scipy.ndimage import gaussian_filter

    fig, axes = plt.subplots(2, 2, figsize=(18, 8))

    title = method_name
    if predicted_class is not None:
        pred_az, pred_el = decode_class_index(predicted_class)
        title += f'  |  Pred: az={pred_az}° el={pred_el}°'
    if ground_truth and 'azimuth' in ground_truth:
        title += f'  |  GT: az={ground_truth["azimuth"]}° el={ground_truth["elevation"]}°'
    if confidence is not None:
        title += f'  |  Conf: {confidence:.4f}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ear_names = ['Left Ear', 'Right Ear']
    sal_abs = np.abs(saliency_map)

    for ear_idx in range(2):
        ax = axes[0, ear_idx]

        # Cochleagram as base image (full opacity, grayscale)
        coch_display = np.log1p(np.maximum(cochleagram[:, :, ear_idx], 0))
        coch_norm = coch_display / (coch_display.max() + 1e-12)

        # Saliency: Gaussian smooth to reduce pixel noise, then
        # clip at 99th percentile to avoid outlier-dominated normalization
        sal = sal_abs[:, :, ear_idx].copy()
        sal = gaussian_filter(sal, sigma=(1.0, 20.0))  # smooth: ~1 freq bin, ~20 time steps
        p99 = np.percentile(sal, 99)
        if p99 > 0:
            sal = np.clip(sal / p99, 0, 1)
        else:
            sal = sal / (sal.max() + 1e-12)

        # Composite: grayscale cochleagram blended with hot saliency
        # where saliency is strong, show the hot colormap; otherwise show cochleagram
        cmap_hot = plt.cm.inferno
        sal_rgb = cmap_hot(sal)[:, :, :3]  # (39, 8000, 3)
        coch_rgb = plt.cm.gray(coch_norm)[:, :, :3]
        blend_weight = sal[:, :, np.newaxis] ** 0.7  # nonlinear blend for contrast
        composited = coch_rgb * (1 - blend_weight) + sal_rgb * blend_weight

        ax.imshow(composited, aspect='auto', origin='lower', interpolation='nearest')
        ax.set_title(f'{ear_names[ear_idx]} — Saliency Overlay', fontsize=10)
        ax.set_xlabel('Time step', fontsize=8)
        ax.set_ylabel('Frequency bin', fontsize=8)
        ax.tick_params(labelsize=6)

    # Frequency marginal (sum over time, then normalize per-ear)
    ax = axes[1, 0]
    freq_marginal = sal_abs.sum(axis=1)  # (39, 2)
    # Normalize so bars are visible relative to each other
    fm_max = freq_marginal.max() + 1e-12
    freq_marginal_norm = freq_marginal / fm_max
    freq_bins = np.arange(39)
    ax.barh(freq_bins, freq_marginal_norm[:, 0], height=0.8, alpha=0.7,
            color='steelblue', label='Left ear')
    ax.barh(freq_bins, freq_marginal_norm[:, 1], height=0.5, alpha=0.7,
            color='coral', label='Right ear')
    ax.set_ylabel('Frequency bin', fontsize=9)
    ax.set_xlabel('Relative saliency (normalized)', fontsize=9)
    ax.set_title('Saliency by frequency band', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.05)

    # Time marginal (sum over frequency, then normalize)
    ax = axes[1, 1]
    time_marginal = sal_abs.sum(axis=0)  # (8000, 2)
    # Downsample for readability
    n_bins = 100
    bin_size = time_marginal.shape[0] // n_bins
    time_binned = np.array([time_marginal[i*bin_size:(i+1)*bin_size].sum(axis=0)
                            for i in range(n_bins)])
    # Normalize
    tb_max = time_binned.max() + 1e-12
    time_binned_norm = time_binned / tb_max
    x = np.arange(n_bins) * bin_size
    ax.fill_between(x, time_binned_norm[:, 0], alpha=0.5, color='steelblue', label='Left ear')
    ax.fill_between(x, time_binned_norm[:, 1], alpha=0.5, color='coral', label='Right ear')
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Relative saliency (normalized)', fontsize=9)
    ax.set_title('Saliency by time region', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved saliency overlay: {output_path}')


def plot_gradcam_ladder(cochleagram, gradcam_maps, layer_names, output_path,
                        predicted_class=None, ground_truth=None):
    """Grad-CAM attention ladder: one row per conv layer, overlaid on cochleagram.

    Shows how the network's spatial attention evolves from early
    (fine-grained temporal/spectral) to deep (coarse pattern) layers.

    Args:
        cochleagram:     np.ndarray (39, 8000, 2) — left ear used for background
        gradcam_maps:    list of np.ndarray — one per layer, each upsampled
                         to (39, 8000) via bilinear interpolation
        layer_names:     list of str — friendly names for each layer
        output_path:     file path to save PNG
        predicted_class: optional int (0–503)
        ground_truth:    optional dict with 'azimuth', 'elevation'
    """
    n_layers = len(gradcam_maps)
    fig, axes = plt.subplots(n_layers + 1, 1,
                             figsize=(16, 2.2 * (n_layers + 1) + 0.8),
                             gridspec_kw={'hspace': 0.35})

    title = 'Grad-CAM Layer Attention Ladder'
    if predicted_class is not None:
        pred_az, pred_el = decode_class_index(predicted_class)
        title += f'  |  Pred: az={pred_az}° el={pred_el}°'
    if ground_truth and 'azimuth' in ground_truth:
        title += f'  |  GT: az={ground_truth["azimuth"]}° el={ground_truth["elevation"]}°'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)

    # Top row: raw cochleagram (left ear)
    ax = axes[0]
    left_ear = np.log1p(np.maximum(cochleagram[:, :, 0], 0))
    ax.imshow(left_ear, aspect='auto', cmap='inferno',
              origin='lower', interpolation='nearest')
    ax.set_title('Input cochleagram (left ear)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Freq', fontsize=8)
    ax.tick_params(labelsize=5)

    # One row per layer
    for i, (gcam, lname) in enumerate(zip(gradcam_maps, layer_names)):
        ax = axes[i + 1]
        ax.imshow(left_ear, aspect='auto', cmap='gray',
                  origin='lower', interpolation='nearest', alpha=0.4)

        gcam_norm = gcam / (gcam.max() + 1e-12)
        im = ax.imshow(gcam_norm, aspect='auto', cmap='jet', alpha=0.65,
                       origin='lower', interpolation='nearest',
                       vmin=0, vmax=1)
        ax.set_title(f'{lname} — Grad-CAM', fontsize=9)
        ax.set_ylabel('Freq', fontsize=8)
        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    axes[-1].set_xlabel('Time step', fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved Grad-CAM ladder: {output_path}')


def plot_fc_output_attribution(fc_activations, output_weights, predicted_class,
                               output_path, ground_truth=None, top_n=20):
    """Decompose the output prediction into per-FC-unit contributions.

    Left panel:  top-N FC units by contribution magnitude to the predicted class
    Right panel: heatmap of those units' contributions across nearby output classes

    Args:
        fc_activations: np.ndarray (512,) — ReLU'd FC layer output
        output_weights: np.ndarray (512, 504) — wc_out_0
        predicted_class: int (0–503)
        output_path:     file path to save PNG
        ground_truth:    optional dict with 'azimuth', 'elevation'
        top_n:           number of top contributing units to show
    """
    # Per-unit contribution to predicted class
    contributions = fc_activations * output_weights[:, predicted_class]

    # Sort by absolute contribution
    order = np.argsort(np.abs(contributions))[::-1]
    top_units = order[:top_n]
    top_contribs = contributions[top_units]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [1, 1.5]})

    pred_az, pred_el = decode_class_index(predicted_class)
    title = f'FC → Output Attribution  |  Pred: az={pred_az}° el={pred_el}°'
    if ground_truth and 'azimuth' in ground_truth:
        title += f'  |  GT: az={ground_truth["azimuth"]}° el={ground_truth["elevation"]}°'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Left: bar chart of top-N contributions
    ax = axes[0]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_contribs]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_contribs, color=colors, edgecolor='none', height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'U{u}' for u in top_units], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Contribution to predicted class logit', fontsize=9)
    ax.set_title(f'Top-{top_n} FC units', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)

    # Annotate activation values
    for i, (u, c) in enumerate(zip(top_units, top_contribs)):
        act_val = fc_activations[u]
        ax.text(c + (0.01 * np.sign(c) * np.abs(top_contribs).max()),
                i, f'act={act_val:.2f}', va='center', fontsize=6, color='#666')

    # Right: heatmap of top units × nearby classes
    ax = axes[1]
    # Show ±5 azimuth bins around predicted class
    pred_az_bin = predicted_class % 72
    pred_el_bin = predicted_class // 72
    nearby_classes = []
    nearby_labels = []
    for el in range(max(0, pred_el_bin - 1), min(7, pred_el_bin + 2)):
        for az_offset in range(-5, 6):
            az = (pred_az_bin + az_offset) % 72
            cls = el * 72 + az
            nearby_classes.append(cls)
            nearby_labels.append(f'{az*5}°/{el*10}°')

    # Contribution matrix: top_units × nearby_classes
    contrib_matrix = fc_activations[top_units, np.newaxis] * \
                     output_weights[top_units][:, nearby_classes]

    im = ax.imshow(contrib_matrix, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest',
                   vmin=-np.abs(contrib_matrix).max(),
                   vmax=np.abs(contrib_matrix).max())
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels([f'U{u}' for u in top_units], fontsize=6)

    # Show fewer x-tick labels to avoid overlap
    step = max(1, len(nearby_classes) // 15)
    ax.set_xticks(np.arange(0, len(nearby_classes), step))
    ax.set_xticklabels([nearby_labels[i] for i in range(0, len(nearby_classes), step)],
                       fontsize=5, rotation=45, ha='right')
    ax.set_xlabel('Nearby output classes (az/el)', fontsize=8)
    ax.set_title('Unit contributions across nearby classes', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Contribution')

    # Mark the predicted class column
    pred_col = nearby_classes.index(predicted_class) if predicted_class in nearby_classes else -1
    if pred_col >= 0:
        ax.axvline(pred_col, color='gold', linewidth=2, alpha=0.8, linestyle='--')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved FC attribution: {output_path}')


def plot_aggregate_saliency(mean_saliency_by_sector, sector_names, output_path):
    """Mean saliency maps averaged across samples grouped by azimuth sector.

    Args:
        mean_saliency_by_sector: dict {sector_name: np.ndarray (39, 8000, 2)}
        sector_names:            list of str
        output_path:             file path to save PNG
    """
    n_sectors = len(sector_names)
    fig, axes = plt.subplots(n_sectors, 2, figsize=(16, 2.5 * n_sectors + 0.8),
                             squeeze=False)
    fig.suptitle('Mean Saliency by Azimuth Sector', fontsize=13, fontweight='bold')

    ear_names = ['Left Ear', 'Right Ear']
    for row, sname in enumerate(sector_names):
        sal = mean_saliency_by_sector[sname]
        for ear in range(2):
            ax = axes[row, ear]
            sal_ear = np.abs(sal[:, :, ear])
            # Gaussian smooth + percentile normalization (same as per-sample)
            from scipy.ndimage import gaussian_filter
            sal_ear = gaussian_filter(sal_ear, sigma=(1.0, 20.0))
            p99 = np.percentile(sal_ear, 99)
            if p99 > 0:
                sal_ear = np.clip(sal_ear / p99, 0, 1)
            else:
                sal_ear = sal_ear / (sal_ear.max() + 1e-12)
            ax.imshow(sal_ear, aspect='auto', cmap='inferno',
                      origin='lower', interpolation='nearest',
                      vmin=0, vmax=1)
            ax.set_title(f'{sname} — {ear_names[ear]}', fontsize=9)
            ax.set_ylabel('Freq', fontsize=7)
            ax.tick_params(labelsize=5)
    axes[-1, 0].set_xlabel('Time step', fontsize=8)
    axes[-1, 1].set_xlabel('Time step', fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved aggregate saliency: {output_path}')


def plot_layer_attention_profile(attention_profile, layer_names, output_path):
    """Heatmap: per-layer fraction of Grad-CAM activation in freq/time regions.

    Args:
        attention_profile: dict with keys:
            'freq_bands': np.ndarray (n_layers, 3) — low/mid/high freq fractions
            'time_regions': np.ndarray (n_layers, 3) — onset/mid/sustain fractions
        layer_names:      list of str
        output_path:      file path to save PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Layer Attention Profile — Where Each Layer Looks',
                 fontsize=12, fontweight='bold')

    freq_labels = ['Low freq\n(bins 0–12)', 'Mid freq\n(bins 13–25)', 'High freq\n(bins 26–38)']
    time_labels = ['Onset\n(0–33%)', 'Middle\n(33–66%)', 'Sustain\n(66–100%)']

    for ax_idx, (data_key, region_labels, title) in enumerate([
        ('freq_bands', freq_labels, 'Frequency band attention'),
        ('time_regions', time_labels, 'Temporal region attention'),
    ]):
        ax = axes[ax_idx]
        data = attention_profile[data_key]  # (n_layers, 3)
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest', vmin=0, vmax=data.max())
        ax.set_yticks(np.arange(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=9)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(region_labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7, label='Fraction of total activation')
        # Annotate cells
        for i in range(len(layer_names)):
            for j in range(3):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if data[i, j] < data.max() * 0.7 else 'white')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved attention profile: {output_path}')
