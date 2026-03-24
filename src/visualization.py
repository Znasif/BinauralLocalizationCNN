import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
