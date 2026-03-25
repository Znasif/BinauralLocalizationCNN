# Tasks

## In progress

---

## Planned

### Unit tuning analysis (`analyze_unit_tuning.py`)

Implement class-conditional activation aggregation: for each unit in selected
layers, compute its mean activation per azimuth/elevation class across many
tfrecord samples. Produces a 7×72 "tuning map" per unit — showing which region
of azimuth/elevation space makes that unit fire.

Contrast with current `--activation_output` (single sample, raw tensor values):
this aggregates across hundreds/thousands of samples and answers *what spatial
concept each unit has learned*, not just what it outputs for one input.

#### Files to create / modify

| File | Action | What changes |
|---|---|---|
| `analyze_unit_tuning.py` | **Create** | Main entry-point script |
| `src/visualization.py` | **Modify** | Add `plot_unit_tuning_grid` and `plot_selectivity_ranking` |

No changes needed to `src/graph_builder.py`, `src/data_io.py`, or `src/inference.py`.

#### New functions in `src/visualization.py`

```python
def plot_unit_tuning_grid(
    tuning_maps,           # np.ndarray (N_units, 7, 72)
    layer_name,            # str
    output_path,           # str
    selectivity_scores=None,   # optional np.ndarray (N_units,)
    max_units=64,          # cap on units shown; sampled evenly when exceeded
    sample_counts=None,    # optional np.ndarray (504,) for annotation
):
    """Grid of 7×72 viridis heatmaps, one cell per unit.
    Layout: ceil(sqrt(n_show)) cols × rows, shared colorbar strip on right.
    x-axis = azimuth (0–355°, ticks every 60°).
    y-axis = elevation (0–60°, all 7 ticks), origin='lower'.
    Cell title: "U{idx}  sel={score:.3f}" when selectivity_scores provided.
    """

def plot_selectivity_ranking(
    selectivity_scores,    # np.ndarray (N_units,)
    layer_name,            # str
    output_path,           # str
    top_k=20,              # highlight top-k bars in darkorange
):
    """Horizontal bar chart of units ranked by variance across 504 class means.
    Top-k bars: darkorange. Rest: steelblue.
    """
```

#### Functions in `analyze_unit_tuning.py`

```python
def build_layer_tensor_map(graph, layer_names_filter=None):
    """Scan graph for Relu and MaxPool ops; assign friendly names
    (relu_0, relu_1, ..., pool_0, pool_1, ...).
    Returns OrderedDict {friendly_name: tf.Tensor}.
    """

def make_accumulators(layer_tensor_map, n_classes=504):
    """Allocate online accumulation arrays per layer.
    Returns:
        sum_acc:   {layer_name: np.zeros((n_classes, n_units), float64)}
        count_acc: {layer_name: np.zeros(n_classes, int64)}
    n_units read from tensor.get_shape() — static in TF1.
    """

def reduce_activation(raw_activation, is_conv):
    """Reduce raw tensor to 1-D unit vector.
    Conv  (1, H, W, C) → max over H, W → shape (C,)
    FC/1D (1, N)       → shape (N,)
    Spatial max chosen over mean: reflects whether a unit fires at all,
    not the average over mostly-inactive spatial positions.
    """

def accumulate_batch(sum_acc, count_acc, layer_activations, class_idx):
    """Update running sums for one sample."""

def compute_tuning_maps(sum_acc, count_acc):
    """Divide sums by counts; reshape to (N_units, 7, 72).
    Classes with count == 0 → NaN (renders as masked/grey in imshow).
    Returns {layer_name: np.ndarray (N_units, 7, 72)}.
    """

def compute_selectivity(tuning_maps_dict):
    """Per-unit variance across 504 class means.
    Returns {layer_name: np.ndarray (N_units,)}.
    """

def main(): ...
```

#### Accumulator memory (all layers combined)

| Layer | Units | Memory |
|---|---|---|
| relu_0  (Conv1 post-ReLU) | 32  | 32 × 504 × 8 B = 129 KB |
| pool_0  (Pool1 output)    | 32  | 129 KB |
| pool_1  (Pool2 output)    | 64  | 258 KB |
| pool_2  (Pool3 output)    | 128 | 516 KB |
| pool_3  (Pool4 output)    | 256 | 1.0 MB |
| relu_fc (FC post-ReLU)    | 512 | 2.1 MB |
| **Total**                 |     | **~4.2 MB** |

Negligible. No chunking or streaming needed.

#### CLI

```bash
# Quick pilot — 50 samples, verify shapes end-to-end before full run
python analyze_unit_tuning.py \
    --model_dir models/net1 \
    --tfrecord data/train0.tfrecords \
    --output_dir tuning_analysis/ \
    --max_samples 50

# Full run — all shards, GPU
python analyze_unit_tuning.py \
    --model_dir models/net1 \
    --tfrecords_dir data/ \
    --output_dir tuning_analysis/ \
    --max_samples 5000 \
    --use_gpu

# Finetuned checkpoint — top-20 selective units saved individually
python analyze_unit_tuning.py \
    --model_dir models/net1 \
    --checkpoint experiments/exp02_alllayers_lr5e-5_frozenbn/checkpoints1/model.ckpt-6 \
    --tfrecords_dir data/ \
    --output_dir tuning_analysis/exp02/ \
    --save_individual_units --top_k_individual 20

# Compare pre- vs post-finetune (run twice with different --output_dir)
python analyze_unit_tuning.py --model_dir models/net1 \
    --tfrecords_dir data/ --output_dir tuning_analysis/pretrain/ --max_samples 5000
python analyze_unit_tuning.py --model_dir models/net1 \
    --checkpoint experiments/exp02.../model.ckpt-6 \
    --tfrecords_dir data/ --output_dir tuning_analysis/finetuned/ --max_samples 5000
```

All CLI arguments:

| Argument | Default | Notes |
|---|---|---|
| `--model_dir` | required | checkpoint + config_array.npy location |
| `--tfrecord` | — | single .tfrecords file |
| `--tfrecords_dir` | — | directory of .tfrecords shards |
| `--output_dir` | required | where to write PNGs and .npz |
| `--checkpoint` | model.ckpt-100000 | explicit ckpt path override |
| `--max_samples` | all | early-stop generator for pilots |
| `--layers` | pool_0 pool_1 pool_2 pool_3 relu_fc | friendly names from build_layer_tensor_map |
| `--spatial_reduction` | max | 'max' or 'mean' over H×W for conv layers |
| `--max_units_in_grid` | 64 | units shown per layer grid PNG |
| `--save_individual_units` | false | also write one PNG per unit |
| `--top_k_individual` | 20 | only top-k selective units when saving individually |
| `--use_gpu` | false | |
| `--quiet` | false | emit PROGRESS: lines only |

#### Output layout

```
tuning_analysis/
├── tuning_maps.npz              ← raw tuning maps, reload for re-plotting
│                                   np.load(f, allow_pickle=True)['pool_3']
│                                   → shape (256, 7, 72)
├── tuning_grid_relu_0.png       ← grid of 32 tuning maps (Conv1 post-ReLU)
├── tuning_grid_pool_0.png       ← 32 units after Pool1
├── tuning_grid_pool_1.png       ← 64 units after Pool2
├── tuning_grid_pool_2.png       ← 128 units after Pool3 (max 64 shown)
├── tuning_grid_pool_3.png       ← 256 units after Pool4 (max 64 shown)
├── tuning_grid_relu_fc.png      ← 512 FC units (max 64 shown)
├── selectivity_relu_0.png       ← bar chart, units ranked by variance
├── selectivity_pool_0.png
├── selectivity_pool_1.png
├── selectivity_pool_2.png
├── selectivity_pool_3.png
├── selectivity_relu_fc.png
└── units/                       ← only with --save_individual_units
    ├── pool_3/
    │   ├── unit_042.png         ← tuning map + az/el marginals (2-panel)
    │   └── ...
    └── relu_fc/
        └── ...
```

Each `tuning_grid_*.png` cell: a 7×72 viridis heatmap.
Bright = high mean activation for that (azimuth, elevation).
High selectivity score = unit cares about location; near-zero = fires uniformly.

#### Implementation order

1. Add `plot_unit_tuning_grid` and `plot_selectivity_ranking` to `src/visualization.py`
   — self-contained, testable without running inference
2. Write `analyze_unit_tuning.py`: `build_layer_tensor_map` → `make_accumulators`
   → `reduce_activation` → `accumulate_batch` → `compute_tuning_maps`
   → `compute_selectivity` → `main`
3. Pilot: `--tfrecord data/train0.tfrecords --max_samples 50` — verify shapes
4. Full run: `--tfrecords_dir data/ --max_samples 5000`

#### Key implementation notes

- Use a **single `sess.run` call** with all target tensors listed — TF computes
  the graph once and returns all requested outputs simultaneously.
- Accumulate in `float64` to avoid drift when summing thousands of `float32`
  activations; cast back to `float32` when writing tuning maps.
- The `generate_inputs` / `INIT_TOTAL` sentinel pattern from
  `test_inference_minimal.py` lines 79-84 must be replicated exactly.
- Ground truth class decoding: `az_bin = class_num % 72`, `el_bin = class_num // 72`,
  `class_idx = class_num` (flat index into 504-class space).
- Warn and skip samples where `metadata` has no `'azimuth'` key (wav inputs
  without ground truth — not useful for tuning analysis).
- Save `tuning_maps.npz` with `np.savez_compressed` before plotting so a crash
  during plotting doesn't lose the accumulation results.

---

### Top-k activating sample retrieval (extension to `analyze_unit_tuning.py`)

For each unit in each target layer, record which input samples produced the
highest activations — so the HTML report can show real cochleagram examples
alongside the tuning map, analogous to dissect's image mosaics.

Depends on: **Unit tuning analysis** task above (shares the same forward-pass
loop; top-k tracking is added to the same accumulation step).

#### What changes

| File | Action | What changes |
|---|---|---|
| `analyze_unit_tuning.py` | **Modify** | Add `RunningTopK` accumulators; save top-k indices to `topk.npz`; add second pass to render cochleagram thumbnails |
| `src/visualization.py` | **Modify** | Add `plot_topk_mosaic` function |

#### `RunningTopK` accumulator

Per layer, per unit: maintain a fixed-size buffer of the `k` highest-seen
activation values and their sample indices. When the buffer fills, compress by
keeping only the true top-k.

```python
class RunningTopK:
    """Streaming top-k tracker for one unit across many samples.

    Attributes:
        k:       number of top samples to retain
        values:  np.ndarray (k,) float32 — highest activations seen
        indices: np.ndarray (k,) int64   — corresponding sample indices
    """
    def update(self, value: float, sample_idx: int): ...
    def top_indices(self) -> np.ndarray: ...  # returns indices sorted desc
```

Storage: `k × N_units × N_layers` entries. At `k=8`, all layers combined:
`8 × (32+32+64+128+256+512) × (4+8) bytes ≈ 200 KB`. Negligible.

#### Second pass — render cochleagram thumbnails

After the main accumulation loop, re-load the top-k samples for each unit by
index. Since `load_tfrecord_samples_generator` is a forward-only iterator, store
sample cochleagrams in a flat `sample_cache` dict `{sample_idx: cochleagram}`
during the first pass (only the indices that appear in any unit's top-k — far
fewer than all samples).

Render each cochleagram channel (left ear, right ear) as a 39×8000 image,
downsampled to thumbnail size for display. Overlay the unit's spatial activation
map (upsampled to match) as a semi-transparent heatmap — bright where the unit
fires strongly, dim elsewhere.

#### New function in `src/visualization.py`

```python
def plot_topk_mosaic(
    cochleagrams,      # list of np.ndarray (39, 8000, 2), length k
    activations,       # list of float — activation value for each sample
    unit_idx,          # int
    layer_name,        # str
    output_path,       # str
    activation_maps=None,  # optional list of np.ndarray (H, W) — spatial maps
                           # to overlay; None for FC/flatten layers
):
    """k-panel figure: each panel shows a cochleagram thumbnail (left ear)
    with the unit's activation map overlaid as a semi-transparent heatmap.
    Panel title: sample rank + activation value.
    """
```

#### CLI additions to `analyze_unit_tuning.py`

| Argument | Default | Notes |
|---|---|---|
| `--top_k` | 8 | samples to retain per unit per layer |
| `--save_topk_mosaics` | false | render and save cochleagram mosaic PNGs |

#### Output additions

```
tuning_analysis/
├── topk.npz                     ← {layer_name: array (N_units, k) of sample indices}
└── mosaics/                     ← only with --save_topk_mosaics
    ├── pool_3/
    │   ├── unit_042_topk.png    ← k-panel cochleagram mosaic
    │   └── ...
    └── relu_fc/
        └── ...
```

#### Implementation order

1. Add `RunningTopK` class to `analyze_unit_tuning.py`
2. Integrate into `make_accumulators` / `accumulate_batch`
3. Add sample caching logic (store cochleagrams by index during first pass)
4. Add `plot_topk_mosaic` to `src/visualization.py`
5. Wire `--save_topk_mosaics` into `main`
6. Save `topk.npz` alongside `tuning_maps.npz`

---

### HTML tuning report (`generate_tuning_report.py`)

Generate a self-contained `index.html` from the output of the tuning analysis
(and optionally the top-k mosaics). No server required — all assets referenced
by relative path. Works as a standalone folder you can zip and share.

Depends on: **Unit tuning analysis** (required). **Top-k retrieval** (optional —
report degrades gracefully when `topk.npz` / mosaic PNGs are absent).

#### Files to create / modify

| File | Action | What changes |
|---|---|---|
| `generate_tuning_report.py` | **Create** | Root-level script; reads output dir, writes `index.html` |
| `src/visualization.py` | **Modify** | Add `generate_tuning_report` function (called by the script) |

#### Report layout

```
┌─────────────────────────────────────────────────────────────┐
│  Unit Tuning Report                                         │
│  [pool_0] [pool_1] [pool_2] [pool_3] [relu_fc]  ← layer tabs│
│  Sort by: [selectivity ▼]  Filter: [________]              │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ U042     │ U017     │ U103     │ U088     │  ...           │
│ sel=0.84 │ sel=0.71 │ sel=0.63 │ sel=0.51 │                │
│ [7×72    │ [7×72    │ [7×72    │ [7×72    │                │
│  heatmap]│  heatmap]│  heatmap]│  heatmap]│                │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
  ↕ click card to expand:
  ┌──────────────────────────────────────────────────────────┐
  │ Unit 042 — pool_3 — selectivity 0.84                     │
  │ [full tuning map]  [az marginal]  [el marginal]          │
  │ Top-8 activating samples:  (if topk.npz present)        │
  │ [cochleagram 1] [cochleagram 2] ... [cochleagram 8]      │
  └──────────────────────────────────────────────────────────┘
```

#### Implementation

Single Python function, ~150 lines. HTML/CSS/JS written as a template string —
no Jinja2 or other templating library needed.

```python
def generate_tuning_report(
    output_dir,          # str — directory containing tuning_maps.npz and PNGs
    title=None,          # str — report title (defaults to output_dir basename)
):
    """Read output_dir, write output_dir/index.html.

    Reads:
      tuning_maps.npz          → selectivity scores, layer names, unit counts
      tuning_grid_*.png        → embedded as card thumbnails (relative src)
      units/{layer}/unit_*.png → linked from expanded card view
      mosaics/{layer}/unit_*_topk.png → shown in expanded card if present

    Writes:
      index.html — self-contained; all JS inline; PNGs referenced by
                   relative path (do not move files after generating)
    """
```

JS features (vanilla, no frameworks):
- Layer tab switching (hide/show card groups by `data-layer` attribute)
- Sort by selectivity descending / ascending (toggle)
- Text filter on unit index (type a number to jump to that unit)
- Click-to-expand card (toggle `display: none` on detail panel)
- No external CDN dependencies

#### CLI

```bash
# After running analyze_unit_tuning.py:
python generate_tuning_report.py --output_dir tuning_analysis/

# With custom title:
python generate_tuning_report.py \
    --output_dir tuning_analysis/exp02/ \
    --title "Exp02 — finetuned, frozen BN, epoch 6"

# Then open in browser:
#   tuning_analysis/index.html
```

#### Output

```
tuning_analysis/
└── index.html     ← open this in any browser; no server needed
```

#### Implementation order

1. Add `generate_tuning_report` to `src/visualization.py`
2. Write `generate_tuning_report.py` root script (thin CLI wrapper)
3. Test with tuning analysis output that has no `topk.npz` — verify graceful fallback
4. Test with `topk.npz` and mosaics present — verify expanded card shows cochleagrams

---

## Completed

- Refactored `finetune_custom.py` and `test_inference_minimal.py` into `src/`
  package with modules: `data_io`, `graph_builder`, `inference`, `conversion`,
  `visualization`, `metrics`
- Implemented `plot_activation_maps` in `src/visualization.py` (single-sample,
  raw tensor visualization — per-channel freq×time heatmaps for conv layers,
  bar charts for FC layers)
- Added `--activation_output` flag to `test_inference_minimal.py`
