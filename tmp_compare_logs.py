"""Quick script to dump TensorBoard scalars from two log dirs for comparison."""
import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. pip install tensorboard")
    sys.exit(1)

BASE = "/mnt/d/Projects/BinauralLocalizationCNN/experiments/exp02_alllayers_lr5e-5_frozenbn"

size_guidance = {
    'compressedHistograms': 0,
    'images': 0,
    'audio': 0,
    'scalars': 10000,
    'histograms': 0,
}

output_lines = []

for log_name in ["logs", "logs1"]:
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"  {log_name}")
    output_lines.append(f"{'='*70}")
    for subdir in ["train", "val", "original"]:
        path = os.path.join(BASE, log_name, subdir)
        if not os.path.isdir(path):
            continue
        ea = EventAccumulator(path, size_guidance=size_guidance)
        ea.Reload()
        tags = ea.Tags()
        scalar_tags = tags.get('scalars', [])
        if not scalar_tags:
            output_lines.append(f"\n  --- {subdir} --- (no scalar tags)")
            continue
        output_lines.append(f"\n  --- {subdir} ---")
        for tag in sorted(scalar_tags):
            events = ea.Scalars(tag)
            output_lines.append(f"  {tag}:")
            for e in events:
                output_lines.append(f"    step={e.step:3d}  value={e.value:.6f}")

out_path = "/mnt/d/Projects/BinauralLocalizationCNN/tmp_compare_output.txt"
with open(out_path, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"Output written to {out_path}")
