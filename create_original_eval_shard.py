"""
create_original_eval_shard.py
─────────────────────────────
One-time script. Reads N records from the original 48kHz TFRecords, downsamples
to 8kHz using the same filter as the model's preprocessing, and writes a new
GZIP TFRecord file in the same format as the finetuning shards so
finetune_custom.py can use it directly as a third validation set.

Run once (inside Docker or rapids env):
    python create_original_eval_shard.py \\
        --data_dir data \\
        --output data/data_original_eval.tfrecords \\
        --n_records 200 \\
        --max_shards 1
"""

import os
import glob
import random
import argparse
import numpy as np
import scipy.signal as signallib
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# ─────────────────────────────────────────────────────────
#  Downsampler (48kHz → 8kHz, same filter as model input)
# ─────────────────────────────────────────────────────────

def make_downsample_filter(sr=48000, env_sr=8000, window_size=4097, beta=10.06):
    ratio = sr // env_sr
    times = np.arange(-window_size / 2, window_size / 2)
    h     = np.sinc(times / ratio) / ratio
    try:
        w = signallib.windows.kaiser(window_size, beta)
    except AttributeError:
        w = signallib.kaiser(window_size, beta)
    return (h * w).astype(np.float32), ratio


def downsample_coch(coch48k, filt, ratio, target=8000):
    """[39, T, 2] → [39, 8000, 2]"""
    out = np.zeros((39, target, 2), dtype=np.float32)
    for ear in range(2):
        for ch in range(39):
            sig      = coch48k[ch, :, ear]
            filtered = np.convolve(sig, filt, mode='same')
            dec      = np.maximum(filtered[::ratio], 0)
            n        = min(len(dec), target)
            out[ch, :n, ear] = dec[:n]
    return out


# ─────────────────────────────────────────────────────────
#  Read original 48kHz records via reservoir sampling
# ─────────────────────────────────────────────────────────

def reservoir_sample(shard_paths, n):
    opts      = tf.compat.v1.python_io.TFRecordOptions(
                    tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    reservoir = []
    count     = 0
    for path in shard_paths:
        for raw in tf.compat.v1.python_io.tf_record_iterator(path, options=opts):
            if len(reservoir) < n:
                reservoir.append(raw)
            else:
                j = random.randint(0, count)
                if j < n:
                    reservoir[j] = raw
            count += 1
            if count % 500 == 0:
                print(f"  Scanned {count} records, {len(reservoir)} selected...", flush=True)
    print(f"  Done: {count} total records scanned, {len(reservoir)} selected.")
    return reservoir


# ─────────────────────────────────────────────────────────
#  Write 8kHz records in the finetuning TFRecord format
# ─────────────────────────────────────────────────────────

def write_eval_shard(raw_records, filt, ratio, output_path):
    opts = tf.compat.v1.python_io.TFRecordOptions(
               tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    n = len(raw_records)
    with tf.compat.v1.python_io.TFRecordWriter(output_path, options=opts) as writer:
        for i, raw in enumerate(raw_records):
            ex48 = tf.train.Example()
            ex48.ParseFromString(raw)
            feat = ex48.features.feature

            # Read 48kHz cochleagram
            img_bytes = feat['train/image'].bytes_list.value[0]
            height    = feat['train/image_height'].int64_list.value[0]
            width     = feat['train/image_width'].int64_list.value[0]
            img48k    = np.frombuffer(img_bytes, dtype=np.float32).reshape(height, width, 2)

            # Labels
            azim_raw  = feat['train/azim'].int64_list.value[0]
            elev_raw  = feat['train/elev'].int64_list.value[0]
            class_num = int((elev_raw // 10) * 72 + (azim_raw // 5))

            # Downsample
            img8k = downsample_coch(img48k, filt, ratio)  # [39, 8000, 2]
            h8, w8 = img8k.shape[0], img8k.shape[1]

            # Write in the finetune TFRecord format
            ex8 = tf.train.Example(features=tf.train.Features(feature={
                'train/image':        tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img8k.tobytes()])),
                'train/image_height': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[h8])),
                'train/image_width':  tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[w8])),
                'train/azim':         tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[azim_raw])),
                'train/elev':         tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[elev_raw])),
                'train/class_num':    tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[class_num])),
                'train/click_type':   tf.train.Feature(   # 0 = no precedence effect
                    int64_list=tf.train.Int64List(value=[0])),
            }))
            writer.write(ex8.SerializeToString())

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"  Written {i+1}/{n}", flush=True)

    print(f"\nSaved: {output_path}  ({n} records, 8kHz, GZIP)")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data',
                        help='Directory containing original 48kHz *.tfrecords')
    parser.add_argument('--output',     default='data/data_original_eval.tfrecords',
                        help='Output path for the downsampled eval shard')
    parser.add_argument('--n_records',  type=int, default=200,
                        help='Number of records to sample')
    parser.add_argument('--max_shards', type=int, default=1,
                        help='Number of source shards to read from (0=all)')
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"Output already exists: {args.output}")
        print("Delete it first if you want to regenerate.")
        return

    shards = glob.glob(os.path.join(args.data_dir, '*.tfrecords'))
    # Exclude any previously created eval shards
    shards = [s for s in shards if 'original_eval' not in s]
    if not shards:
        print(f"No .tfrecords found in {args.data_dir}")
        return

    random.shuffle(shards)
    if args.max_shards > 0:
        shards = shards[:args.max_shards]
    print(f"Sampling {args.n_records} records from {len(shards)} shard(s)...")

    raw = reservoir_sample(shards, args.n_records)
    filt, ratio = make_downsample_filter()

    print(f"\nDownsampling and writing to {args.output}...")
    write_eval_shard(raw, filt, ratio, args.output)


if __name__ == '__main__':
    main()
