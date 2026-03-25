import os
import sys
import glob
import numpy as np
import scipy.signal as signallib
from scipy.io import wavfile

# Ensure project root is in sys.path so pycochleagram is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pycochleagram import erbfilter as erb
    from pycochleagram import cochleagram as cgram
    _cochleagram_available = True
except ImportError:
    _cochleagram_available = False

try:
    import tensorflow as tf
    _tf_available = True
except ImportError:
    _tf_available = False


# ─────────────────────────────────────────────────────────
#  WAV / audio loading
# ─────────────────────────────────────────────────────────

def load_wav_file(wav_path):
    """Load a stereo wav file and return (audio_float32, sample_rate).

    Mono files are duplicated to stereo.  Output dtype is always float32
    with amplitude in [-1, 1].
    """
    sr, audio = wavfile.read(wav_path)

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.ndim == 2 and audio.shape[1] > 2:
        audio = audio.T

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, sr


def wav_to_cochleagram(audio, sr, target_sr=48000, n_channels=39,
                       low_lim=30, hi_lim=20000):
    """Convert stereo audio to cochleagram format (39, n_samples, 2).

    Args:
        audio:     (n_samples, 2) float32 array
        sr:        sample rate of *audio*
        target_sr: resample to this rate before filtering (model expects 48 kHz)
        n_channels: number of ERB frequency bins to keep
        low_lim, hi_lim: filterbank frequency bounds (Hz)

    Returns:
        cochleagram: (n_channels, n_samples, 2) float32
    """
    if not _cochleagram_available:
        raise ImportError('pycochleagram is required for wav_to_cochleagram')

    if sr != target_sr:
        from scipy import signal
        n_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, n_samples)
        sr = target_sr

    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
    sample_factor = 1

    cochleagrams = []
    for channel in range(2):
        signal_data = audio[:, channel]
        coch = cgram.cochleagram(
            signal_data, sr, n, low_lim, hi_lim, sample_factor,
            padding_size=None, downsample=None, nonlinearity=None,
            fft_mode='np', ret_mode='subband', strict=False)
        if coch.shape[0] > n_channels:
            start_idx = (coch.shape[0] - n_channels) // 2
            coch = coch[start_idx:start_idx + n_channels]
        cochleagrams.append(coch)

    return np.stack(cochleagrams, axis=2).astype(np.float32)


# ─────────────────────────────────────────────────────────
#  Cochleagram downsampling
# ─────────────────────────────────────────────────────────

def make_downsample_filter(sr=48000, env_sr=8000, window_size=4097, beta=10.06):
    """Build a sinc-Kaiser lowpass filter for cochleagram downsampling."""
    downsample_ratio = sr // env_sr
    t = np.arange(-window_size / 2, int(window_size / 2))
    h = np.sinc(t / downsample_ratio) / downsample_ratio

    if hasattr(signallib, 'windows') and hasattr(signallib.windows, 'kaiser'):
        kaiser_func = signallib.windows.kaiser
    elif hasattr(signallib, 'kaiser'):
        kaiser_func = signallib.kaiser
    else:
        from scipy.signal.windows import kaiser as kaiser_func

    window = kaiser_func(window_size, beta)
    return (window * h).astype(np.float32), downsample_ratio


def downsample_cochleagram(cochleagram, sr=48000, target_sr=8000):
    """Downsample cochleagram from *sr* to *target_sr*.

    Args:
        cochleagram: (n_channels, n_samples, 2)
    Returns:
        downsampled: (n_channels, n_samples // ratio, 2), ReLU applied
    """
    filt, ratio = make_downsample_filter(sr, target_sr)
    n_channels, n_samples, n_ears = cochleagram.shape
    n_out = n_samples // ratio
    out   = np.zeros((n_channels, n_out, n_ears), dtype=np.float32)

    for ear in range(n_ears):
        for ch in range(n_channels):
            filtered = np.convolve(cochleagram[ch, :, ear], filt, mode='same')
            out[ch, :, ear] = filtered[::ratio][:n_out]

    return np.maximum(out, 0)


def prepare_input_for_model(cochleagram, target_samples=8000):
    """Pad/trim cochleagram to *target_samples* and add batch dim.

    Args:
        cochleagram: (39, n_samples, 2)
    Returns:
        model_input: (1, 39, 8000, 2) float32
    """
    n_channels, n_samples, n_ears = cochleagram.shape

    if n_samples > target_samples * 2:
        #print(f'Downsampling cochleagram from {n_samples} to ~{target_samples} samples...')
        cochleagram = downsample_cochleagram(cochleagram, sr=48000, target_sr=8000)
        n_samples = cochleagram.shape[1]

    if n_samples < target_samples:
        cochleagram = np.pad(cochleagram,
                             ((0, 0), (0, target_samples - n_samples), (0, 0)))
    elif n_samples > target_samples:
        cochleagram = cochleagram[:, :target_samples, :]

    # Guard for x^0.3 nonlinearity in the graph
    cochleagram = np.maximum(cochleagram, 0)

    return cochleagram[np.newaxis, ...].astype(np.float32)


# ─────────────────────────────────────────────────────────
#  TFRecord readers  (require TensorFlow)
# ─────────────────────────────────────────────────────────

def _require_tf():
    if not _tf_available:
        raise ImportError('TensorFlow is required for TFRecord I/O')


def parse_tfrecord_fn(example_proto):
    """TF-graph-level parser for the binaural tfrecord format."""
    _require_tf()
    feature_description = {
        'train/azim':         tf.io.FixedLenFeature([], tf.int64),
        'train/elev':         tf.io.FixedLenFeature([], tf.int64),
        'train/class_num':    tf.io.FixedLenFeature([], tf.int64),
        'train/image':        tf.io.FixedLenFeature([], tf.string),
        'train/image_height': tf.io.FixedLenFeature([], tf.int64),
        'train/image_width':  tf.io.FixedLenFeature([], tf.int64),
        'train/click_type':   tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image  = tf.io.decode_raw(parsed['train/image'], tf.float32)
    height = tf.cast(parsed['train/image_height'], tf.int32)
    width  = tf.cast(parsed['train/image_width'],  tf.int32)
    image  = tf.reshape(image, (height, width, 2))
    return image, parsed['train/class_num'], parsed['train/click_type']


def make_pipeline(shard_paths, batch_size, shuffle=True):
    """Build a tf.data pipeline from a list of TFRecord shard paths.

    Returns (iterator, next_element) using TF1 initializable iterator.
    """
    _require_tf()
    file_ds = tf.data.Dataset.from_tensor_slices(shard_paths)
    if shuffle:
        file_ds = file_ds.shuffle(buffer_size=len(shard_paths))

    record_ds = file_ds.interleave(
        lambda path: tf.data.TFRecordDataset(path, compression_type='GZIP'),
        cycle_length=min(10, len(shard_paths)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    record_ds = record_ds.map(parse_tfrecord_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        record_ds = record_ds.shuffle(buffer_size=2000)
    record_ds = record_ds.batch(batch_size, drop_remainder=True)
    record_ds = record_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator     = tf.compat.v1.data.make_initializable_iterator(record_ds)
    next_element = iterator.get_next()
    return iterator, next_element


def count_records(shard_paths):
    """Count total records across a list of TFRecord shards."""
    _require_tf()
    n    = 0
    opts = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
    for path in shard_paths:
        for _ in tf.compat.v1.python_io.tf_record_iterator(path, options=opts):
            n += 1
    return n


def split_shards(all_shards, val_fraction=0.2):
    """Deterministic train/val split balanced across 0-click and 1-click shards."""
    shards_0 = sorted([s for s in all_shards if '0click' in os.path.basename(s)])
    shards_1 = sorted([s for s in all_shards if '1click' in os.path.basename(s)])

    def holdout(lst, frac):
        n_val = max(1, int(len(lst) * frac))
        return lst[:-n_val], lst[-n_val:]

    train_0, val_0 = holdout(shards_0, val_fraction)
    train_1, val_1 = holdout(shards_1, val_fraction)
    return train_0 + train_1, val_0 + val_1


def load_tfrecord_samples_generator(tfrecord_path, max_samples=None):
    """Yield (cochleagram, metadata) from a tfrecord file.

    cochleagram shape: (39, n_samples, 2)
    metadata keys: 'azimuth', 'elevation', optionally 'condition'
    """
    _require_tf()
    options = tf.io.TFRecordOptions(
        tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    count = 0
    for serialized in tf.compat.v1.python_io.tf_record_iterator(
            tfrecord_path, options=options):
        if max_samples and count >= max_samples:
            break
        example = tf.train.Example()
        example.ParseFromString(serialized)
        features = example.features.feature

        if 'train/image' not in features:
            continue
        raw = features['train/image'].bytes_list.value[0]
        try:
            data = np.frombuffer(raw, dtype=np.float32)
        except Exception:
            data = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
        cochleagram = data.reshape(39, len(data) // (39 * 2), 2)

        metadata = {}
        if 'train/class_num' in features:
            cls = features['train/class_num'].int64_list.value[0]
            metadata['azimuth']   = (cls % 72) * 5
            metadata['elevation'] = (cls // 72) * 10
        else:
            # Original Francl data: train/azim and train/elev store raw degrees
            # (0-355 and 0-60), not bin indices — no multiplication needed.
            if 'train/azim' in features:
                metadata['azimuth']   = features['train/azim'].int64_list.value[0]
            if 'train/elev' in features:
                metadata['elevation'] = features['train/elev'].int64_list.value[0]
        if 'train/click_type' in features:
            click_val = features['train/click_type'].int64_list.value[0]
            metadata['condition'] = '1click' if click_val == 1 else '0click'

        yield cochleagram, metadata
        count += 1


def load_tfrecord_sample(tfrecord_path, sample_index=0):
    """Load a single cochleagram from a tfrecord by index.

    Returns (cochleagram, metadata).
    """
    _require_tf()
    options = tf.io.TFRecordOptions(
        tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    for count, serialized in enumerate(
            tf.compat.v1.python_io.tf_record_iterator(tfrecord_path, options=options)):
        if count != sample_index:
            continue
        example = tf.train.Example()
        example.ParseFromString(serialized)
        features = example.features.feature

        if 'train/image' not in features:
            raise ValueError("No 'train/image' found in tfrecord")
        raw = features['train/image'].bytes_list.value[0]
        try:
            data = np.frombuffer(raw, dtype=np.float32)
        except Exception:
            data = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
        cochleagram = data.reshape(39, len(data) // (39 * 2), 2)

        metadata = {}
        if 'train/class_num' in features:
            cls = features['train/class_num'].int64_list.value[0]
            metadata['azimuth']   = (cls % 72) * 5
            metadata['elevation'] = (cls // 72) * 10
        else:
            # Original Francl data: train/azim and train/elev store raw degrees
            # (0-355 and 0-60), not bin indices — no multiplication needed.
            if 'train/azim' in features:
                metadata['azimuth']   = features['train/azim'].int64_list.value[0]
            if 'train/elev' in features:
                metadata['elevation'] = features['train/elev'].int64_list.value[0]
        if 'train/click_type' in features:
            click_val = features['train/click_type'].int64_list.value[0]
            metadata['condition'] = '1click' if click_val == 1 else '0click'

        return cochleagram, metadata

    raise ValueError(f'Sample index {sample_index} out of range in {tfrecord_path}')


# ─────────────────────────────────────────────────────────
#  Inference input generator (used by test_inference_minimal)
# ─────────────────────────────────────────────────────────

def generate_inputs(args, model_dir):
    """Generator yielding (cochleagram, metadata, png_filename) for each input.

    Dispatch order: --wav_file → --tfrecords_dir → --tfrecord → --wav_folder → dummy.
    """
    # Single wav file
    if args.wav_file:
        print(f'Loading wav file: {args.wav_file}')
        png_file = args.wav_file.split('/')[-1].split('.')[0] + '.png'
        audio, sr = load_wav_file(args.wav_file)
        print(f'Audio shape: {audio.shape}, Sample rate: {sr}')
        print('Converting wav to cochleagram using pycochleagram...')
        coch = wav_to_cochleagram(audio, sr)
        print(f'Cochleagram shape: {coch.shape}')
        yield coch, {'azimuth': -1000, 'elevation': 0}, png_file
        return

    # TFRecord directory (all shards, up to 200 samples per shard)
    if args.tfrecords_dir:
        shards = glob.glob(os.path.join(args.tfrecords_dir, '*.tfrecords'))
        if not shards:
            if not args.quiet:
                print(f'Warning: No .tfrecords found in {args.tfrecords_dir}')
            return
        total_samples = len(shards) * 200
        if not args.quiet:
            print(f'Found {len(shards)} shards, planning up to {total_samples} samples.')
        yield 'INIT_TOTAL', total_samples, ''
        for shard_path in shards:
            if not args.quiet:
                print(f'Loading from tfrecord shard: {shard_path} (limit 200 samples)')
            for coch, metadata in load_tfrecord_samples_generator(shard_path, max_samples=200):
                yield coch, metadata, ''
        return

    # Single TFRecord file
    if args.tfrecord:
        if args.sample_index:
            print(f'Loading from tfrecord: {args.tfrecord}, sample {args.sample_index}')
            coch, metadata = load_tfrecord_sample(args.tfrecord, args.sample_index)
            yield coch, metadata, ''
            return
        print(f'Loading from tfrecord: {args.tfrecord} (first 200 samples)')
        for coch, metadata in load_tfrecord_samples_generator(args.tfrecord, max_samples=200):
            yield coch, metadata, ''
        return

    # WAV folder (recursive)
    if args.wav_folder:
        print(f'Loading wav files recursively from folder: {args.wav_folder}')
        wav_files = glob.glob(os.path.join(args.wav_folder, '**', '*.wav'), recursive=True)
        if not wav_files:
            print(f'Warning: No .wav files found in {args.wav_folder}')
            return
        for file_path in wav_files:
            filename = os.path.basename(file_path)
            print(f'Loading wav file: {file_path}')
            audio, sr = load_wav_file(file_path)
            coch = wav_to_cochleagram(audio, sr)
            az, ev = -1000, 0
            try:
                if 'az' in filename:
                    az = int(filename.split('az')[1].split('_')[0])
                    if az < 0:
                        az += 360
                if 'el' in filename:
                    ev = int(filename.split('el')[1].split('_')[0])
                    if ev < 0:
                        continue
            except (IndexError, ValueError):
                pass
            condition = ('0click' if '0click' in filename
                         else '1click' if '1click' in filename
                         else 'unknown')
            rel_path  = os.path.relpath(file_path, args.wav_folder)
            safe_name = rel_path.replace(os.sep, '_').replace('/', '_')
            png_file  = os.path.splitext(safe_name)[0] + '.png'
            yield coch, {'azimuth': az, 'elevation': ev, 'condition': condition}, png_file
        return

    # Fallback: dummy random input
    print('No input specified, using random dummy data for testing...')
    coch = np.abs(np.random.randn(39, 48000, 2).astype(np.float32))
    yield coch, {'source': 'dummy'}, ''
