import os
import sys
import numpy as np

# Ensure project root is in sys.path so NetBuilder_valid_pad is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf


def remap_devices(arr, target_device='/cpu:0'):
    """Recursively remap GPU device strings in a config_array to *target_device*.

    Args:
        arr:           config_array (numpy array, list, or scalar)
        target_device: e.g. '/cpu:0' or '/gpu:0'
    Returns:
        A copy of *arr* with all '/gpu:…' strings replaced.
    """
    if isinstance(arr, np.ndarray):
        return np.array([remap_devices(x, target_device) for x in arr], dtype=object)
    if isinstance(arr, list):
        return [remap_devices(x, target_device) for x in arr]
    if isinstance(arr, str) and '/gpu' in arr.lower():
        return target_device
    return arr


def load_config(model_dir, config_path=None, target_device='/cpu:0'):
    """Load config_array.npy and remap device strings.

    Args:
        model_dir:     directory containing config_array.npy
        config_path:   explicit path; overrides model_dir lookup when given
        target_device: passed to remap_devices

    Returns:
        config_array (numpy object array)
    """
    path = config_path or os.path.join(model_dir, 'config_array.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Config file not found: {path}')
    config_array = np.load(path, allow_pickle=True)
    return remap_devices(config_array, target_device)


def build_inference_graph(config_array, batch_size=1, n_classes=504):
    """Build the TF1 inference graph.

    Constructs:
        input_placeholder → x^0.3 nonlinearity → NetBuilder → softmax output

    Args:
        config_array: remapped config array from load_config
        batch_size:   fixed batch size for the placeholder (use 1 for single-sample)
        n_classes:    number of output classes (504 for binaural localization)

    Returns:
        (input_placeholder, logits, probabilities, predictions)
    """
    from NetBuilder_valid_pad import NetBuilder

    input_placeholder = tf.compat.v1.placeholder(
        tf.float32, [batch_size, 39, 8000, 2], name='input')
    nonlin = tf.pow(input_placeholder, 0.3)

    net  = NetBuilder()
    out  = net.build(config_array, nonlin,
                     training_state=False,
                     dropout_training_state=False,
                     filter_dtype=tf.float32,
                     padding='VALID',
                     n_classes_localization=n_classes,
                     n_classes_recognition=780,
                     branched=False,
                     regularizer=None)

    probabilities = tf.nn.softmax(out, name='output_probabilities')
    predictions   = tf.argmax(out, axis=1)
    return input_placeholder, out, probabilities, predictions


def build_training_graph(config_array, batch_size, n_classes=504, bn_momentum=0.99):
    """Build the TF1 training graph (forward pass + cross-entropy loss only).

    Optimizer, update_ops filtering, and TensorBoard summaries are intentionally
    left to the caller so training-specific logic (freeze_conv, weight_decay, etc.)
    stays in finetune_custom.py.

    Args:
        config_array: remapped config array from load_config
        batch_size:   training mini-batch size
        n_classes:    number of output classes
        bn_momentum:  batch normalisation momentum for moving averages

    Returns:
        (input_ph, labels_ph, is_training_ph, logits, cost, predictions)
    """
    from NetBuilder_valid_pad import NetBuilder

    input_ph     = tf.compat.v1.placeholder(
        tf.float32, [batch_size, 39, 8000, 2], name='input')
    labels_ph    = tf.compat.v1.placeholder(tf.int64, [batch_size], name='labels')
    is_training  = tf.compat.v1.placeholder(tf.bool, shape=(), name='is_training')

    nonlin = tf.pow(input_ph, 0.3)
    net    = NetBuilder()
    out    = net.build(config_array, nonlin,
                       training_state=is_training,
                       dropout_training_state=is_training,
                       filter_dtype=tf.float32,
                       padding='VALID',
                       n_classes_localization=n_classes,
                       n_classes_recognition=780,
                       branched=False,
                       regularizer=None,
                       bn_momentum=bn_momentum)

    cost     = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=labels_ph))
    preds_op = tf.argmax(out, 1)
    return input_ph, labels_ph, is_training, out, cost, preds_op
