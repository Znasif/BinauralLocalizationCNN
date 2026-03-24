import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tensorflow as tf

from src.graph_builder import load_config, build_inference_graph
from src.inference import resolve_checkpoint


# ─────────────────────────────────────────────────────────
#  Converter discovery
# ─────────────────────────────────────────────────────────

def _get_converter_class():
    """Search standard and internal TF locations for TFLiteConverter."""
    def _from_module(module):
        for name in ('TFLiteConverter', 'TocoConverter'):
            cls = getattr(module, name, None)
            if cls is not None:
                if name == 'TocoConverter':
                    print(f'Using TocoConverter from {module.__name__}')
                return cls
        return None

    # Try the most stable paths first
    for module in [
        getattr(tf.compat.v1, 'lite', None),
        getattr(tf, 'lite', None),
        getattr(getattr(tf, 'contrib', None), 'lite', None),
    ]:
        if module is None:
            continue
        cls = _from_module(module)
        if cls:
            print(f'Found TFLiteConverter in {module.__name__}')
            return cls

    # Internal fallback
    for path in ('tensorflow.lite.python.lite',
                 'tensorflow.contrib.lite.python.lite',
                 'tensorflow.python.lite'):
        try:
            mod = __import__(path, fromlist=['TFLiteConverter'])
            cls = getattr(mod, 'TFLiteConverter', None)
            if cls:
                print(f'Found TFLiteConverter in {path}')
                return cls
        except ImportError:
            pass

    return None


# ─────────────────────────────────────────────────────────
#  Frozen .pb fallback
# ─────────────────────────────────────────────────────────

def freeze_graph_to_pb(sess, output_file, output_node_names, input_shape):
    """Freeze all variables to constants and save a .pb file.

    Args:
        sess:              active tf.compat.v1.Session
        output_file:       path for the .pb (or path with .tflite swapped out)
        output_node_names: list of output node name strings
        input_shape:       model input shape, logged for user reference
    """
    try:
        from tensorflow.compat.v1 import graph_util
    except ImportError:
        from tensorflow.python.framework import graph_util

    pb_path = output_file.replace('.tflite', '.pb')
    if pb_path == output_file:
        pb_path += '.pb'

    print(f'Freezing graph to {pb_path}...')
    frozen = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_node_names)
    with tf.io.gfile.GFile(pb_path, 'wb') as f:
        f.write(frozen.SerializeToString())

    print(f'\nSUCCESS: Frozen graph saved to: {pb_path}')
    print(f'Output nodes: {output_node_names}')
    print(f'Input node: "input"   shape: {input_shape}')
    print('Convert with: tensorflowjs_converter or tflite_convert in another env.')


# ─────────────────────────────────────────────────────────
#  Main conversion entry point
# ─────────────────────────────────────────────────────────

def convert_to_tflite(model_dir, output_file, config_path=None):
    """Convert a checkpoint in *model_dir* to a .tflite file.

    If no TFLite converter is available the graph is frozen to a .pb file
    instead, which can be converted in a separate environment.

    Args:
        model_dir:   directory containing checkpoint and config_array.npy
        output_file: destination .tflite path
        config_path: explicit path to config_array.npy; defaults to model_dir
    """
    print(f'Converting model in {model_dir} to TFLite...')

    config_array = load_config(model_dir, config_path=config_path, target_device='/cpu:0')
    input_shape  = [1, 39, 8000, 2]

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        print('Building graph...')
        input_placeholder, _, probabilities, _ = build_inference_graph(
            config_array, batch_size=1, n_classes=504)

        print('Restoring checkpoint...')
        ckpt_path = resolve_checkpoint(model_dir)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_path)
        print(f'Loaded checkpoint: {ckpt_path}')

        print('Converting to TFLite...')
        converter_class = _get_converter_class()

        if converter_class is None:
            print('\n' + '!' * 60)
            print('COULD NOT FIND TFLITE CONVERTER')
            print('Falling back to frozen .pb output.')
            print('!' * 60 + '\n')
            freeze_graph_to_pb(sess, output_file,
                                ['output_probabilities'], input_shape)
            return

        converter = converter_class.from_session(
            sess, [input_placeholder], [probabilities])

        try:
            if hasattr(tf, 'lite') and hasattr(tf.lite, 'OpsSet'):
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
            else:
                print('tf.lite.OpsSet not available — using default TFLITE_BUILTINS.')
        except Exception as e:
            print(f'Warning: Could not configure supported_ops: {e}')

        converter.allow_custom_ops = True
        tflite_model = converter.convert()

        with open(output_file, 'wb') as f:
            f.write(tflite_model)
        print(f'TFLite model saved to: {output_file}')
