import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

try:
    import tflite_runtime.interpreter as _tflite_rt
    _tflite = _tflite_rt
except ImportError:
    try:
        import tensorflow.lite as _tflite
    except ImportError:
        _tflite = None


# ─────────────────────────────────────────────────────────
#  TF1 / checkpoint inference
# ─────────────────────────────────────────────────────────

def make_session_config(use_gpu=False):
    """Return a tf.compat.v1.ConfigProto for session creation.

    Args:
        use_gpu: if True, allow GPU with memory growth;
                 if False, disable GPU entirely (CPU-only).
    """
    if use_gpu:
        cfg = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        cfg.gpu_options.allow_growth = True
    else:
        cfg = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=True)
    return cfg


def resolve_checkpoint(model_dir, preferred='model.ckpt-100000'):
    """Find the best available checkpoint path in *model_dir*.

    Tries *preferred* first, then falls back to whatever
    tf.train.get_checkpoint_state reports.

    Returns the checkpoint path string.
    Raises FileNotFoundError if nothing is found.
    """
    preferred_path = os.path.join(model_dir, preferred)
    if os.path.exists(preferred_path + '.index'):
        return preferred_path

    state = tf.train.get_checkpoint_state(model_dir)
    if state and state.model_checkpoint_path:
        return state.model_checkpoint_path

    raise FileNotFoundError(f'No checkpoint found in {model_dir}')


def restore_checkpoint(sess, model_dir, checkpoint_path=None, max_to_keep=None):
    """Create a Saver, resolve the checkpoint path, and restore weights.

    Args:
        sess:            active tf.compat.v1.Session
        model_dir:       directory used for fallback checkpoint lookup
        checkpoint_path: explicit path; if None, resolve_checkpoint is called
        max_to_keep:     passed to tf.compat.v1.train.Saver (None = TF default)

    Returns:
        (saver, resolved_checkpoint_path)
    """
    saver_kwargs = {}
    if max_to_keep is not None:
        saver_kwargs['max_to_keep'] = max_to_keep
    saver = tf.compat.v1.train.Saver(**saver_kwargs)

    ckpt_path = checkpoint_path or resolve_checkpoint(model_dir)
    print(f'Restoring checkpoint: {ckpt_path}')
    saver.restore(sess, ckpt_path)
    return saver, ckpt_path


def run_ckpt_inference(sess, input_ph, predictions, probabilities, model_input):
    """Run a single forward pass through the TF1 graph.

    Args:
        sess:          active tf.compat.v1.Session
        input_ph:      input placeholder tensor
        predictions:   argmax tensor
        probabilities: softmax tensor
        model_input:   numpy array of shape (1, 39, 8000, 2)

    Returns:
        (pred_array, prob_array) — both numpy arrays
    """
    return sess.run([predictions, probabilities],
                    feed_dict={input_ph: model_input})


# ─────────────────────────────────────────────────────────
#  TFLite inference
# ─────────────────────────────────────────────────────────

def load_tflite_interpreter(model_path):
    """Load a .tflite file and allocate tensors.

    Returns an Interpreter instance ready for inference.
    """
    if _tflite is None:
        raise ImportError(
            'Neither tflite_runtime nor tensorflow.lite could be imported. '
            'Install tflite-runtime or tensorflow.')
    interpreter = _tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_tflite_inference_on_input(interpreter, input_data):
    """Set input tensor, invoke, and return the output probabilities array.

    Args:
        interpreter: allocated tflite.Interpreter
        input_data:  numpy array of shape (1, 39, 8000, 2) float32

    Returns:
        probabilities: 1-D numpy array of length 504
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]
