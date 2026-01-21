#!/usr/bin/env python
import os
import argparse
import numpy as np
import tensorflow as tf

# Include current directory in path to find NetBuilder_valid_pad
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NetBuilder_valid_pad import NetBuilder

def remap_devices(arr):
    """Recursively remap devices in config_array to CPU"""
    target_device = '/cpu:0'
    
    if isinstance(arr, np.ndarray):
        return np.array([remap_devices(x) for x in arr], dtype=object)
    elif isinstance(arr, list):
        return [remap_devices(x) for x in arr]
    elif isinstance(arr, str) and '/gpu' in arr.lower():
        return target_device
    return arr

def convert_to_tflite(model_dir, output_file):
    print(f"Converting model in {model_dir} to TFLite...")
    
    # Load config
    config_path = os.path.join(model_dir, 'config_array.npy')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    config_array = np.load(config_path, allow_pickle=True)
    config_array = remap_devices(config_array)
    
    # Graph construction parameters
    n_classes_localization = 504
    batch_size = 1
    input_shape = [batch_size, 39, 8000, 2]
    
    # Reset graph to ensure clean state
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        print("Building graph...")
        # Define input placeholder
        input_placeholder = tf.compat.v1.placeholder(tf.float32, input_shape, name='input')
        
        # Apply preprocessing (non-linearity) as in inference script
        # Note: This means the TFLite model will expect the raw cochleagram input
        nonlin = tf.pow(input_placeholder, 0.3)
        
        net = NetBuilder()
        out = net.build(config_array, nonlin, 
                       training_state=False,
                       dropout_training_state=False,
                       filter_dtype=tf.float32,
                       padding='VALID',
                       n_classes_localization=n_classes_localization, # 504
                       n_classes_recognition=780, # not used for output but likely needed for build
                       branched=False,
                       regularizer=None)
        
        # Define output
        probabilities = tf.nn.softmax(out, name='output_probabilities')
        
        print("Restoring checkpoint...")
        saver = tf.compat.v1.train.Saver()
        ckpt_path = os.path.join(model_dir, 'model.ckpt-100000')
        
        if not os.path.exists(ckpt_path + ".index"):
             # Try finding ANY checkpoint if specific one doesn't exist
            print(f"Specific checkpoint {ckpt_path} not found. Searching for others...")
            chkpt = tf.train.get_checkpoint_state(model_dir)
            if chkpt and chkpt.model_checkpoint_path:
                ckpt_path = chkpt.model_checkpoint_path
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_dir}")
                
        print(f"Loading checkpoint: {ckpt_path}")
        saver.restore(sess, ckpt_path)
        
        # Convert to TFLite
        print("Converting to TFLite...")
        
        converter = None
        converter_class = None
        
        # Explicit import attempt
        try:
            import tensorflow.lite
        except ImportError:
            pass

        # Helper to find class in module
        def get_converter_class(module):
            if hasattr(module, 'TFLiteConverter'):
                return module.TFLiteConverter
            if hasattr(module, 'TocoConverter'):
                print(f"Using TocoConverter from {module.__name__}")
                return module.TocoConverter
            return None

        # Search locations
        search_modules = []
        if hasattr(tf, 'lite'): search_modules.append(tf.lite)
        
        # Check compat.v1.lite safely
        try:
            if hasattr(tf.compat.v1, 'lite'): search_modules.append(tf.compat.v1.lite)
        except AttributeError:
            pass
            
        # Check contrib.lite safely
        try:
            if hasattr(tf, 'contrib') and hasattr(tf.contrib, 'lite'): 
                search_modules.append(tf.contrib.lite)
        except AttributeError:
            pass
        
        for module in search_modules:
            converter_class = get_converter_class(module)
            if converter_class:
                print(f"Found converter class in {module.__name__}")
                break
                
        # Last resort: internal imports
        if converter_class is None:
            print("Standard accessors failed. Trying internal imports...")
            internal_paths = [
                'tensorflow.lite.python.lite',
                'tensorflow.contrib.lite.python.lite',
                'tensorflow.python.lite'
            ]
            
            for path in internal_paths:
                try:
                    module = __import__(path, fromlist=['TFLiteConverter'])
                    if hasattr(module, 'TFLiteConverter'):
                        converter_class = module.TFLiteConverter
                        print(f"Found TFLiteConverter in {path}")
                        break
                except ImportError:
                    pass

        # If we still haven't found a converter, fallback to freezing the graph
        if converter_class is None:
            print("\n" + "!"*60)
            print("COULD NOT FIND TFLITE CONVERTER")
            print("This usually means TFLite is not included in this TensorFlow build.")
            print("Falling back to saving a Frozen Graph (.pb) file.")
            print("You can convert this .pb file to TFLite in another environment (e.g., Google Colab).")
            print("!"*60 + "\n")
            
            pb_output = output_file.replace('.tflite', '.pb')
            if pb_output == output_file:
                pb_output += '.pb'
                
            print(f"Freezing graph to {pb_output}...")
            
            # Use graph_util to convert variables to constants
            try:
                from tensorflow.compat.v1 import graph_util
            except ImportError:
                from tensorflow.python.framework import graph_util
                
            # Get the graph definition
            input_graph_def = sess.graph.as_graph_def()
            
            # Convert variables to constants
            # The output node name is 'output_probabilities' as defined in tf.nn.softmax
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                ['output_probabilities']
            )
            
            # Write to file
            with tf.io.gfile.GFile(pb_output, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            
            print(f"\nSUCCESS: Frozen graph saved to: {pb_output}")
            print(f"Output Node Name: output_probabilities")
            print(f"Input Node Name: input")
            print(f"Input Shape: {input_shape}")
            return

        # If we found a converter class, proceed with conversion
        print("Instantiating converter...")
        
        # Instantiate converter
        converter = converter_class.from_session(sess, [input_placeholder], [probabilities])
        
        # Configure supported ops safely
        try:
            # Try to use OpsSet if available (TF 1.15+)
            if hasattr(tf, 'lite') and hasattr(tf.lite, 'OpsSet'):
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            else:
                print("tf.lite.OpsSet not found. Using default supported_ops (TFLITE_BUILTINS).")
        except Exception as e:
            print(f"Warning: Could not configure supported_ops: {e}")
            
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        
        # Save model
        with open(output_file, 'wb') as f:
            f.write(tflite_model)
            
        print(f"TFLite model saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert BinauralLocalizationCNN model to TFLite')
    parser.add_argument('--model_dir', required=True, help='Directory containing model checkpoint (e.g., models/net1)')
    parser.add_argument('--output', help='Output .tflite file path. Defaults to model_dir/model.tflite')
    
    args = parser.parse_args()
    
    model_dir = os.path.abspath(args.model_dir)
    
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(model_dir, 'model.tflite')
        
    try:
        convert_to_tflite(model_dir, output_file)
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
