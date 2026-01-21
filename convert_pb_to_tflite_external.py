import tensorflow as tf
import argparse
import os

def convert_pb_to_tflite(pb_file, output_file):
    """
    Converts a frozen graph (.pb) to TFLite (.tflite).
    This script is intended to be run in a TF 2.x environment or a TF 1.15 environment 
    that has TFLite support enabled.
    """
    print(f"Converting {pb_file} to {output_file}...")
    
    try:
        # TF 2.x approach
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file=pb_file,
            input_arrays=['input'],
            output_arrays=['output_probabilities'],
            input_shapes={'input': [1, 39, 8000, 2]}
        )
    except (AttributeError, ImportError):
        try:
            # TF 1.x approach
            converter = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file=pb_file,
                input_arrays=['input'],
                output_arrays=['output_probabilities'],
                input_shapes={'input': [1, 39, 8000, 2]}
            )
        except Exception as e:
            print(f"Error initializing converter: {e}")
            return

    # optimize for size/compatibility if needed
    converter.allow_custom_ops = True
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PB to TFLite (Run in environment with working TFLite)')
    parser.add_argument('pb_file', help='Path to frozen graph .pb file')
    parser.add_argument('output_file', help='Path to output .tflite file')
    
    args = parser.parse_args()
    convert_pb_to_tflite(args.pb_file, args.output_file)
