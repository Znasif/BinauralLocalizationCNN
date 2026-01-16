"""
Script to extract weights from TensorFlow 1.x checkpoint and save to .npz file.
Run this INSIDE the TF 1.x Docker container.
"""
import os
import tensorflow as tf
import numpy as np
import sys

# Disable GPU for extraction script
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def extract_weights(model_dir, output_path):
    print(f"Loading checkpoint from {model_dir}")
    
    # We need to construct the graph first? 
    # Actually, we can just use inspect_checkpoint features or restore into a session with import_meta_graph,
    # BUT the code doesn't save meta graphs (write_meta_graph=False in tf_record...py line 714).
    #
    # However, New checkpoints (Saver) usually save .meta unless explicitly disabled efficiently.
    # The code `saver.save(sess, ..., write_meta_graph=False)` implies no meta file.
    # 
    # BUT, we have `model.ckpt-100000.data...` and `index`.
    # Without meta graph, we can't `import_meta_graph`.
    #
    # We have two options:
    # 1. Build the graph using NetBuilder (like inference script) and restore. (Safest)
    # 2. Use `tf.train.list_variables` and `tf.train.load_variable` to read directly from file without graph. (Easiest)
    
    checkpoint_path = os.path.join(model_dir, 'model.ckpt-100000')
    
    try:
        vars_list = tf.train.list_variables(checkpoint_path)
    except Exception as e:
        print(f"Error listing variables: {e}")
        return
        
    print(f"Found {len(vars_list)} variables.")
    
    weights = {}
    
    for name, shape in vars_list:
        if "Adam" in name or "power" in name:
            continue # Skip optimizer variables
            
        print(f"Loading {name} {shape}")
        data = tf.train.load_variable(checkpoint_path, name)
        weights[name] = data
        
    print(f"Saving {len(weights)} variables to {output_path}")
    np.savez(output_path, **weights)
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_weights_tf1.py <model_dir> <output_file>")
        sys.exit(1)
        
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    extract_weights(model_dir, output_path)
