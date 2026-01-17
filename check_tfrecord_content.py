#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import argparse

def inspect_tfrecord(tfrecord_path, index):
    print(f"Inspecting {tfrecord_path} at index {index}")
    options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    iterator = tf.python_io.tf_record_iterator(tfrecord_path, options=options)
    
    for i, serialized_example in enumerate(iterator):
        if i == index:
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            features = example.features.feature
            
            image_bytes = features['train/image'].bytes_list.value[0]
            try:
                data = np.frombuffer(image_bytes, dtype=np.float32)
            except:
                data = np.frombuffer(image_bytes, dtype=np.float64).astype(np.float32)
            
            print(f"Data shape (flat): {data.shape}")
            print(f"Min: {data.min()}")
            print(f"Max: {data.max()}")
            print(f"Mean: {data.mean()}")
            print(f"Std: {data.std()}")
            print(f"Non-zero count: {np.count_nonzero(data)}")
            
            # Check for rectification
            if data.min() >= 0:
                print("Data appears to be RECTIFIED (non-negative).")
            else:
                print("Data contains NEGATIVE values (raw subbands?).")
                
            return
    print("Index not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tfrecord')
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    inspect_tfrecord(args.tfrecord, args.index)
