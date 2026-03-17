import tensorflow as tf
import os

# Ensure logs are visible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def verify_gpu():
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check if GPU is available
    if tf.test.is_gpu_available():
        print("GPU is available via tf.test.is_gpu_available()")
    else:
        print("GPU is NOT available via tf.test.is_gpu_available()")

    print("Attempting simple matrix multiplication on GPU...")
    
    with tf.device('/gpu:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        
        # Test Conv2D as well, since that's what the model uses
        print("Attempting Conv2D on GPU...")
        input_data = tf.random.normal([1, 10, 10, 3])
        filter_data = tf.random.normal([3, 3, 3, 8])
        conv_op = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.compat.v1.Session(config=config) as sess:
        try:
            result = sess.run(c)
            print("Successfully ran matmul on GPU!")
            print(f"Result:\n{result}")
            
            print("Running Conv2D...")
            conv_result = sess.run(conv_op)
            print("Successfully ran Conv2D on GPU!")
            print(f"Conv result shape: {conv_result.shape}")
            
        except Exception as e:
            print(f"Failed to run on GPU: {e}")

if __name__ == "__main__":
    verify_gpu()
