import os
import glob
import numpy as np
import tensorflow as tf
from pycochleagram import erbfilter as erb, cochleagram as cgram
import scipy.signal as signallib

# Limit GPU memory growth
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

def load_wav_file(wav_path):
    from scipy.io import wavfile
    sr, audio = wavfile.read(wav_path)
    if audio.ndim == 1: audio = np.stack([audio, audio], axis=1)
    if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
    return audio, sr

def make_downsample_filter(sr=48000, env_sr=8000, window_size=4097, beta=10.06):
    downsample_ratio = sr // env_sr
    downsample_filter_times = np.arange(-window_size/2, int(window_size/2))
    downsample_filter_response_orig = np.sinc(downsample_filter_times / downsample_ratio) / downsample_ratio
    downsample_filter_window = signallib.kaiser(window_size, beta)
    downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    return downsample_filter_response.astype(np.float32), downsample_ratio

def wav_to_model_input(audio, sr=48000, target_sr=48000, n_channels=39, low_lim=30, hi_lim=20000, final_samples=8000):
    # Generates 48kHz cochleagram
    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
    cochleagrams = []
    for channel in range(2):
        coch = cgram.cochleagram(audio[:, channel], sr, n, low_lim, hi_lim, 1, 
                                 ret_mode='subband', strict=False)
        if coch.shape[0] > n_channels:
            start_idx = (coch.shape[0] - n_channels) // 2
            coch = coch[start_idx:start_idx + n_channels]
        cochleagrams.append(coch)
    coch_stereo = np.stack(cochleagrams, axis=2).astype(np.float32)
    
    # Downsample to 8kHz and Rectify
    downsample_filter, ratio = make_downsample_filter()
    downsampled = np.zeros((n_channels, coch_stereo.shape[1] // ratio, 2), dtype=np.float32)
    for ear in range(2):
        for ch in range(n_channels):
            filtered = np.convolve(coch_stereo[ch, :, ear], downsample_filter, mode='same')
            downsampled[ch, :, ear] = filtered[::ratio][:downsampled.shape[1]]
    
    downsampled = np.maximum(downsampled, 0) # ReLU
    
    # Ensure exact length
    if downsampled.shape[1] < final_samples:
        pad_width = final_samples - downsampled.shape[1]
        downsampled = np.pad(downsampled, ((0, 0), (0, pad_width), (0, 0)))
    else:
        downsampled = downsampled[:, :final_samples, :]
        
    return downsampled

def get_dataset(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.wav"))
    print(f"Found {len(files)} items in {data_dir}. Extracting cochleagrams (this takes a moment)...")
    
    inputs, labels = [], []
    for f in files:
        # Assumes filename format: anything_azXXX_anything.wav  e.g., echo_az245_001.wav
        # Extract azimuth value
        try:
            filename = os.path.basename(f)
            # Find the 'az' token and get the number after it
            az_str = [part for part in filename.replace('_', '-').split('-') if "az" in part][0]
            az = int(az_str.replace("az", ""))
            if az < 0: az += 360
            
            # Map absolute degree to bin index (5-deg bins)
            az_bin = az // 5
            
            # Compute Cochleagram
            audio, sr = load_wav_file(f)
            features = wav_to_model_input(audio, sr)
            
            inputs.append(features)
            labels.append(az_bin)
            
            if len(inputs) % 100 == 0:
                print(f"Processed {len(inputs)} / {len(files)}...")
        except Exception as e:
            print(f"Error parsing file {f}: {e}")
            
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Directory with the 7200 wav files")
    parser.add_argument('--model_dir', required=True, help="Path to models/net1 containing original checkpoints")
    parser.add_argument('--output_dir', required=True, help="Where to save finetuned checkpoints")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    # 1. Load Data
    X_train, Y_train = get_dataset(args.data_dir)
    print(f"Dataset shape: X={X_train.shape}, Y={Y_train.shape}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. Build Graph
    tf.compat.v1.reset_default_graph()
    config_array = np.load(os.path.join(args.model_dir, 'config_array.npy'), allow_pickle=True)
    
    # Remap everything to single GPU/CPU
    target_device = '/gpu:0'
    def remap_devices(arr):
        if isinstance(arr, np.ndarray): return np.array([remap_devices(x) for x in arr], dtype=object)
        elif isinstance(arr, list): return [remap_devices(x) for x in arr]
        elif isinstance(arr, str) and '/gpu' in arr.lower(): return target_device
        return arr
    config_array = remap_devices(config_array)

    from NetBuilder_valid_pad import NetBuilder
    input_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 39, 8000, 2], name='input')
    labels_placeholder = tf.compat.v1.placeholder(tf.int64, [None], name='labels')
    
    # Non-linear scaling as in original network
    nonlin = tf.pow(input_placeholder, 0.3)
    
    net = NetBuilder()
    out = net.build(config_array, nonlin, training_state=True, dropout_training_state=True,
                    filter_dtype=tf.float32, padding='VALID', n_classes_localization=504,
                    n_classes_recognition=780, branched=False, regularizer=None)

    # Calculate classification loss
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=labels_placeholder))
    
    # Accuracy metric
    correct_pred = tf.equal(tf.argmax(out, 1), labels_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Optimizer (All layers un-frozen)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) # Needed for BatchNorm
    with tf.control_dependencies(update_ops):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4).minimize(cost)

    # 3. Training Loop
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Restore the original weights!
    saver = tf.compat.v1.train.Saver(max_to_keep=3)
    # Search for the checkpoint file dynamically
    chkpt = tf.train.get_checkpoint_state(args.model_dir)
    if chkpt and chkpt.model_checkpoint_path:
        ckpt_path = chkpt.model_checkpoint_path
    else:
        # Fallback if checkpoint state file is missing
        ckpt_path = os.path.join(args.model_dir, 'model.ckpt-100000')
    
    print(f"Restoring old weights from: {ckpt_path}")
    saver.restore(sess, ckpt_path)
    
    print("\n--- Starting Finetuning ---")
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        epoch_loss = 0
        epoch_acc = 0
        batches = 0
        
        for start_idx in range(0, n_samples, args.batch_size):
            batch_idx = indices[start_idx:start_idx + args.batch_size]
            batch_x = X_train[batch_idx]
            batch_y = Y_train[batch_idx]
            
            _, loss_val, acc_val = sess.run([optimizer, cost, accuracy], 
                                            feed_dict={input_placeholder: batch_x, labels_placeholder: batch_y})
            
            epoch_loss += loss_val
            epoch_acc += acc_val
            batches += 1
            
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/batches:.4f}, Accuracy: {epoch_acc/batches:.4f}")
        
    print("Saving finetuned model...")
    save_path = saver.save(sess, os.path.join(args.output_dir, 'model.ckpt'), global_step=args.epochs)
    print(f"Model saved to: {save_path}")
    print("Done! You can now convert this new checkpoint using convert_to_tflite.py")

if __name__ == '__main__':
    main()
