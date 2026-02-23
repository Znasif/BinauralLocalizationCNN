import os
import glob
import random
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Limit GPU memory growth
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


# ─────────────────────────────────────────────────────────
#  Dataset Pipeline
# ─────────────────────────────────────────────────────────

def parse_tfrecord_fn(example_proto):
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
    image = tf.io.decode_raw(parsed['train/image'], tf.float32)
    height = tf.cast(parsed['train/image_height'], tf.int32)
    width  = tf.cast(parsed['train/image_width'],  tf.int32)
    image  = tf.reshape(image, (height, width, 2))
    label      = parsed['train/class_num']
    click_type = parsed['train/click_type']
    return image, label, click_type


def make_pipeline(shard_paths, batch_size, shuffle=True):
    """
    Build a tf.data pipeline from a list of TFRecord shard paths.
    When shuffle=True (training), we interleave multiple shards in parallel
    so 0-click and 1-click examples are deeply mixed inside every batch.
    """
    file_ds = tf.data.Dataset.from_tensor_slices(shard_paths)
    if shuffle:
        file_ds = file_ds.shuffle(buffer_size=len(shard_paths))

    record_ds = file_ds.interleave(
        lambda path: tf.data.TFRecordDataset(path, compression_type='GZIP'),
        cycle_length=min(10, len(shard_paths)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    record_ds = record_ds.map(parse_tfrecord_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        record_ds = record_ds.shuffle(buffer_size=2000)
    record_ds = record_ds.batch(batch_size, drop_remainder=True)
    record_ds = record_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator    = tf.compat.v1.data.make_initializable_iterator(record_ds)
    next_element = iterator.get_next()
    return iterator, next_element


def count_records(shard_paths):
    """Count total records across a list of TFRecord shards."""
    n = 0
    opts = tf.io.TFRecordOptions(
        tf.compat.v1.io.TFRecordCompressionType.GZIP)
    for path in shard_paths:
        for _ in tf.compat.v1.python_io.tf_record_iterator(path, options=opts):
            n += 1
    return n


def split_shards(all_shards, val_fraction=0.2):
    """
    Sort shards deterministically, then split into (train_shards, val_shards).
    We separate 0-click and 1-click shards before splitting so
    both sets end up balanced.
    Sorting (not shuffling) ensures the same shards always go to train vs val
    regardless of how many times finetune_custom.py is rerun.
    """
    shards_0 = sorted([s for s in all_shards if '0click' in os.path.basename(s)])
    shards_1 = sorted([s for s in all_shards if '1click' in os.path.basename(s)])

    def holdout(lst, frac):
        n_val = max(1, int(len(lst) * frac))
        # Take val from the end so the split is stable as new shards are added
        return lst[:-n_val], lst[-n_val:]

    train_0, val_0 = holdout(shards_0, val_fraction)
    train_1, val_1 = holdout(shards_1, val_fraction)

    train_shards = train_0 + train_1
    val_shards   = val_0   + val_1
    return train_shards, val_shards


# ─────────────────────────────────────────────────────────
#  Per-click-type accuracy helper (runs in NumPy, no graph)
# ─────────────────────────────────────────────────────────

def split_accuracy(preds, labels, click_types):
    """Returns (acc_0click, acc_1click, n_0click, n_1click)."""
    correct = (preds == labels)
    m0 = (click_types == 0)
    m1 = (click_types == 1)
    acc0 = float(correct[m0].mean()) if m0.any() else float('nan')
    acc1 = float(correct[m1].mean()) if m1.any() else float('nan')
    return acc0, acc1, int(m0.sum()), int(m1.sum())

def fold_front_back(az_deg):
    """Map azimuth (0-360) to 0-90 by reflecting across median and coronal planes.
    This matches the paper's front-back folding convention."""
    az = az_deg % 360
    az = np.where(az > 180, 360 - az, az)
    az = np.where(az > 90, 180 - az, az)
    return az

def calc_mae(preds, labels):
    # Classes: 504 total = 7 elevations * 72 azimuths
    # az_bin = class % 72
    # el_bin = class // 72
    p_az = (preds % 72) * 5
    p_el = (preds // 72) * 10

    l_az = (labels % 72) * 5
    l_el = (labels // 72) * 10

    # Handle azimuth wraparound natively (e.g. 355 vs 0 is 5 deg, not 355 deg)
    diff_az = np.abs(p_az - l_az)
    err_az = np.minimum(diff_az, 360 - diff_az)

    # Front-back folded azimuth MAE (comparable to paper's 4.4 deg)
    err_az_fb = np.abs(fold_front_back(p_az) - fold_front_back(l_az))

    err_el = np.abs(p_el - l_el)

    return err_az.sum(), err_el.sum(), err_az_fb.sum()


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_dir', required=True,
                        help="Directory containing *.tfrecords shards")
    parser.add_argument('--model_dir',   required=True,
                        help="Path to original checkpoint (e.g. models/net1)")
    parser.add_argument('--output_dir',  required=True,
                        help="Where to save finetuned checkpoints")
    parser.add_argument('--log_dir',     default='logs',
                        help="TensorBoard log directory")
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=5e-5)
    parser.add_argument('--val_fraction',type=float, default=0.2,
                        help="Fraction of shards to hold out for validation")
    parser.add_argument('--original_eval_shard', default='data/data_original_eval.tfrecords',
                        help="Pre-downsampled original-data shard for forgetting eval "
                             "(created by create_original_eval_shard.py). "
                             "Skipped if the file does not exist.")
    parser.add_argument('--freeze_conv', action='store_true',
                        help="Freeze all conv/BN layers; only train fully-connected layers. "
                             "Reduces overfitting when the finetune dataset is small.")
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help="L2 weight decay coefficient (e.g. 1e-4). 0 = disabled.")
    parser.add_argument('--save_best_val', action='store_true',
                        help="Also save the checkpoint with the highest val acc_0click "
                             "as best_val.ckpt (in addition to the rolling checkpoints).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'train'),    exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'val'),      exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'original'), exist_ok=True)

    use_orig_eval = os.path.exists(args.original_eval_shard)
    if use_orig_eval:
        print(f"Forgetting eval shard: {args.original_eval_shard}")
    else:
        print(f"[INFO] No forgetting eval shard found at '{args.original_eval_shard}'. "
              f"Run create_original_eval_shard.py to enable acc_original tracking.")

    # ── 1. Discover and split shards ──────────────────────
    all_shards = glob.glob(os.path.join(args.tfrecords_dir, '*.tfrecords'))
    if not all_shards:
        print(f"Error: No .tfrecords found in {args.tfrecords_dir}")
        return

    train_shards, val_shards = split_shards(all_shards, args.val_fraction)
    print(f"Shards  →  Train: {len(train_shards)}  |  Val: {len(val_shards)}")

    print("Counting training records (one-time scan)...")
    n_train = count_records(train_shards)
    n_val   = count_records(val_shards)
    print(f"Records →  Train: {n_train}  |  Val: {n_val}")

    batches_per_epoch = n_train // args.batch_size
    val_batches       = n_val   // args.batch_size

    # ── 2. Build TF graph ─────────────────────────────────
    tf.compat.v1.reset_default_graph()
    config_array = np.load(
        os.path.join(args.model_dir, 'config_array.npy'), allow_pickle=True)

    # Remap everything to single GPU
    def remap_devices(arr):
        if isinstance(arr, np.ndarray):
            return np.array([remap_devices(x) for x in arr], dtype=object)
        if isinstance(arr, list):
            return [remap_devices(x) for x in arr]
        if isinstance(arr, str) and '/gpu' in arr.lower():
            return '/gpu:0'
        return arr
    config_array = remap_devices(config_array)

    from NetBuilder_valid_pad import NetBuilder

    input_ph  = tf.compat.v1.placeholder(
        tf.float32, [args.batch_size, 39, 8000, 2], name='input')
    labels_ph = tf.compat.v1.placeholder(
        tf.int64, [args.batch_size], name='labels')

    is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name='is_training')

    nonlin = tf.pow(input_ph, 0.3)
    net    = NetBuilder()
    out    = net.build(config_array, nonlin,
                       training_state=is_training, dropout_training_state=is_training,
                       filter_dtype=tf.float32, padding='VALID',
                       n_classes_localization=504,
                       n_classes_recognition=780,
                       branched=False, regularizer=None)

    cost     = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=out, labels=labels_ph))
    preds_op = tf.argmax(out, 1)

    # Optionally freeze conv/BN vars — only compute gradients for FC layers
    if args.freeze_conv:
        trainable = [v for v in tf.compat.v1.trainable_variables()
                     if '_fc_' in v.name or '_out_' in v.name]
        if not trainable:
            print("[WARN] --freeze_conv: no FC variables matched by name; training all vars.")
            trainable = None
        else:
            print(f"--freeze_conv: training {len(trainable)} FC variables only.")
    else:
        trainable = None  # train everything

    # L2 weight decay (scoped to trainable vars only when freeze_conv is active)
    if args.weight_decay > 0:
        wd_candidates = trainable if trainable else tf.compat.v1.trainable_variables()
        l2_vars = [v for v in wd_candidates if 'bias' not in v.name]
        if l2_vars:
            l2_loss = args.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in l2_vars])
            cost    = cost + l2_loss
            print(f"Weight decay: {args.weight_decay} applied to {len(l2_vars)} variables")

    # Filter BN update_ops: when freeze_conv, only update FC-layer BN stats
    all_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    if args.freeze_conv and trainable:
        # Count conv BN layers in config to identify FC BN scope index
        n_conv_bn = 0
        for lst in config_array:
            for element in lst[1:]:
                if element[0] == 'bn':
                    n_conv_bn += 1
                if element[0] == 'branch':
                    break
        # FC bn is the next scope after all conv bn layers
        if n_conv_bn == 0:
            fc_bn_prefix = 'batch_normalization/'
        else:
            fc_bn_prefix = f'batch_normalization_{n_conv_bn}/'
        update_ops = [op for op in all_update_ops if fc_bn_prefix in op.name]
        print(f"--freeze_conv: filtered UPDATE_OPS from {len(all_update_ops)} to "
              f"{len(update_ops)} (FC BN only, prefix='{fc_bn_prefix}')")
    else:
        update_ops = all_update_ops

    with tf.control_dependencies(update_ops):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=args.lr, epsilon=1e-4).minimize(cost, var_list=trainable)


    # TensorBoard summaries — scalars written from Python via feed-dict
    loss_ph     = tf.compat.v1.placeholder(tf.float32, name='loss_summary_ph')
    acc_ph      = tf.compat.v1.placeholder(tf.float32, name='acc_summary_ph')
    acc0_ph     = tf.compat.v1.placeholder(tf.float32, name='acc0_summary_ph')
    acc1_ph     = tf.compat.v1.placeholder(tf.float32, name='acc1_summary_ph')
    orig_acc_ph = tf.compat.v1.placeholder(tf.float32, name='orig_acc_ph')
    tf.compat.v1.summary.scalar('loss',          loss_ph)
    tf.compat.v1.summary.scalar('accuracy',      acc_ph)
    tf.compat.v1.summary.scalar('acc_0click',    acc0_ph)
    tf.compat.v1.summary.scalar('acc_1click',    acc1_ph)
    
    mae_az_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_az_summary_ph')
    mae_el_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_el_summary_ph')
    mae_az_fb_ph = tf.compat.v1.placeholder(tf.float32, name='mae_az_fb_summary_ph')
    tf.compat.v1.summary.scalar('mae_az_deg', mae_az_ph)
    tf.compat.v1.summary.scalar('mae_el_deg', mae_el_ph)
    tf.compat.v1.summary.scalar('mae_az_fb_deg', mae_az_fb_ph)
    
    summary_op = tf.compat.v1.summary.merge_all()
    orig_summary_op = tf.compat.v1.summary.scalar('acc_original', orig_acc_ph)

    # ── 3. Data iterators (built after graph, before session) ──
    train_iter, train_next = make_pipeline(train_shards, args.batch_size, shuffle=True)
    val_iter,   val_next   = make_pipeline(val_shards,   args.batch_size, shuffle=False)
    if use_orig_eval:
        orig_iter, orig_next = make_pipeline([args.original_eval_shard],
                                             args.batch_size, shuffle=False)
        orig_batches = count_records([args.original_eval_shard]) // args.batch_size

    # ── 4. Session + weight restore ───────────────────────
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(max_to_keep=3)

    # Resume from output_dir if a checkpoint already exists there,
    # otherwise fall back to the pre-trained model_dir weights.
    resume_ckpt = tf.train.get_checkpoint_state(args.output_dir)
    if resume_ckpt and resume_ckpt.model_checkpoint_path:
        restore_path = resume_ckpt.model_checkpoint_path
        print(f"Resuming finetuning from: {restore_path}")
    else:
        pretrain_ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if pretrain_ckpt and pretrain_ckpt.model_checkpoint_path:
            restore_path = pretrain_ckpt.model_checkpoint_path
        else:
            restore_path = os.path.join(args.model_dir, 'model.ckpt-100000')
        print(f"Starting fresh finetuning from: {restore_path}")

    saver.restore(sess, restore_path)

    train_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(args.log_dir, 'train'), sess.graph)
    val_writer   = tf.compat.v1.summary.FileWriter(
        os.path.join(args.log_dir, 'val'))
    orig_writer  = tf.compat.v1.summary.FileWriter(
        os.path.join(args.log_dir, 'original'))

    global_step = 0
    best_val_acc0 = -1.0  # for --save_best_val

    # ── 5. Training loop ──────────────────────────────────
    print("\n─── Starting Finetuning ───")
    print(f"    Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"    ~{batches_per_epoch} train batches / epoch")
    print(f"    ~{val_batches} validation batches / epoch\n")

    for epoch in range(args.epochs):
        sess.run(train_iter.initializer)

        e_loss = 0.0
        e_correct = e_total = 0
        e_correct0 = e_total0 = e_correct1 = e_total1 = 0
        e_mae_az = e_mae_el = e_mae_az_fb = 0.0

        for b in range(batches_per_epoch):
            try:
                bx, by, bc = sess.run(train_next)
            except tf.errors.OutOfRangeError:
                break

            _, loss_val, preds_val = sess.run(
                [optimizer, cost, preds_op],
                feed_dict={input_ph: bx, labels_ph: by, is_training: True})

            correct = (preds_val == by)
            e_loss     += loss_val
            e_correct  += correct.sum()
            e_total    += len(by)

            m_az, m_el, m_az_fb = calc_mae(preds_val, by)
            e_mae_az   += m_az
            e_mae_el   += m_el
            e_mae_az_fb += m_az_fb

            m0, m1 = (bc == 0), (bc == 1)
            e_correct0 += correct[m0].sum();  e_total0 += m0.sum()
            e_correct1 += correct[m1].sum();  e_total1 += m1.sum()

            global_step += 1

            if (b + 1) % 50 == 0:
                acc0 = correct[m0].mean() if m0.any() else float('nan')
                acc1 = correct[m1].mean() if m1.any() else float('nan')
                print(f"  Epoch {epoch+1} / Batch {b+1:4d}"
                      f"  Loss: {loss_val:.4f}"
                      f"  Acc: {correct.mean():.4f}"
                      f"  [0-Click: {acc0:.2f}  1-Click: {acc1:.2f}]")

        # ── Epoch-level train stats ──
        e_acc  = e_correct  / e_total  if e_total  else 0
        e_acc0 = e_correct0 / e_total0 if e_total0 else 0
        e_acc1 = e_correct1 / e_total1 if e_total1 else 0
        e_avg_loss = e_loss / batches_per_epoch
        e_avg_mae_az = e_mae_az / e_total if e_total else 0
        e_avg_mae_el = e_mae_el / e_total if e_total else 0
        e_avg_mae_az_fb = e_mae_az_fb / e_total if e_total else 0

        print(f"\n→ END EPOCH {epoch+1}/{args.epochs}"
              f"  AvgLoss: {e_avg_loss:.4f}"
              f"  Acc: {e_acc:.4f}"
              f"  [0-Click: {e_acc0:.4f}  1-Click: {e_acc1:.4f}]"
              f"  MAE Degrees: [Az: {e_avg_mae_az:.1f}°  Az(FB): {e_avg_mae_az_fb:.1f}°  El: {e_avg_mae_el:.1f}°]")

        # Write train summaries
        summ = sess.run(summary_op, feed_dict={
            loss_ph: e_avg_loss, acc_ph: e_acc,
            acc0_ph: e_acc0, acc1_ph: e_acc1,
            mae_az_ph: e_avg_mae_az, mae_el_ph: e_avg_mae_el,
            mae_az_fb_ph: e_avg_mae_az_fb})
        train_writer.add_summary(summ, epoch + 1)

        # ── Validation pass ───────────────────────────────
        sess.run(val_iter.initializer)
        v_loss = 0.0
        v_correct = v_total = 0
        v_correct0 = v_total0 = v_correct1 = v_total1 = 0
        v_mae_az = v_mae_el = v_mae_az_fb = 0.0

        for _ in range(val_batches):
            try:
                bx, by, bc = sess.run(val_next)
            except tf.errors.OutOfRangeError:
                break
            loss_val, preds_val = sess.run(
                [cost, preds_op],
                feed_dict={input_ph: bx, labels_ph: by, is_training: False})

            correct = (preds_val == by)
            v_loss    += loss_val
            v_correct += correct.sum()
            v_total   += len(by)

            m_az, m_el, m_az_fb = calc_mae(preds_val, by)
            v_mae_az  += m_az
            v_mae_el  += m_el
            v_mae_az_fb += m_az_fb
            
            m0, m1 = (bc == 0), (bc == 1)
            v_correct0 += correct[m0].sum();  v_total0 += m0.sum()
            v_correct1 += correct[m1].sum();  v_total1 += m1.sum()

        v_acc  = v_correct  / v_total  if v_total  else 0
        v_acc0 = v_correct0 / v_total0 if v_total0 else 0
        v_acc1 = v_correct1 / v_total1 if v_total1 else 0
        v_avg_loss = v_loss / val_batches if val_batches else 0
        v_avg_mae_az = v_mae_az / v_total if v_total else 0
        v_avg_mae_el = v_mae_el / v_total if v_total else 0
        v_avg_mae_az_fb = v_mae_az_fb / v_total if v_total else 0

        print(f"  VAL  Epoch {epoch+1}/{args.epochs}"
              f"  AvgLoss: {v_avg_loss:.4f}"
              f"  Acc: {v_acc:.4f}"
              f"  [0-Click: {v_acc0:.4f}  1-Click: {v_acc1:.4f}]"
              f"  MAE Degrees: [Az: {v_avg_mae_az:.1f}°  Az(FB): {v_avg_mae_az_fb:.1f}°  El: {v_avg_mae_el:.1f}°]")

        # Write val summaries
        summ = sess.run(summary_op, feed_dict={
            loss_ph: v_avg_loss, acc_ph: v_acc,
            acc0_ph: v_acc0, acc1_ph: v_acc1,
            mae_az_ph: v_avg_mae_az, mae_el_ph: v_avg_mae_el,
            mae_az_fb_ph: v_avg_mae_az_fb})
        val_writer.add_summary(summ, epoch + 1)

        # ── Forgetting eval on original data ─────────────
        if use_orig_eval:
            sess.run(orig_iter.initializer)
            o_correct = o_total = 0
            for _ in range(orig_batches):
                try:
                    bx, by, _ = sess.run(orig_next)
                except tf.errors.OutOfRangeError:
                    break
                preds_val = sess.run(preds_op, feed_dict={input_ph: bx, labels_ph: by, is_training: False})
                o_correct += (preds_val == by).sum()
                o_total   += len(by)
            o_acc = o_correct / o_total if o_total else 0.0
            print(f"  ORIG Epoch {epoch+1}/{args.epochs}"
                  f"  acc_original: {o_acc:.4f}  ({o_correct}/{o_total})")
            summ = sess.run(orig_summary_op, feed_dict={orig_acc_ph: o_acc})
            orig_writer.add_summary(summ, epoch + 1)
            orig_writer.flush()

        # ── Save checkpoint ───────────────────────────────
        ckpt_path = saver.save(
            sess,
            os.path.join(args.output_dir, 'model.ckpt'),
            global_step=epoch + 1)
        print(f"  Checkpoint saved: {ckpt_path}")

        if args.save_best_val and v_acc0 > best_val_acc0:
            best_val_acc0 = v_acc0
            best_path = saver.save(
                sess, os.path.join(args.output_dir, 'best_val.ckpt'))
            print(f"  ★ New best val acc_0click={v_acc0:.4f} → {best_path}")
        print()

    train_writer.close()
    val_writer.close()
    orig_writer.close()
    print("Done!  Run:  tensorboard --logdir logs/")
    print(f"Final checkpoint in: {args.output_dir}")
    print("Convert with:  python convert_to_tflite.py")


if __name__ == '__main__':
    main()
