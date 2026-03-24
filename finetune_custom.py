import os
import glob
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_io       import parse_tfrecord_fn, make_pipeline, count_records, split_shards
from src.graph_builder import load_config, build_training_graph
from src.inference     import make_session_config, restore_checkpoint
from src.metrics       import fold_front_back, calc_mae, split_accuracy, print_epoch_table


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_dir', required=True,
                        help='Directory containing *.tfrecords shards')
    parser.add_argument('--model_dir',    required=True,
                        help='Path to original checkpoint (e.g. models/net1)')
    parser.add_argument('--output_dir',   required=True,
                        help='Where to save finetuned checkpoints')
    parser.add_argument('--log_dir',      default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Fraction of shards to hold out for validation')
    parser.add_argument('--original_eval_shard',
                        default='data/data_original_eval.tfrecords',
                        help='Pre-downsampled original-data shard for forgetting eval. '
                             'Skipped if the file does not exist.')
    parser.add_argument('--freeze_conv',  action='store_true',
                        help='Freeze all conv/BN layers; only train FC layers.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 weight decay coefficient (0 = disabled).')
    parser.add_argument('--bn_momentum',  type=float, default=0.99,
                        help='Batch normalisation momentum for moving averages.')
    parser.add_argument('--freeze_bn_stats', action='store_true',
                        help='Freeze BN running mean/variance (use pretrained stats).')
    parser.add_argument('--save_best_val',   action='store_true',
                        help='Also save best_val.ckpt for the highest val acc_0click.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'train'),    exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'val'),      exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'original'), exist_ok=True)

    use_orig_eval = os.path.exists(args.original_eval_shard)
    if use_orig_eval:
        print(f'Forgetting eval shard: {args.original_eval_shard}')
    else:
        print(f'[INFO] No forgetting eval shard at "{args.original_eval_shard}". '
              f'Run create_original_eval_shard.py to enable acc_original tracking.')

    # ── 1. Discover and split shards ──────────────────────
    all_shards = glob.glob(os.path.join(args.tfrecords_dir, '*.tfrecords'))
    if not all_shards:
        print(f'Error: No .tfrecords found in {args.tfrecords_dir}')
        return

    train_shards, val_shards = split_shards(all_shards, args.val_fraction)
    print(f'Shards  →  Train: {len(train_shards)}  |  Val: {len(val_shards)}')

    print('Counting training records (one-time scan)...')
    n_train = count_records(train_shards)
    n_val   = count_records(val_shards)
    print(f'Records →  Train: {n_train}  |  Val: {n_val}')

    batches_per_epoch = n_train // args.batch_size
    val_batches       = n_val   // args.batch_size

    # ── 2. Build TF graph ─────────────────────────────────
    tf.compat.v1.reset_default_graph()

    config_array = load_config(args.model_dir, target_device='/gpu:0')
    input_ph, labels_ph, is_training, out, cost, preds_op = build_training_graph(
        config_array, args.batch_size, n_classes=504, bn_momentum=args.bn_momentum)

    # Optionally freeze conv/BN vars
    if args.freeze_conv:
        trainable = [v for v in tf.compat.v1.trainable_variables()
                     if '_fc_' in v.name or '_out_' in v.name]
        if not trainable:
            print('[WARN] --freeze_conv: no FC variables matched; training all vars.')
            trainable = None
        else:
            print(f'--freeze_conv: training {len(trainable)} FC variables only.')
    else:
        trainable = None

    # L2 weight decay
    if args.weight_decay > 0:
        wd_candidates = trainable if trainable else tf.compat.v1.trainable_variables()
        l2_vars = [v for v in wd_candidates if 'bias' not in v.name]
        if l2_vars:
            l2_loss = args.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in l2_vars])
            cost    = cost + l2_loss
            print(f'Weight decay: {args.weight_decay} on {len(l2_vars)} variables')

    # BN update ops
    all_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    if args.freeze_bn_stats:
        update_ops = []
        print(f'--freeze_bn_stats: all BN UPDATE_OPS disabled '
              f'(skipped {len(all_update_ops)} ops)')
    elif args.freeze_conv and trainable:
        n_conv_bn = sum(
            1 for lst in config_array
            for element in lst[1:]
            if element[0] == 'bn')
        fc_bn_prefix = (f'batch_normalization_{n_conv_bn}/'
                        if n_conv_bn > 0 else 'batch_normalization/')
        update_ops = [op for op in all_update_ops if fc_bn_prefix in op.name]
        print(f'--freeze_conv: UPDATE_OPS filtered '
              f'{len(all_update_ops)} → {len(update_ops)} (FC BN only)')
    else:
        update_ops = all_update_ops

    with tf.control_dependencies(update_ops):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=args.lr, epsilon=1e-4).minimize(cost, var_list=trainable)

    # TensorBoard summary placeholders
    loss_ph     = tf.compat.v1.placeholder(tf.float32, name='loss_summary_ph')
    acc_ph      = tf.compat.v1.placeholder(tf.float32, name='acc_summary_ph')
    acc0_ph     = tf.compat.v1.placeholder(tf.float32, name='acc0_summary_ph')
    acc1_ph     = tf.compat.v1.placeholder(tf.float32, name='acc1_summary_ph')
    tf.compat.v1.summary.scalar('loss',       loss_ph)
    tf.compat.v1.summary.scalar('accuracy',   acc_ph)
    tf.compat.v1.summary.scalar('acc_0click', acc0_ph)
    tf.compat.v1.summary.scalar('acc_1click', acc1_ph)

    mae_az_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_az_summary_ph')
    mae_el_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_el_summary_ph')
    mae_az_fb_ph = tf.compat.v1.placeholder(tf.float32, name='mae_az_fb_summary_ph')
    tf.compat.v1.summary.scalar('mae_az_deg',    mae_az_ph)
    tf.compat.v1.summary.scalar('mae_el_deg',    mae_el_ph)
    tf.compat.v1.summary.scalar('mae_az_fb_deg', mae_az_fb_ph)

    mae_az_0ck_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_az_0ck_ph')
    mae_el_0ck_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_el_0ck_ph')
    mae_az_fb_0ck_ph = tf.compat.v1.placeholder(tf.float32, name='mae_az_fb_0ck_ph')
    mae_az_1ck_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_az_1ck_ph')
    mae_el_1ck_ph    = tf.compat.v1.placeholder(tf.float32, name='mae_el_1ck_ph')
    mae_az_fb_1ck_ph = tf.compat.v1.placeholder(tf.float32, name='mae_az_fb_1ck_ph')
    tf.compat.v1.summary.scalar('mae_az_0click',    mae_az_0ck_ph)
    tf.compat.v1.summary.scalar('mae_el_0click',    mae_el_0ck_ph)
    tf.compat.v1.summary.scalar('mae_az_fb_0click', mae_az_fb_0ck_ph)
    tf.compat.v1.summary.scalar('mae_az_1click',    mae_az_1ck_ph)
    tf.compat.v1.summary.scalar('mae_el_1click',    mae_el_1ck_ph)
    tf.compat.v1.summary.scalar('mae_az_fb_1click', mae_az_fb_1ck_ph)

    summary_op = tf.compat.v1.summary.merge_all()

    orig_acc_ph       = tf.compat.v1.placeholder(tf.float32, name='orig_acc_ph')
    orig_mae_az_ph    = tf.compat.v1.placeholder(tf.float32, name='orig_mae_az_ph')
    orig_mae_el_ph    = tf.compat.v1.placeholder(tf.float32, name='orig_mae_el_ph')
    orig_mae_az_fb_ph = tf.compat.v1.placeholder(tf.float32, name='orig_mae_az_fb_ph')
    orig_summary_op   = tf.compat.v1.summary.merge([
        tf.compat.v1.summary.scalar('acc_original',       orig_acc_ph),
        tf.compat.v1.summary.scalar('orig_mae_az_deg',    orig_mae_az_ph),
        tf.compat.v1.summary.scalar('orig_mae_el_deg',    orig_mae_el_ph),
        tf.compat.v1.summary.scalar('orig_mae_az_fb_deg', orig_mae_az_fb_ph),
    ])

    # ── 3. Data iterators ─────────────────────────────────
    train_iter, train_next = make_pipeline(train_shards, args.batch_size, shuffle=True)
    val_iter,   val_next   = make_pipeline(val_shards,   args.batch_size, shuffle=False)
    if use_orig_eval:
        orig_iter, orig_next = make_pipeline(
            [args.original_eval_shard], args.batch_size, shuffle=False)
        orig_batches = count_records([args.original_eval_shard]) // args.batch_size

    # ── 4. Session + restore ──────────────────────────────
    sess = tf.compat.v1.Session(config=make_session_config(use_gpu=True))
    sess.run(tf.compat.v1.global_variables_initializer())

    # Resume from output_dir if a checkpoint exists there; otherwise use model_dir.
    resume_state = tf.train.get_checkpoint_state(args.output_dir)
    if resume_state and resume_state.model_checkpoint_path:
        restore_path = resume_state.model_checkpoint_path
        print(f'Resuming finetuning from: {restore_path}')
    else:
        pretrain_state = tf.train.get_checkpoint_state(args.model_dir)
        restore_path = (pretrain_state.model_checkpoint_path
                        if pretrain_state and pretrain_state.model_checkpoint_path
                        else os.path.join(args.model_dir, 'model.ckpt-100000'))
        print(f'Starting fresh finetuning from: {restore_path}')

    saver, _ = restore_checkpoint(sess, args.model_dir,
                                   checkpoint_path=restore_path,
                                   max_to_keep=3)

    train_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(args.log_dir, 'train'), sess.graph)
    val_writer   = tf.compat.v1.summary.FileWriter(os.path.join(args.log_dir, 'val'))
    orig_writer  = tf.compat.v1.summary.FileWriter(os.path.join(args.log_dir, 'original'))

    global_step   = 0
    best_val_acc0 = -1.0

    # ── 5. Training loop ──────────────────────────────────
    print('\n─── Starting Finetuning ───')
    print(f'    Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}')
    if args.freeze_bn_stats:
        print('    BN stats: FROZEN (using pretrained running mean/variance)')
    else:
        print(f'    BN momentum: {args.bn_momentum}')
    print(f'    ~{batches_per_epoch} train batches / epoch')
    print(f'    ~{val_batches} validation batches / epoch\n')

    for epoch in range(args.epochs):
        sess.run(train_iter.initializer)

        e_loss = 0.0
        e_correct = e_total = 0
        e_correct0 = e_total0 = e_correct1 = e_total1 = 0
        e_mae_az = e_mae_el = e_mae_az_fb = 0.0
        e_mae_az0 = e_mae_el0 = e_mae_az_fb0 = 0.0
        e_mae_az1 = e_mae_el1 = e_mae_az_fb1 = 0.0

        for b in range(batches_per_epoch):
            try:
                bx, by, bc = sess.run(train_next)
            except tf.errors.OutOfRangeError:
                break

            _, loss_val, preds_val = sess.run(
                [optimizer, cost, preds_op],
                feed_dict={input_ph: bx, labels_ph: by, is_training: True})

            correct = (preds_val == by)
            e_loss    += loss_val
            e_correct += correct.sum()
            e_total   += len(by)

            m_az, m_el, m_az_fb = calc_mae(preds_val, by)
            e_mae_az    += m_az
            e_mae_el    += m_el
            e_mae_az_fb += m_az_fb

            m0, m1 = (bc == 0), (bc == 1)
            e_correct0 += correct[m0].sum();  e_total0 += m0.sum()
            e_correct1 += correct[m1].sum();  e_total1 += m1.sum()
            if m0.any():
                az0, el0, azfb0 = calc_mae(preds_val[m0], by[m0])
                e_mae_az0 += az0; e_mae_el0 += el0; e_mae_az_fb0 += azfb0
            if m1.any():
                az1, el1, azfb1 = calc_mae(preds_val[m1], by[m1])
                e_mae_az1 += az1; e_mae_el1 += el1; e_mae_az_fb1 += azfb1

            global_step += 1
            if (b + 1) % 50 == 0:
                acc0 = correct[m0].mean() if m0.any() else float('nan')
                acc1 = correct[m1].mean() if m1.any() else float('nan')
                print(f'  Epoch {epoch+1} / Batch {b+1:4d}'
                      f'  Loss: {loss_val:.4f}'
                      f'  Acc: {correct.mean():.4f}'
                      f'  [0-Click: {acc0:.2f}  1-Click: {acc1:.2f}]')

        # Epoch-level train stats
        e_acc  = e_correct  / e_total  if e_total  else 0
        e_acc0 = e_correct0 / e_total0 if e_total0 else 0
        e_acc1 = e_correct1 / e_total1 if e_total1 else 0
        e_avg_loss       = e_loss / batches_per_epoch
        e_avg_mae_az     = e_mae_az     / e_total  if e_total  else 0
        e_avg_mae_el     = e_mae_el     / e_total  if e_total  else 0
        e_avg_mae_az_fb  = e_mae_az_fb  / e_total  if e_total  else 0
        e_avg_mae_az0    = e_mae_az0    / e_total0 if e_total0 else float('nan')
        e_avg_mae_el0    = e_mae_el0    / e_total0 if e_total0 else float('nan')
        e_avg_mae_az_fb0 = e_mae_az_fb0 / e_total0 if e_total0 else float('nan')
        e_avg_mae_az1    = e_mae_az1    / e_total1 if e_total1 else float('nan')
        e_avg_mae_el1    = e_mae_el1    / e_total1 if e_total1 else float('nan')
        e_avg_mae_az_fb1 = e_mae_az_fb1 / e_total1 if e_total1 else float('nan')

        epoch_str = f'Epoch {epoch+1:2d}/{args.epochs}'
        print()
        print_epoch_table('TRAIN', epoch_str, [
            dict(name='all',   loss=e_avg_loss, acc=e_acc,  n_ok=e_correct,  n=e_total,
                 mae_az=e_avg_mae_az,  mae_azfb=e_avg_mae_az_fb,  mae_el=e_avg_mae_el),
            dict(name='0-clk', loss=None,       acc=e_acc0, n_ok=e_correct0, n=e_total0,
                 mae_az=e_avg_mae_az0, mae_azfb=e_avg_mae_az_fb0, mae_el=e_avg_mae_el0),
            dict(name='1-clk', loss=None,       acc=e_acc1, n_ok=e_correct1, n=e_total1,
                 mae_az=e_avg_mae_az1, mae_azfb=e_avg_mae_az_fb1, mae_el=e_avg_mae_el1),
        ])

        summ = sess.run(summary_op, feed_dict={
            loss_ph: e_avg_loss, acc_ph: e_acc, acc0_ph: e_acc0, acc1_ph: e_acc1,
            mae_az_ph: e_avg_mae_az, mae_el_ph: e_avg_mae_el,
            mae_az_fb_ph: e_avg_mae_az_fb,
            mae_az_0ck_ph: e_avg_mae_az0, mae_el_0ck_ph: e_avg_mae_el0,
            mae_az_fb_0ck_ph: e_avg_mae_az_fb0,
            mae_az_1ck_ph: e_avg_mae_az1, mae_el_1ck_ph: e_avg_mae_el1,
            mae_az_fb_1ck_ph: e_avg_mae_az_fb1})
        train_writer.add_summary(summ, epoch + 1)

        # ── Validation pass ───────────────────────────────
        sess.run(val_iter.initializer)
        v_loss = 0.0
        v_correct = v_total = 0
        v_correct0 = v_total0 = v_correct1 = v_total1 = 0
        v_mae_az = v_mae_el = v_mae_az_fb = 0.0
        v_mae_az0 = v_mae_el0 = v_mae_az_fb0 = 0.0
        v_mae_az1 = v_mae_el1 = v_mae_az_fb1 = 0.0

        for _ in range(val_batches):
            try:
                bx, by, bc = sess.run(val_next)
            except tf.errors.OutOfRangeError:
                break
            loss_val, preds_val = sess.run(
                [cost, preds_op],
                feed_dict={input_ph: bx, labels_ph: by, is_training: False})

            correct  = (preds_val == by)
            v_loss    += loss_val
            v_correct += correct.sum()
            v_total   += len(by)

            m_az, m_el, m_az_fb = calc_mae(preds_val, by)
            v_mae_az    += m_az
            v_mae_el    += m_el
            v_mae_az_fb += m_az_fb

            m0, m1 = (bc == 0), (bc == 1)
            v_correct0 += correct[m0].sum();  v_total0 += m0.sum()
            v_correct1 += correct[m1].sum();  v_total1 += m1.sum()
            if m0.any():
                az0, el0, azfb0 = calc_mae(preds_val[m0], by[m0])
                v_mae_az0 += az0; v_mae_el0 += el0; v_mae_az_fb0 += azfb0
            if m1.any():
                az1, el1, azfb1 = calc_mae(preds_val[m1], by[m1])
                v_mae_az1 += az1; v_mae_el1 += el1; v_mae_az_fb1 += azfb1

        v_acc  = v_correct  / v_total  if v_total  else 0
        v_acc0 = v_correct0 / v_total0 if v_total0 else 0
        v_acc1 = v_correct1 / v_total1 if v_total1 else 0
        v_avg_loss       = v_loss / val_batches if val_batches else 0
        v_avg_mae_az     = v_mae_az     / v_total  if v_total  else 0
        v_avg_mae_el     = v_mae_el     / v_total  if v_total  else 0
        v_avg_mae_az_fb  = v_mae_az_fb  / v_total  if v_total  else 0
        v_avg_mae_az0    = v_mae_az0    / v_total0 if v_total0 else float('nan')
        v_avg_mae_el0    = v_mae_el0    / v_total0 if v_total0 else float('nan')
        v_avg_mae_az_fb0 = v_mae_az_fb0 / v_total0 if v_total0 else float('nan')
        v_avg_mae_az1    = v_mae_az1    / v_total1 if v_total1 else float('nan')
        v_avg_mae_el1    = v_mae_el1    / v_total1 if v_total1 else float('nan')
        v_avg_mae_az_fb1 = v_mae_az_fb1 / v_total1 if v_total1 else float('nan')

        print_epoch_table('VAL  ', epoch_str, [
            dict(name='all',   loss=v_avg_loss, acc=v_acc,  n_ok=v_correct,  n=v_total,
                 mae_az=v_avg_mae_az,  mae_azfb=v_avg_mae_az_fb,  mae_el=v_avg_mae_el),
            dict(name='0-clk', loss=None,       acc=v_acc0, n_ok=v_correct0, n=v_total0,
                 mae_az=v_avg_mae_az0, mae_azfb=v_avg_mae_az_fb0, mae_el=v_avg_mae_el0),
            dict(name='1-clk', loss=None,       acc=v_acc1, n_ok=v_correct1, n=v_total1,
                 mae_az=v_avg_mae_az1, mae_azfb=v_avg_mae_az_fb1, mae_el=v_avg_mae_el1),
        ])

        summ = sess.run(summary_op, feed_dict={
            loss_ph: v_avg_loss, acc_ph: v_acc, acc0_ph: v_acc0, acc1_ph: v_acc1,
            mae_az_ph: v_avg_mae_az, mae_el_ph: v_avg_mae_el,
            mae_az_fb_ph: v_avg_mae_az_fb,
            mae_az_0ck_ph: v_avg_mae_az0, mae_el_0ck_ph: v_avg_mae_el0,
            mae_az_fb_0ck_ph: v_avg_mae_az_fb0,
            mae_az_1ck_ph: v_avg_mae_az1, mae_el_1ck_ph: v_avg_mae_el1,
            mae_az_fb_1ck_ph: v_avg_mae_az_fb1})
        val_writer.add_summary(summ, epoch + 1)

        # ── Forgetting eval ───────────────────────────────
        if use_orig_eval:
            sess.run(orig_iter.initializer)
            o_correct = o_total = 0
            o_mae_az = o_mae_el = o_mae_az_fb = 0.0
            for _ in range(orig_batches):
                try:
                    bx, by, _ = sess.run(orig_next)
                except tf.errors.OutOfRangeError:
                    break
                preds_val = sess.run(
                    preds_op, feed_dict={input_ph: bx, labels_ph: by, is_training: False})
                o_correct += (preds_val == by).sum()
                o_total   += len(by)
                m_az, m_el, m_az_fb = calc_mae(preds_val, by)
                o_mae_az += m_az; o_mae_el += m_el; o_mae_az_fb += m_az_fb
            o_acc           = o_correct / o_total if o_total else 0.0
            o_avg_mae_az    = o_mae_az    / o_total if o_total else float('nan')
            o_avg_mae_el    = o_mae_el    / o_total if o_total else float('nan')
            o_avg_mae_az_fb = o_mae_az_fb / o_total if o_total else float('nan')
            print_epoch_table('ORIG ', epoch_str, [
                dict(name='all', loss=None, acc=o_acc, n_ok=o_correct, n=o_total,
                     mae_az=o_avg_mae_az, mae_azfb=o_avg_mae_az_fb, mae_el=o_avg_mae_el),
            ])
            summ = sess.run(orig_summary_op, feed_dict={
                orig_acc_ph: o_acc, orig_mae_az_ph: o_avg_mae_az,
                orig_mae_el_ph: o_avg_mae_el, orig_mae_az_fb_ph: o_avg_mae_az_fb})
            orig_writer.add_summary(summ, epoch + 1)
            orig_writer.flush()

        # ── Save checkpoint ───────────────────────────────
        ckpt_path = saver.save(
            sess, os.path.join(args.output_dir, 'model.ckpt'), global_step=epoch + 1)
        print(f'  Checkpoint saved: {ckpt_path}')

        if args.save_best_val and v_acc0 > best_val_acc0:
            best_val_acc0 = v_acc0
            best_path = saver.save(
                sess, os.path.join(args.output_dir, 'best_val.ckpt'))
            print(f'  ★ New best val acc_0click={v_acc0:.4f} → {best_path}')
        print()

    train_writer.close()
    val_writer.close()
    orig_writer.close()
    print('Done!  Run:  tensorboard --logdir logs/')
    print(f'Final checkpoint in: {args.output_dir}')
    print('Convert with:  python convert_to_tflite.py')


if __name__ == '__main__':
    main()
