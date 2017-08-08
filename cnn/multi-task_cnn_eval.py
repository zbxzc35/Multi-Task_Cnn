from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf

import cnn
import gen_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/admin/zhexuanxu/multi-task_cnn_pairwise/tmp',
                          """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('eval_batch_size', 256,
                           """Number of evaluation images to process in a batch""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/admin/zhexuanxu/multi-task_cnn_pairwise/tmp',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

abspath = os.path.abspath('..')
num_splits_path = os.path.join(abspath, 'data/num_splits')
meta_path = os.path.join(abspath, 'data/batches.meta')
NUM_CLASSES, attrs_labels, attrs_hots = gen_tfrecord.obtain_attrs(meta_path) 

def record_wrong_predict(images, logits, labels, num_splits):
    with open('wrong_pred', 'wb') as f:
        logits_split = tf.split(logits, num_splits, 1)
        labels_split = tf.split(labels, num_splits, 1)
        for i in range(logits.shape[0]):
            print i
            image = (images[i].reshape((1, 30000)) + 0.5) * 255
            image = [str(x) for x in image[0]]
            image = '\001'.join(image)
            k=0
            for j in range(len(labels_split)):
                logit_argmax = tf.argmax(logits_split[j][i]).eval()
                label_argmax = tf.argmax(labels_split[j][i]).eval()
                if (label_argmax!=0) and (logit_argmax != label_argmax):
                    f.write('{}\t{}\t{}\n'.format(image, attrs_labels[k+logit_argmax], attrs_labels[k+label_argmax]))
                k += num_splits[j]

def eval_once(saver, summary_writer, summary_op, pred, images, logits, labels, num_splits):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/mulit-task_cnn_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.floor(FLAGS.num_examples / FLAGS.eval_batch_size))
            total_sample_count = num_iter * FLAGS.eval_batch_size
            step = 0
            precisions, imag, logi, labe, nsplit = sess.run([pred, images, logits, labels, num_splits])
            precisions = [precisions]
            #record_wrong_predict(imag, logi, labe, nsplit)
            #print precisions
            while step < num_iter and not coord.should_stop():
                p, imag, logi, labe, nsplit = sess.run([pred, images, logits, labels, num_splits])
                #record_wrong_predict(imag, logi, labe, nsplit)
                precisions = np.concatenate([precisions, [p]], axis=0)
                step += 1

            precisions = np.mean(precisions, axis=0)
            # Compute precision @ 1.
            print('{}: precision @ 1 = {}'.format(datetime.now(), precisions))
            
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
           # summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval Multi-task_cnn for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for Multi-task_cnn.
        # eval_data = FLAGS.eval_data == 'test'
        images, labels, hots = cnn.inputs(eval_data=True, batch_size=FLAGS.eval_batch_size)
      
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cnn.inference(images, n_cnn=5)
        
        hots = tf.cast(hots, tf.float32)
        logits = tf.multiply(logits, hots, name='assign_label')

        num_splits = tf.constant(cnn.obtain_splits(num_splits_path))
        # Calculate predictions.
        pred = cnn.calcuate_prediction(logits, labels, num_splits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cnn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, summary_op, pred, images, logits, labels, num_splits)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    evaluate()
