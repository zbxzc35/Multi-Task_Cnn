from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf
import sys
import cnn
import gen_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/admin/zhexuanxu/multi-task_cnn_pairwise/tmp',
                          """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('test_batch_size', 512,
                           """Number of evaluation images to process in a batch""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/admin/zhexuanxu/multi-task_cnn_pairwise/tmp',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

abspath = os.path.abspath('..')
num_splits_path = os.path.join(abspath, 'data/num_splits')
meta_path = os.path.join(abspath, 'data/batches.meta')
NUM_CLASSES, attrs_labels, attrs_hots = gen_tfrecord.obtain_attrs(meta_path) 

def record_sku_pred(sku, multi_cnns, origin, num_splits, test_pred_file, mode):
    num_splits = num_splits.eval()
    
    with open(test_pred_file, mode) as f:
        n = len(sku)
        for i in range(n):
            #image = (images[i].reshape((1, 30000)) + 0.5) * 255
            #image = [str(x) for x in image[0]]
            #image = '\001'.join(image)
            
            sku_id = sku[i]
            cnn_pred = multi_cnns[i]
            origin_pred = origin[i]
            
            j=0; k=0; cid=''
            while (cid=='') and (j<len(cnn_pred)):
                if cnn_pred[j]!=0:
                    cid, attr, attrv = attrs_labels[k + cnn_pred[j]].split('|')
                else:
                    k+=num_splits[j]
                    j+=1
            if cid=='':
                continue
    
            cnn_attr=[]
            origin_attr=[]
            k=0
            for j in range(len(cnn_pred)):
                cid_j, attr, attrv = attrs_labels[k + origin_pred[j]].split('|')
                if (cid_j==cid):
                    origin_attr.append('{}|{}'.format(attr, attrv))
                
                cid_j, attr, attrv = attrs_labels[k + cnn_pred[j]].split('|')
                if (cid_j==cid):
                    cnn_attr.append('{}|{}'.format(attr, attrv))
                k += num_splits[j]

            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(sku_id, cid, 'Original', ','.join(origin_attr), 'Multi-CNNs_image', ','.join(cnn_attr)))

def evaluate(test_num, test_tfrecord_file, test_pred_file):
    """Eval Multi-task_cnn for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for Multi-task_cnn.
        # eval_data = FLAGS.eval_data == 'test'
        images, skuid, labels, hots = cnn.inputs(test_tfrecord_file, eval_data=True, batch_size=FLAGS.test_batch_size)
      
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cnn.inference(images, n_cnn=5)
        
        hots = tf.cast(hots, tf.float32)
        logits = tf.multiply(logits, hots, name='assign_label')

        num_splits = tf.constant(cnn.obtain_splits(num_splits_path))
        # Calculate predictions.
        cnn_pred = cnn.predict(logits, num_splits)
        origin_pred = cnn.predict(labels, num_splits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cnn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restore from file")
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
 
               
                num_iter = int(math.ceil(test_num / FLAGS.test_batch_size))
                # Compute precision @ 1.
                sku, cnn_predi, origin_predi = sess.run([skuid, cnn_pred, origin_pred])
                record_sku_pred(sku, cnn_predi, origin_predi, num_splits, test_pred_file, 'wb')
                
                step = 0
                while step < num_iter and not coord.should_stop():
                    sku, cnn_predi, origin_predi = sess.run([skuid, cnn_pred, origin_pred])
                    record_sku_pred(sku, cnn_predi, origin_predi, num_splits, test_pred_file, 'ab')
                    step += 1

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                # summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    test_num = int(sys.argv[1])
    test_tfrecord_file = os.path.abspath(sys.argv[2])
    test_pred_file = os.path.abspath(sys.argv[3])
    evaluate(test_num, test_tfrecord_file, test_pred_file)
