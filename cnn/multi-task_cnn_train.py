#-*-coding:utf8-*-
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import time
import numpy as np
import os 
from PIL import Image
import cnn
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/admin/zhexuanxu/multi-task_cnn/tmp',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 110,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


if __name__ == '__main__':

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
          images, labels, hots = cnn.inputs()

        # Build a Graph that computes the logits predictions from the inference model.
        logits = cnn.inference(images, n_cnn=3)

        # Calculate loss.
        loss = cnn.loss(logits, labels, hots)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cnn.train(loss, global_step)

        # 创建一个saver对象，用于保存参数到文件中
        saver = tf.train.Saver(tf.global_variables())

#        lst = cnn.getloss()

        # 返回所有summary对象先merge再serialize后的的字符串类型tensor
        summary_op = tf.summary.merge_all()

        # log_device_placement参数可以记录每一个操作使用的设备，这里的操作比较多，故设置为False
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        # 变量初始化
        init = tf.global_variables_initializer()
        sess.run(init)

        # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     # Restores from checkpoint
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     print ("restore from file")
        # else:
        #     print('No checkpoint file found')

           # 启动所有的queuerunners
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph=sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            # 用于验证当前迭代计算出的loss_value是否合理
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

#            lsst = sess.run(lst)

#            print len(lsst)
#            for loss_item in lsst:
#                print loss_item

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.log_frequency
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration/ FLAGS.log_frequency)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


        # class _LoggerHook(tf.train.SessionRunHook):
        #     """Logs loss and runtime."""
        #
        #     def begin(self):
        #         self._step = -1
        #         self._start_time = time.time()
        #
        #     def before_run(self, run_context):
        #         self._step += 1
        #         return tf.train.SessionRunArgs(loss)  # Asks for loss value.
        #
        #     def after_run(self, run_context, run_values):
        #         if self._step % FLAGS.log_frequency == 0:
        #             current_time = time.time()
        #             duration = current_time - self._start_time
        #             self._start_time = current_time
        #
        #             loss_value = run_values.results
        #             examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
        #             sec_per_batch = float(duration / FLAGS.log_frequency)
        #
        #             format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
        #                           'sec/batch)')
        #             print (format_str % (datetime.now(), self._step, loss_value,
        #                                  examples_per_sec, sec_per_batch))
        #
        # with tf.train.MonitoredTrainingSession(
        #     checkpoint_dir=FLAGS.train_dir,
        #     hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
        #            tf.train.NanTensorHook(loss),
        #            _LoggerHook()],
        #     config=tf.ConfigProto(
        #         log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        #     while not mon_sess.should_stop():
        #         mon_sess.run(train_op)
        #

#     images, labels, hots = cnn.inputs()
#    
#     init = tf.global_variables_initializer()
#    
#     with tf.Session() as sess:
#         sess.run(init)
#    
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#    
#         for i in range(1):
#             label, image, hot = sess.run([labels, images, hots])
#             for j in range(1):
#                 with open('testl','wb') as f:
#                     ll = [str(x) for x in label[j]]
#                     hh = [str(x) for x in hot[j]]
#                     f.write('{}\n'.format(ll))
#                     f.write('{}\n'.format(hh))
#                 #print label[j] 
#                 #print image[j]
#                 #print hot[j]
#         coord.request_stop()
#         coord.join(threads)
