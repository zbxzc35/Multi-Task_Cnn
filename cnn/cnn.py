from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import numpy as np
import gen_tfrecord
import os
import re
import sys
import tarfile

IMAGE_CLASSES = gen_tfrecord.NUM_CLASSES
IMAGE_SIZE = gen_tfrecord.IMAGE_SIZE

NUM_CLASSES = gen_tfrecord.NUM_CLASSES
IMAGE_SIZE = gen_tfrecord.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 476466
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('num_splits_dir', '/home/admin/zhexuanxu/multi-task_cnn/data/num_splits',
                            """Directory where to obtain number of attributes""")



# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def inputs(eval_data=False, batch_size=128):
    """Construct distorted input for muliti-task_cnn training using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    abspath = os.path.abspath('..')
    train_input_file = os.path.join(abspath, 'data/data_train.bin')
    eval_input_file = os.path.join(abspath, 'data/test.bin')
    if not eval_data:
        images, labels, neg_labels, hots = gen_tfrecord.decode_from_tfrecord(train_input_file, batch_size)
    else:
        images, labels, neg_labels, hots = gen_tfrecord.decode_from_tfrecord(eval_input_file, batch_size)

    return images, labels, neg_labels, hots

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        #dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd=None):
    """Helper to create an initialized Variable with weight decay.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    #dtype = tf.float32
    # var = _variable_on_cpu(
    #     name,
    #     shape,
    #     tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    var = _variable_on_cpu(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _conv(name, in_, ksize, strides=[1, 1, 1, 1], padding='SAME', group=1):
    n_kern = ksize[3]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)

    with tf.variable_scope(name) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split(in_, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv)

    print name, conv.get_shape().as_list()
    return conv

def _maxpool(name, in_, ksize, strides, padding='SAME'):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                           padding=padding, name=name)

    return pool

def _lrn(name, in_, depth_radius, bias, alpha, beta):
    norm = tf.nn.lrn(in_, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta,
                      name=name)

    return norm

def _fc(name, in_, outsize, dropout=1.0):
    with tf.variable_scope(name) as scope:
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)
        _activation_summary(fc)

    print name, fc.get_shape().as_list()
    return fc

def _view_pool(name, view_features):
    vf = view_features[0]
    for v in view_features[1:]:
        vf = tf.concat([vf, v], 1, name=name)

    print name, vf.get_shape().as_list()
    return vf

def inference(images, n_cnn):
    """Build the Multi-task_cnn model.
     Args:
       images: Images returned from decode_from_tfrecord()
     Returns:
       Logits.
     """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    view_pool = []
    for i in xrange(n_cnn):
        with tf.variable_scope('CNN%d' % i):
            conv1 = _conv('conv1', images, ksize=[11, 11, 3, 96], strides=[1, 4, 4, 1], padding='SAME')
            pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            #lrn1 = _lrn('lrn1', pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            conv2 = _conv('conv2', pool1, ksize=[5, 5, 96, 256], strides=[1, 1, 1 ,1], padding='SAME', group=2)
            #lrn2 = _lrn('lrn2', conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            conv3 = _conv('conv3', pool2, ksize=[3, 3, 256, 384])
            conv4 = _conv('conv4', conv3, ksize=[3, 3, 384, 384], group=2)
            conv5 = _conv('conv5', conv4, ksize=[3, 3, 384, 256], group=2)
            pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            dim2 = np.prod(pool5.get_shape().as_list()[1:])
            reshape = tf.reshape(pool5, [-1, dim2])
            view_pool.append(reshape)

    concat5 = _view_pool('concat5',view_pool)
    fc6 = _fc('fc6', concat5, 4096, dropout=0.6)
    fc7 = _fc('fc7', fc6, 4096, dropout=0.6)
    fc8 = _fc('fc8', fc7, NUM_CLASSES)

    return fc8

def obtain_splits(filepath):
    num_splits = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            num_splits.append(int(line.strip()))
    return num_splits

def loss(logits, labels, neg_labels, hots, loss_type=1):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, NUM_CLASSES]
      hots: hots from inputs(), shape: [batch_size, NUM_CLASSES]
      loss_type: 1: softmax_cross_entropy; 2: hingeloss
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    hots = tf.cast(hots, dtype)
    logits = tf.multiply(logits, hots, name='assign_label')
    logits = tf.nn.softmax(logits)
    
    if loss_type == 1:
        num_splits = tf.constant(obtain_splits(FLAGS.num_splits_dir))
        logits_split = tf.split(logits, num_splits, 1)
        labels_split = tf.split(labels, num_splits, 1)
        for i in xrange(len(labels_split)):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels_split[i], logits=logits_split[i]
                        )
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_{}'.format(i))
            tf.add_to_collection('losses', cross_entropy_mean)
    elif loss_type == 2:
        # We first need to convert binary labels to -1/1 labels (as floats).
        labels = tf.cast(labels, tf.float32)
        all_ones = array_ops.ones_like(labels)
        labels = math_ops.subtract(2 * labels, all_ones)
        
        logits = tf.nn.softmax(logits)
        hinge_loss = tf.nn.relu(math_ops.subtract(all_ones, math_ops.multiply(labels, logits))) ** 2 / 2.0
        hinge_loss_sum = tf.reduce_sum(hinge_loss, name='hinge_loss')
        tf.add_to_collection('losses', hinge_loss_sum)
    elif loss_type == 3:
        num_splits = tf.constant(obtain_splits(FLAGS.num_splits_dir))
        logits_split = tf.split(logits, num_splits, 1)
        labels_split = tf.split(labels, num_splits, 1)
        neg_labels_split = tf.split(neg_labels, num_splits, 1)

        for i in xrange(len(labels_split)):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels_split[i], logits=logits_split[i]
                        )
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_{}'.format(i))
            
            cross_entropy_neg = tf.nn.softmax_cross_entropy_with_logits(
                        labels=neg_labels_split[i], logits=logits_split[i]
                        )
            cross_entropy_mean_neg = tf.reduce_mean(cross_entropy_neg, name='cross_entropy_neg_{}'.format(i))
            
            neg_loss = tf.add(10.0, tf.subtract(cross_entropy_mean, cross_entropy_mean_neg), name='neg_loss')
            tf.add_to_collection('losses', neg_loss)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

    
def calcuate_prediction(logits, labels, num_splits):
    logits_split = tf.split(logits, num_splits, 1)
    
    labels = tf.cast(labels, tf.float32)
    labels_split = tf.split(labels, num_splits, 1)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits_split[0], 1), tf.argmax(labels_split[0], 1)), tf.float32)
    correct_prediction = tf.multiply(correct_prediction, tf.reduce_sum(labels_split[0], 1))
    accuracy = tf.div(tf.reduce_sum(correct_prediction), tf.reduce_sum(labels_split[0]))
    pred = tf.expand_dims(accuracy, 0)
    for i in range(1, len(labels_split)):
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits_split[i], 1), tf.argmax(labels_split[i], 1)), tf.float32)
        correct_prediction = tf.multiply(correct_prediction, tf.reduce_sum(labels_split[i], 1))
        accuracy = tf.div(tf.reduce_sum(correct_prediction), tf.reduce_sum(labels_split[i]))
        pred = tf.concat([tf.reshape(pred, [-1]), tf.expand_dims(accuracy, 0)], axis=0)
    
    tf.summary.histogram("prediction", pred) 

    return pred   


def getloss():
    return tf.get_collection('losses')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in mulit-task_cnn model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

def train(total_loss, global_step):
    """Train multi-task_cnn model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

