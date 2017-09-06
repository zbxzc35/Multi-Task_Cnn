#-*-coding:utf8-*-
import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image
import cStringIO
import base64
import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')

NUM_CLASSES = 5771
IMAGE_SIZE = 100
NUM_ATTR = 5

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

def _int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_to_tfrecords_pair(data_paths, output_path, attrs_labels, attrs_hots):
    """把pair_data 转成 tfrecord
    Args:
        data_path: pair_data数据的地址集合
            pair_data: cid3, sku1_id, img1, attrs1, sku2_id, img2, attrs2
            attrs1, attrs2: 如 闭合方式_套脚|风格_英伦|颜色_黑色|功能_透气
        output_path: 转成的tfrecord地址
        attrs_labels:  由label组成的list, attrs_labels[label_id] = label
        attrs_hots: 一个dict, attrs_hots[label] = cid_mask
    """
    writer = tf.python_io.TFRecordWriter(output_path)
    num_img = 0
    for data_path in data_paths:
        for line in file(data_path):
            case = line.strip().split('\t')
            if len(case) != 7:
                continue

            [cid3, sku1, img1, attrs1_all, sku2, img2, attrs2_all] = case
            
            label1 = [0]*NUM_CLASSES
            attrs_bool=''
            for attrs in attrs1_all.split('|'):
                [attr, attrv] = attrs.split('_')
                classes = '{}|{}|{}'.format(cid3, attr, attrv)
                if classes in attrs_labels:
                    label1[attrs_labels.index(classes)] = 1
                    attrs_bool = classes
            if attrs_bool == '':
                continue

            hots1 = attrs_hots[attrs_bool]
            
            img1 = base64.b64decode(img1)
            file_like1 = cStringIO.StringIO(img1)
            img1 = Image.open(file_like1)
            img1 = img1.resize((IMAGE_SIZE, IMAGE_SIZE))
            img1 = img1.tobytes()
            
            label2 = [0]*NUM_CLASSES
            attrs_bool=''
            for attrs in attrs2_all.split('|'):
                [attr, attrv] = attrs.split('_')
                classes = '{}|{}|{}'.format(cid3, attr, attrv)
                if classes in attrs_labels:
                    label2[attrs_labels.index(classes)] = 1
                    attrs_bool = classes
            if attrs_bool == '':
                continue

            hots2 = attrs_hots[attrs_bool]
            
            img2 = base64.b64decode(img2)
            file_like2 = cStringIO.StringIO(img2)
            img2 = Image.open(file_like2)
            img2 = img2.resize((IMAGE_SIZE, IMAGE_SIZE))
            img2 = img2.tobytes()

            num_img += 1

            example = tf.train.Example(features=tf.train.Features(feature={
                                        'image1': _byte_feature(img1),
                                        'label1': _int64_feature(label1),
                                        'hots1': _int64_feature(hots1),
                                        'image2': _byte_feature(img2),
                                        'label2': _int64_feature(label2),
                                        'hots2': _int64_feature(hots2)}))
            writer.write(example.SerializeToString())
    writer.close()
    print num_img


def encode_to_tfrecords(data_paths, output_path, attrs_labels, attrs_hots):
    """把pair_data 转成 tfrecord
    Args:
        data_path: test_data数据的地址集合
            test_data: cid3, sku_id, img, attrs
            attrs: 如 闭合方式_套脚|风格_英伦|颜色_黑色|功能_透气
        output_path: 转成的tfrecord地址
        attrs_labels:  由label组成的list, attrs_labels[label_id] = label
        attrs_hots: 一个dict, attrs_hots[label] = cid_mask
    """
    writer = tf.python_io.TFRecordWriter(output_path)
    num_img = 0
    for data_path in data_paths:
        for line in file(data_path):
            case = line.strip().split('\t')
            if len(case) != 4:
                continue

            [cid3, skuid, img, attrs_all] = case
            
            label = [0]*NUM_CLASSES
            attrs_bool=''

            attrs_all_split = attrs_all.split('|')
            if len(attrs_all_split)<2:
                continue

            for attrs in attrs_all_split:
                [attr, attrv] = attrs.split('_')
                classes = '{}|{}|{}'.format(cid3, attr, attrv)
                if classes in attrs_labels:
                    label[attrs_labels.index(classes)] = 1
                    attrs_bool = classes
            if attrs_bool == '':
                continue

            num_img += 1
            if num_img>50000:
                break

            hots = attrs_hots[attrs_bool]
           
            img = base64.b64decode(img)
            file_like = cStringIO.StringIO(img)
            img = Image.open(file_like)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_raw = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _byte_feature(img_raw),
                'skuid': _byte_feature(skuid),
                'label': _int64_feature(label),
                'hots': _int64_feature(hots)}))

            writer.write(example.SerializeToString())
    writer.close()
    print num_img

def decode_from_tfrecord_pair(filename, batch_size=128, shuffle_batch=True):
    """读取paird 的 tfrecord数据
    """
    filequeuelist = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()  # 文件读取

    _, example = reader.read(filequeuelist)
    features = tf.parse_single_example(example,
                                       features={
                                           'image1': tf.FixedLenFeature([], tf.string),
                                           'label1': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
                                           'hots1': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
                                           'image2': tf.FixedLenFeature([], tf.string),
                                           'label2': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
                                           'hots2': tf.FixedLenFeature([NUM_CLASSES], tf.int64)
                                       })

    images1 = tf.decode_raw(features['image1'], tf.uint8)
    images1 = tf.reshape(images1, [IMAGE_SIZE, IMAGE_SIZE, 3])
    images1 = tf.cast(images1, tf.float32) * (1. / 255) - 0.5
    labels1 = tf.cast(features['label1'], tf.int32)
    hots1 = tf.cast(features['hots1'], tf.int32)

    images2 = tf.decode_raw(features['image2'], tf.uint8)
    images2 = tf.reshape(images2, [IMAGE_SIZE, IMAGE_SIZE, 3])
    images2 = tf.cast(images2, tf.float32) * (1. / 255) - 0.5
    labels2 = tf.cast(features['label2'], tf.int32)
    hots2 = tf.cast(features['hots2'], tf.int32)

    if shuffle_batch:
        images1_batch, labels1_batch, hots1_batch, images2_batch, labels2_batch, hots2_batch = tf.train.shuffle_batch([images1, labels1, hots1, images2, labels2, hots2],
                                                            batch_size=batch_size,
                                                            capacity=3000,
                                                            min_after_dequeue=2000)
    else:
        images1_batch, labels1_batch, hots1_batch, images2_batch, labels2_batch, hots2_batch = tf.train.batch([images1, labels1, hots1, images2, labels2, hots2],
                                                    batch_size=batch_size,
                                                    capacity=2000)

    # Display the training images in the visualizer.
    tf.summary.image('images1', images1_batch)
    tf.summary.image('images2', images2_batch)

    return images1_batch, labels1_batch, hots1_batch, images2_batch, labels2_batch, hots2_batch

def decode_from_tfrecord(filename, batch_size=128,shuffle_batch=True):
    """读取paird 的 tfrecord数据
    """
    filequeuelist = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()  # 文件读取

    _, example = reader.read(filequeuelist)
    features = tf.parse_single_example(example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'skuid': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
                                           'hots': tf.FixedLenFeature([NUM_CLASSES], tf.int64)
                                       })

    images = tf.decode_raw(features['image_raw'], tf.uint8)
    images = tf.reshape(images, [IMAGE_SIZE, IMAGE_SIZE, 3])
    images = tf.cast(images, tf.float32) * (1. / 255) - 0.5
    #images = tf.cast(images, tf.float32)
    
    skuid = features['skuid']
    labels = tf.cast(features['label'], tf.int32)
    hots = tf.cast(features['hots'], tf.int32)

    if shuffle_batch:
        images_batch, skuid_batch, labels_batch, hots_batch = tf.train.shuffle_batch([images, skuid, labels, hots],
                                                            batch_size=batch_size,
                                                            capacity=3000,
                                                            min_after_dequeue=2000)
    else:
        images_batch, skuid_batch, labels_batch, hots_batch = tf.train.batch([images, skuid, labels, hots],
                                                    batch_size=batch_size,
                                                    capacity=2000)

    # Display the training images in the visualizer.
    tf.summary.image('images', images_batch)

    return images_batch, skuid_batch, labels_batch, hots_batch

def obtain_attrs(meta_path):
    """ 从batches.meta 中获取label 信息
    Args:
        meta_path: batches.meta 地址
        batches.meta: label_id, label, cid_mask 
        eg: 0 1348|颜色|灰色 cid_mask
    Return:
        NUM_CLASSES: 总的label数
        attrs_labels:  由label组成的list, attrs_labels[label_id] = label
        attrs_hots: 一个dict, attrs_hots[label] = cid_mask
    """
    with open(meta_path, 'r') as mf:
        filelist = mf.readlines()
        NUM_CLASSES = int(filelist[-1].strip().split('\t')[0])+1

        attrs_labels = range(NUM_CLASSES)
        attrs_hots = {}
        for line in filelist:
            [label, attr, hots] = line.strip().split('\t')
            attrs_labels[int(label)] = attr
            hots = [int(x) for x in hots.split('\001')]
            attrs_hots[attr] = hots

    return NUM_CLASSES, attrs_labels, attrs_hots



if __name__ == '__main__':
#    input_file = sys.argv[1]
#    output_file = sys.argv[2]
    abspath = os.path.abspath('..')
    train_output_file = os.path.join(abspath, 'data/data_train.bin')
    train_input_file = [os.path.join(abspath, 'data/train_data/part-r-0000%d' % i) for i in xrange(10)]
    train_input_file = train_input_file + [os.path.join(abspath, 'data/train_data/part-r-000%d' % i) for i in xrange(10, 100)]

    test_output_file = os.path.join(abspath, 'data/test.bin')
    test_input_file = [os.path.join(abspath, 'data/test_data/part-0000%d' % i) for i in xrange(1)]
    meta_path = os.path.join(abspath, 'data/batches.meta')

    NUM_CLASSES, attrs_labels, attrs_hots = obtain_attrs(meta_path)

    encode_to_tfrecords_pair(train_input_file, train_output_file, attrs_labels, attrs_hots)
#    encode_to_tfrecords(test_input_file, test_output_file, attrs_labels, attrs_hots)

