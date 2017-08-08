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

NUM_CLASSES = 198
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

def encode_to_tfrecords_pair(data_paths, name, attrs_labels, attrs_hots):
    writer = tf.python_io.TFRecordWriter(name)
    num_img = 0
    max_n = 0  #number of images in list
    img_list = []
    label_list = []
    hots_list = []
    sku_list = []
    for data_path in data_paths:
        for line in file(data_path):
            case = line.strip().split('\t')
            if len(case) != 3:
                continue

            [skuid, img, attrs_all] = case
            
            label = [0]*NUM_CLASSES
            attrs_bool=''
            for attrs in attrs_all.split(','):
                if attrs in attrs_labels:
                    label[attrs_labels.index(attrs)] = 1
                    attrs_bool = attrs
            if attrs_bool == '':
                continue

            hots = attrs_hots[attrs_bool]
            
            img = base64.b64decode(img)
            file_like = cStringIO.StringIO(img)
            img = Image.open(file_like)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img.tobytes()
            
            if len(img_list)>max_n:
                max_n = len(img_list)
                print max_n

            if len(img_list)==0:
                img_list.append(img)
                label_list.append(label)
                hots_list.append(hots)
                sku_list.append(skuid)
            else:
                find_pair = False
                i = 0
                while ((not find_pair) and (i<len(img_list))):
                    diff_compare_i = np.sum(np.abs(np.array(label_list[i])-np.array(label)))
                    if (diff_compare_i<NUM_ATTR):
                        i += 1
                    else:
                        find_pair = True
                        num_img += 1
                        example = tf.train.Example(features=tf.train.Features(feature={
                                        'image1': _byte_feature(img_list[i]),
                                        'label1': _int64_feature(label_list[i]),
                                        'hots1': _int64_feature(hots_list[i]),
                                        'image2': _byte_feature(img),
                                        'label2': _int64_feature(label),
                                        'hots2': _int64_feature(hots)}))
                        writer.write(example.SerializeToString())
                        del img_list[i]
                        del label_list[i]
                        del hots_list[i]
                        del sku_list[i]
                        break
                if (not find_pair):
                    img_list.append(img)
                    label_list.append(label)
                    hots_list.append(hots)
                    sku_list.append(skuid)
    writer.close()
    print num_img


def encode_to_tfrecords(data_paths, name, attrs_labels, attrs_hots):
    writer = tf.python_io.TFRecordWriter(name)
    num_img = 0
    for data_path in data_paths:
        for line in file(data_path):
            case = line.strip().split('\t')
            if len(case) != 3:
                continue

            [skuid, img, attrs_all] = case
            
            label = [0]*NUM_CLASSES
            attrs_bool=''
            for attrs in attrs_all.split(','):
                if attrs in attrs_labels:
                    label[attrs_labels.index(attrs)] = 1
                    attrs_bool = attrs
            if attrs_bool == '':
                continue

            num_img += 1
            hots = attrs_hots[attrs_bool]
            
            img = base64.b64decode(img)
            file_like = cStringIO.StringIO(img)
            img = Image.open(file_like)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_raw = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _byte_feature(img_raw),
                'label': _int64_feature(label),
                'hots': _int64_feature(hots)}))

            writer.write(example.SerializeToString())
    writer.close()
    print num_img

def decode_from_tfrecord_pair(filename, batch_size=128, shuffle_batch=True):
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

def decode_from_tfrecord(filename, batch_size=5,shuffle_batch=True):
    filequeuelist = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()  # 文件读取

    _, example = reader.read(filequeuelist)
    features = tf.parse_single_example(example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
                                           'hots': tf.FixedLenFeature([NUM_CLASSES], tf.int64)
                                       })

    images = tf.decode_raw(features['image_raw'], tf.uint8)
    images = tf.reshape(images, [IMAGE_SIZE, IMAGE_SIZE, 3])
    images = tf.cast(images, tf.float32) * (1. / 255) - 0.5
    #images = tf.cast(images, tf.float32)

    labels = tf.cast(features['label'], tf.int32)
    hots = tf.cast(features['hots'], tf.int32)

    if shuffle_batch:
        images_batch, labels_batch, hots_batch = tf.train.shuffle_batch([images, labels, hots],
                                                            batch_size=batch_size,
                                                            capacity=3000,
                                                            min_after_dequeue=2000)
    else:
        images_batch, labels_batch, hots_batch = tf.train.batch([images, labels, hots],
                                                    batch_size=batch_size,
                                                    capacity=2000)

    # Display the training images in the visualizer.
    tf.summary.image('images', images_batch)

    return images_batch, labels_batch, hots_batch

def obtain_attrs(meta_path):
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
    train_input_file = [os.path.join(abspath, 'data/part-0000%d' % i) for i in xrange(10)]
    train_input_file = train_input_file + [os.path.join(abspath, 'data/part-000%d' % i) for i in xrange(10, 90)]

    test_output_file = os.path.join(abspath, 'data/test.bin')
    test_input_file = [os.path.join(abspath, 'data/part-000%d' % i) for i in xrange(90, 100)]
    meta_path = os.path.join(abspath, 'data/batches.meta')

#    NUM_CLASSES, attrs_labels, attrs_hots = generate_meta(input_file, meta_path)
#    print NUM_CLASSES
    NUM_CLASSES, attrs_labels, attrs_hots = obtain_attrs(meta_path)

    encode_to_tfrecords_pair(train_input_file, train_output_file, attrs_labels, attrs_hots)
    encode_to_tfrecords(test_input_file, test_output_file, attrs_labels, attrs_hots)

