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

def encode_to_tfrecords(data_file, output_file, attrs_labels, attrs_hots):
    writer = tf.python_io.TFRecordWriter(output_file)
    num_img = 0
    for line in file(data_file):
        case = line.strip().split('\t')
        if len(case) != 4:
            continue

        [cid3, skuid, img, attrs_all] = case
            
        label = [0]*NUM_CLASSES
        attrs_bool=''

        attrs_all_split = attrs_all.split('|')
       # if len(attrs_all_split)<2:
       #     continue

        for attrs in attrs_all_split:
            [attr, attrv] = attrs.split('_')
            classes = '{}|{}|{}'.format(cid3, attr, attrv)
            if classes in attrs_labels:
                label[attrs_labels.index(classes)] = 1
                attrs_bool = classes
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
                'skuid': _byte_feature(skuid),
                'label': _int64_feature(label),
                'hots': _int64_feature(hots)}))

        writer.write(example.SerializeToString())
    writer.close()
    print num_img

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
    test_input_file = os.path.abspath(sys.argv[1])
    test_output_file = os.path.abspath(sys.argv[2])
    #print test_input_file
    #print test_output_file
    abspath = os.path.abspath('..')
    meta_path = os.path.join(abspath, 'data/batches.meta')
    NUM_CLASSES, attrs_labels, attrs_hots = obtain_attrs(meta_path)

    encode_to_tfrecords(test_input_file, test_output_file, attrs_labels, attrs_hots)

