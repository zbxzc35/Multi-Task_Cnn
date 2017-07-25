import os
import sys
import base64
import gzip
import struct
import png
import cStringIO
from  cStringIO import StringIO
import argparse
import tensorflow as tf
import numpy as np
sys.path.append('.')
import tfcrc

parser = argparse.ArgumentParser()
parser.add_argument("--conf", help = '')
parser.add_argument("--gzip_in", action = 'store_true')
args = parser.parse_args()

def write_record(content, ofst):
    content_len = len(content)
    len_byte = struct.pack("Q", content_len)
    len_mask = tfcrc.calc_mask_crc(len_byte) & 0xffffffff
    len_mask_byte = struct.pack("I", len_mask)

    content_mask = tfcrc.calc_mask_crc(content) & 0xffffffff
    content_mask_byte = struct.pack("I", content_mask)

    ofst.write(len_byte)
    ofst.write(len_mask_byte)
    ofst.write(content)
    ofst.write(content_mask_byte)

def _bytes_feature(data):
    if data.__class__ == list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=data))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

def _float_feature(data):
    if data.__class__ == list:
        return tf.train.Feature(float_list=tf.train.FloatList(value=data))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))

def _int64_feature(data):
    if data.__class__ == list:
        return tf.train.Feature(int64_list = tf.train.Int64List(value = data))
    else:
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [data]))

def convert(input):
    if isinstance(input, dict):
        return dict((convert(key), convert(value)) 
                     for key, value in input.iteritems())
    elif isinstance(input, list):
        return [convert(element) for element in input]

    elif isinstance(input, unicode):
        return input.encode('utf-8')

    else:
        return input

HEIGHT = 0
WIDTH = 922 + 95 - 1
PLACEHOLDER = ''
uchar2idx = dict()
conf = {}
with open(args.conf, 'r') as ifst:
    jsn = json.loads(ifst.read())
    conf = jsn
    uchar2idx = jsn[u'char2idx']
    HEIGHT = max(uchar2idx.values())
    PLACEHOLDER = jsn[u'placeholder']

def string2img(ustr, h, w):
    png_writer = png.Writer(height = h, width = w, greyscale = True, bitdepth = 1)
    raw_image = np.int8(np.zeros([h, w]))
    for i in xrange(0, w):
        uchar = ustr[i]
        if uchar == PLACEHOLDER:
            pass;
        elif uchar in uchar2idx:
            idx = uchar2idx[uchar] - 1
            raw_image[idx][i] = 1
    oss = cStringIO.StringIO()
    png_writer.write(oss, raw_image)
    return oss.getvalue()

ii = 0
for line in sys.stdin:
    itemarr = line.strip().split('\t')
    if len(itemarr) != 2:
        continue
    feature, label = itemarr[0], itemarr[1]
    ft = {}
    ft['id'] = _bytes_feature(id)
    ft['lbl'] = _int64_feature((int)(label))
    idx = 0
    if args.gzip_in:
        assert(len(itemarr) == 3)
        zip_str = base64.b64decode(itemarr[2])
        inbuffer = StringIO(zip_str)
        gfst = gzip.GzipFile(mode="rb", fileobj=inbuffer)
        slot_str_arr = gfst.read().split('\t')
    else:
        slot_str_arr = itemarr[2:]
    for i in xrange(0, len(slot_str_arr)):
        [type_str, value] = slot_str_arr[i].split('\001')
        if type_str == 'string':
            uvalue = value.decode('utf8')
            ft['s%d' % idx] = _bytes_feature(string2img(uvalue, HEIGHT, len(uvalue)))
        elif type_str == 'int':
            intarr = [(int)(x) for x in value.split(conf['int_fender'])]
            ft['s%d' % idx] = _int64_feature(intarr)
        elif type_str == 'float':
            floatarr = [(float)(x) for x in value.split(conf['float_fender'])]
            ft['s%d' % idx] = _float_feature(floatarr)
        else:
            continue;
        idx += 1

    example = tf.train.Example(features = tf.train.Features(feature = ft))
    ss = example.SerializeToString()
    oss = cStringIO.StringIO()
    write_record(ss, oss)
    bstr = oss.getvalue()
    print '\t'.join([id, label, base64.b64encode(bstr)])
