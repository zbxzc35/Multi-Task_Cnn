#-*-coding:utf8-*-
import os
import sys
import argparse
os.environ['PYTHON_EGG_CACHE'] = './'
reload(sys)
sys.setdefaultencoding("utf-8")

parser = argparse.ArgumentParser()
parser.add_argument('--category_attribute', help = "category_attribute")
parser.add_argument('--lable_mapping', help = "lable_mapping")
parser.add_argument('--select_attr',help = "selected attributes")
args = parser.parse_args()

cate2attr = {}

for line in file(args.category_attribute):
    fields = line.strip().split('\t')
    if len(fields) != 3:
        continue
    [cate, attr, val] = fields
    cate2attr.setdefault('{}|{}'.format(cate, attr), val)

label2int = {}

for line in file(args.lable_mapping):
    fields = line.strip().split('\t')
    if len(fields) != 2:
        continue
    [label, int64] = fields
    label2int.setdefault(label, int64)

select_attrs = {}

for line in file(args.select_attr):
    attr = line.strip()
    if not attr in select_attrs:
        select_attrs[attr]=[]

for line in sys.stdin:
    result = []

    fields = line.rstrip('\n').split('\t')
    if len(fields) != 4:
        continue

    [skuid, cate, image, attr] = fields
    attrs = attr.split(',')
    for it in attrs:
        key = '{}|{}'.format(cate, it.split(':')[0])
        if (it.split(':')[0] in select_attrs) and (key in cate2attr):
          if it.split(':')[1] in cate2attr[key]:
            try:
                label = '{}|{}|{}'.format(cate, it.split(':')[0], it.split(':')[1]) 
                print '{}\t{}\t{}'.format(label,label2int[label], image)
            except:
                pass
