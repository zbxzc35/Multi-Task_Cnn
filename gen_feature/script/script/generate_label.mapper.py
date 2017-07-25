#-*-coding:utf8-*-
import os
import sys
import argparse
os.environ['PYTHON_EGG_CACHE'] = './'
reload(sys)
sys.setdefaultencoding("utf-8")

parser = argparse.ArgumentParser()
parser.add_argument('--category_attribute', help = "category_attribute")
args = parser.parse_args()

cate2attr = {}

for line in file(args.category_attribute):
    fields = line.strip().split('\t')
    if len(fields) != 3:
        continue
    [cate, attr, val] = fields
    cate2attr.setdefault('{}|{}'.format(cate, attr), val)

labels=[]
for line in sys.stdin:
    result = []

    fields = line.rstrip('\n').split('\t')
    if len(fields) != 4:
        continue

    [skuid, cate, image, attr] = fields
    attrs = attr.split(',')
    for it in attrs:
        key = '{}|{}'.format(cate, it.split(':')[0])
        if key in cate2attr:
          if it.split(':')[1] in cate2attr[key]:
            try:
                label = '{}|{}|{}'.format(cate, it.split(':')[0], it.split(':')[1]) 
                if label not in labels:
                    labels.append(label)
                    print label
            except:
                pass


