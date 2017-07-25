#-*-coding:utf8-*-
import os
import sys
import argparse
os.environ['PYTHON_EGG_CACHE'] = './'
reload(sys)
sys.setdefaultencoding("utf-8")

parser = argparse.ArgumentParser()
parser.add_argument('--category_attribute', help = "category_attribute")
parser.add_argument('--select_attr',help = "selected attributes")
args = parser.parse_args()

cate2attr = {}

for line in file(args.category_attribute):
    fields = line.strip().split('\t')
    if len(fields) != 3:
        continue
    [cid, attr, val] = fields
    cate2attr.setdefault('{}|{}'.format(cid, attr), val)

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
    
    [skuid, cid, image, attr_all] = fields
    attrs_all = attr_all.split(',')
    sel_attrs=[]
    select = False
    for it in attrs_all:
        key = '{}|{}'.format(cid, it.split(':')[0])
        if (it.split(':')[0] in select_attrs) and (key in cate2attr):
            if it.split(':')[1] in cate2attr[key]:
                try:
                    select = True
                    attrs = '{}|{}|{}'.format(cid, it.split(':')[0], it.split(':')[1]) 
                    sel_attrs.append(attrs)
                except:
                    pass
    if select:
        print '{}\t{}\t{}'.format(skuid, image, ','.join(sel_attrs))
