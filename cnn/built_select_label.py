#-*-coding:utf8-*-
import os
import sys
import numpy as np

abspath = os.path.abspath('..')
meta_path = os.path.join(abspath, 'data/batches.meta')
num_splits_path = os.path.join(abspath, 'data/num_splits')

select_attr=[]
with open('select_attr','r') as f:
    for attr in f.readlines():
        select_attr.append(attr.strip())

labels={}        
with open('all_labels','r') as f:
    for line in f.readlines():
        field = line.strip().split('|')
        if len(field) != 3:
            continue
        [cid, attr, attrv] = field
        
        if cid not in labels:
            labels[cid] = {attr:[attrv]}
        elif attr not in labels[cid]:
            labels[cid][attr] = [attrv]
        elif attrv not in labels[cid][attr]:
            labels[cid][attr].append(attrv)

attrs_labels=[]
num_splits=[0]
NUM_CLASSES=0
for attr in select_attr:
    for (cid, value) in labels.items():
        if attr in value:
            for attrv in value[attr]:
                attrs_labels.append('{}|{}|{}'.format(cid,attr,attrv))
                NUM_CLASSES += 1
    num_splits.append(NUM_CLASSES)
num_splits = np.diff(num_splits)       

attrs_hots = {}
for attrs in attrs_labels:
    hots = [0]*NUM_CLASSES
    [cid, attr, attrv] = attrs.split('|')
    for i in xrange(NUM_CLASSES):
        if cid in attrs_labels[i]:
            hots[i] = 1
    attrs_hots[attrs] = hots

with open(meta_path, 'wb') as f:
    for index, value in enumerate(attrs_labels):
        hots = [str(x) for x in attrs_hots[value]]
        f.write('{}\t{}\t{}\n'.format(index, value, '\001'.join(hots)))

with open(num_splits_path,'wb') as f:
    for sp in num_splits:
        f.write('{}\n'.format(sp))

print NUM_CLASSES
