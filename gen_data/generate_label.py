# -*- coding:utf-8 -*
import os
import sys
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')
"""
生成 batches.meta, num_splits, label_combine
"""
abspath = os.path.abspath('..')
meta_path = os.path.join(abspath, 'data/batches.meta')
num_splits_path = os.path.join(abspath, 'data/num_splits')
label_combine_path = os.path.join(abspath, 'data/label_combine')

select_cid3 = []
with open('cand_cid', 'r') as f:
    for line in f.readlines():
        case = line.strip().split('\t')
        if len(case) != 6:
            continue

        select_cid3.append(case[4])

#select_attr = ['颜色', '材质', '适用季节', '款式', '花型'] 
select_attr = ['颜色',  '适用季节', '款式', '花型', '类型', '领型', '版型', '图案', '人群', '功能', '闭合方式', '袖型', '衣门襟']
# 生成label的字典
attrs_dict = {}
with open('category_attribute', 'r') as f:
    for line in f.readlines():
        case = line.strip().split('\t')
        if len(case) != 3:
            continue
        [cid3, attr, attrvs] = case
        if (cid3 in select_cid3) and (attr in select_attr):
            for attrv in attrvs.split('|'):
                if cid3 not in attrs_dict:
                    attrs_dict[cid3] = {attr:[attrv]}
                elif attr not in attrs_dict[cid3]:
                    attrs_dict[cid3][attr] = [attrv]
                elif attrv not in attrs_dict[cid3][attr]:
                    attrs_dict[cid3][attr].append(attrv)

attrs_labels=[] #所有class的字典
num_splits=[0] #每一个属性所拥有的class
NUM_CLASSES=0 #总类目数

for attr in select_attr:
    for (cid, value) in attrs_dict.items():
        if attr in value:
            for attrv in value[attr]:
                attrs_labels.append('{}|{}|{}'.format(cid, attr, attrv))
                NUM_CLASSES += 1
    num_splits.append(NUM_CLASSES)
num_splits = np.diff(num_splits)

print NUM_CLASSES

#写入num_splits
with open(num_splits_path,'wb') as f:
    for sp in num_splits:
        f.write('{}\n'.format(sp))

# 生成hots
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

with open('label_combine', 'wb') as f:
    for index, value in enumerate(attrs_labels):
        f.write('{}\t{}\n'.format(index, value))

