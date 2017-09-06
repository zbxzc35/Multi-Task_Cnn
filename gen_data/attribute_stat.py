#-*-coding:utf8-*-
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')
"""
生成每个cid下面的类目数
生成每个属性的跨类目数
"""
result = {}
attrs=[]
num_cate=0
select_cid3=[]
cid3_number={}
with open('cand_cid', 'r') as f:
    for line in f.readlines():
        case = line.strip().split('\t')
        if len(case) != 6:
            continue
            
        select_cid3.append(case[4])

with open('category_attribute', 'r') as f:
    for line in f.readlines():
        fields = line.strip().split('\t')
        if len(fields) != 3:
            continue
 
        [cid3, attr, value] = fields
        if cid3 in select_cid3:
            if attr not in result:
                result[attr]=[cid3]
                attrs.append(attr)
            else:
                if cid3 not in result[attr]:
                    result[attr].append(cid3)

            if cid3 not in cid3_number:
                cid3_number[cid3]=[attr]
            else:
                cid3_number[cid3].append(attr)


attr_num=np.array([int(len(result[x])) for x in attrs])
total_attr = sum(attr_num)
attr_index = np.argsort(-attr_num)
num_cate=len(select_cid3)
print num_cate

for (cid3,attr_all) in cid3_number.items():
    print '{}\t{}'.format(cid3, len(attr_all))

for i in attr_index:
    print '{}\t{}\t{}'.format(attrs[i], attr_num[i], '%.4f'%(float(attr_num[i]/float(num_cate))))
  

