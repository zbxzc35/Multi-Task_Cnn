import sys
import numpy as np

result = {}
attrs=[]
num_cate=0
cates=[]

for line in sys.stdin:
  fields = line.strip().split('|')
  if len(fields) != 3:
    continue
  [cate, attr, value] = fields
  if cate not in cates:
    cates.append(cate)
    num_cate += 1
  if attr not in result:
    result[attr]=[cate]
    attrs.append(attr)
  else:
    if cate not in result[attr]:
      result[attr].append(cate)

attr_num=np.array([int(len(result[x])) for x in attrs])
total_attr = sum(attr_num)
attr_index = np.argsort(-attr_num)
print num_cate
for i in attr_index:
  print '{}\t{}\t{}'.format(attrs[i], attr_num[i], '%.4f'%(float(attr_num[i]/num_cate)))

