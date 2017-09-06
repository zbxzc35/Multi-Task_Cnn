#-*-coding:utf8-*-

import os
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
计算预测增补率和precision recall F1
"""
def stat_num():
    # 计算每种cid出现的次数
    origin_num=np.zeros(16)
    pred_num=np.zeros(16)

    fns = [os.path.join(root,fn) for root,dirs,files in os.walk('./test_pred') for fn in files]

    for filename in fns:
        with open(filename, 'r') as f:
            for line in f.readlines():
                case = line.strip().split('\t')
                if len(case) != 6:
                    print filename
                    print '\t'.join(case)
                    continue
                else:
                    skuid, cid, type1, type1_attrs, type2, type2_attrs = line.strip().split('\t')
            
                origin_attrs = type1_attrs.split(',')
                origin_n = len(origin_attrs)
                origin_num[origin_n] += 1

                pred_attrs = type2_attrs.split(',')
                pred_n = len(pred_attrs)
                pred_num[pred_n] += 1

    return origin_num, pred_num

def calculate_avg(num_array):
    #平均属性数
    all_sum=0
    num_sum=np.sum(num_array)
    for i in range(len(num_array)):
        all_sum += num_array[i]*i

    return all_sum/float(num_sum)

def performance_stat():
    #计算每个class的precision和recall

    attrs_labels = []
    with open('label_combine', 'r') as f:
        for line in f.readlines():
            labelid, label = line.strip().split('\t')
            attrs_labels.append(label)

    N = len(attrs_labels)
    TP = np.zeros(N)
    P = np.zeros(N) # TP + FP
    T = np.zeros(N) # TP + FN

    fns = [os.path.join(root,fn) for root,dirs,files in os.walk('./test_pred') for fn in files]

    for filename in fns:
        with open(filename, 'r') as f:
            for line in f.readlines():
                case = line.strip().split('\t')
                if len(case) != 6:
                    continue
                else:
                    skuid, cid, type1, type1_attrs, type2, type2_attrs = line.strip().split('\t')

                for attrs in type1_attrs.split(','):
                    label = cid + '|' + attrs
                    T[attrs_labels.index(label)] += 1

                for attrs in type2_attrs.split(','):
                    label = cid + '|' + attrs
                    P[attrs_labels.index(label)] +=1
                    if attrs in type1_attrs:
                        TP[attrs_labels.index(label)] +=1
    
#    for i in range(N):
#        print TP[i], P[i], T[i]

    precision = TP / P
    recall = TP / T
    F1 = 2*precision*recall/(precision+recall)
    
    with open('PR', 'wb') as f:
        for i in range(N):
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(TP[i], P[i], T[i], precision[i], recall[i], F1[i]))
#    return precision, recall, F1

if __name__ == '__main__':
    origin_num, pred_num = stat_num()
    print origin_num, pred_num

    origin_avg = calculate_avg(origin_num)
    pred_avg = calculate_avg(pred_num)
    print origin_avg, pred_avg

    performance_stat()
#    print precision
#    print recall
#    print F1
