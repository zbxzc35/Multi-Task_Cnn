# -*- coding:utf-8 -*- 
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext
from pyspark.sql import SparkSession
import sys
import json
import random

reload(sys)
sys.setdefaultencoding('utf8')


#相关路径 
input_paths = sys.argv[1]
save_path = sys.argv[2]

"""

假设表的形式是如下：
-------------------------------
sku_id   c_id1   c_id2  c_id3
***      ***      ***    ***

-----------------------------

"""

# 提取candidate的cid3
cid_mapping={}
with open('script/cand_cid', 'r') as f:
    for line in f.readlines():
        case = line.strip().split('\t')
        #cid = '{}|{}|{}'.format(case[1].decode('utf-8'), case[3].decode('utf-8'), case[5].decode('utf-8')) 
        cid = '{}|{}|{}'.format(case[1], case[3], case[5]) 
        cid_mapping[cid]=case[4]

select_attrs = [u'颜色', u'适用季节', u'款式', u'花型', u'类型', u'领型', u'版型', u'图案', u'人群', u'功能', u'闭合方式', u'袖型', u'衣门襟']

#select_attrs = [u'颜色', u'材质', u'适用季节', u'款式', u'花型']
def filter_attrs(attrs_all):
    #筛选出属性
    filter_attrs = []
    choose_attrs = []
    for attrs in attrs_all.split('|'):
        case = attrs.split('_')
        if len(case) != 2:
            continue

        [attr, attrv] = case
        if attr in select_attrs:
            if attr in choose_attrs:
                filter_attrs=''
                break
            else:
                choose_attrs.append(attr)
                filter_attrs.append(attrs)
    
    #if len(filter_attrs)<3:
    #    filter_attrs=''
    return '|'.join(filter_attrs)

def choose_column(line):
    """
    每一行数据为：
    sku_id, sku_name, item_name, item_desc, barndname_full, cid1, cid2, cid3, slogan, item_type, title, query, image, sku_attrs
    筛选出 cid3_number, sku_id, sku_attrs
    cid3_number要在我们所选的组别内
    """
    line = line.strip().split('\t')
    cid = '{}|{}|{}'.format(line[5].encode('utf8'),line[6].encode('utf8'),line[7].encode('utf8'))
    if cid in cid_mapping:
        cid3_number = cid_mapping[cid]
    else:
        cid3_number = ''
    return cid3_number, line[0], line[13]

def map_to_cid3(line):
    """
    每一行数据为：cid3, sku_id, sku_attrs
    """
    attrs = filter_attrs(line[2])
    return (line[0], [(line[1], attrs)])

def select_query(line):
    #从每个CID下面随机选取一定数量的query sample, 要求尽可能包括所有的类目
    num_query = 100

    key = line[0]
    items = line[1]

    query = []
    query_attrs = []
    for i in range(num_query):
        sample = random.choice(items)
        query.append((key, sample))
        
        sample_attrs_all = sample[1]
        for attrs in sample_attrs_all.split('|'):
            if (attrs not in query_attrs):
                query_attrs.append(attrs)

    max_search_query = 1000
    k=0
    while (k<max_search_query):
        k+=1
        sample = random.choice(items)
        sample_attrs_all = sample[1]

        b = False #判断有没有新的属性进来
        for attrs in sample_attrs_all.split('|'):
            if (attrs not in query_attrs):
                query_attrs.append(attrs)
                b = True
        if (b==True):
            query.append((key, sample))

    return query

def sample_pair(line):
    #给每个正样本找负样本
    cid = line[0]
    
    query = line[1][0]
    query_attrs_all = query[1]

    iterms = line[1][1]

    max_iter = 1000 #寻找负样本的最大iteration次数
    num_pair = 10  #针对每个属性的pair数
    pairs = []
    for attrs in query_attrs_all.split('|'):
        # attrs 是query 的某一个属性 如颜色_红色
        # 找 num_pair个同样 颜色_红色 的 positive sample
        # 找 num_pair个  颜色_其他 的 negative sample
        k = 0 #迭代次数
        num_pos = 0
        while (k<max_iter) and (num_pos <= num_pair):
            k += 1
            
            sample = random.choice(iterms)
            attrs_all_sample = sample[1]
            if (attrs in attrs_all_sample):
                num_pos += 1
                pair = [cid] + list(query) + list(sample)
                pairs.append(('\t'.join(pair)))
        
        k = 0
        num_neg = 0
        while (k<max_iter) and (num_neg <= num_pair):
            k += 1
            
            sample = random.choice(iterms)
            attrs_all_sample = sample[1]
            attr, attrv = attrs.split('_')
            if (attr in attrs_all_sample) and (attrs not in attrs_all_sample):
                num_neg += 1
                pair = [cid] + list(query) + list(sample)
                pairs.append(('\t'.join(pair)))

    return pairs

#sc = SparkContext("master","test")
sc = SparkSession.builder.appName("generate_pairwse_data").enableHiveSupport().getOrCreate()

# spark从Input_path里Load原始表
input_rdd = sc.sparkContext.textFile(input_paths)

# 提取在所需cid3中的图片
processed_rdd1 = input_rdd.map(choose_column).filter(lambda x: x[0] != '')

# 变换key为c_id3,且筛选出属性表中的属性
processed_rdd2 = processed_rdd1.map(map_to_cid3).filter(lambda x: x[1][0][1] != '').sample(False, 0.3, 81)

# 按key聚合数据，聚合后形式为:
# cid_3,[(item1),(item2),.........]
processed_rdd3 = processed_rdd2.reduceByKey(lambda x, y: x + y)

# 筛选出query sample，每个cid选择N个
processed_rdd_query = processed_rdd3.flatMap(select_query)

#随机抽样，配对Pair, 分别找到positive 和 negative sample
processed_pair = processed_rdd_query.leftOuterJoin(processed_rdd3).flatMap(sample_pair)

#储存结果
processed_pair.repartition(100).saveAsTextFile(save_path)

