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

cid_mapping={}
with open('cand_cid', 'r') as f:
    for line in f.readlines():
        case = line.strip().split('\t')
        cid = '{}|{}|{}'.format(case[1], case[3], case[5]) 
        cid_mapping[cid]=case[4]

#select_attrs = [u'颜色', u'材质', u'适用季节', u'款式', u'花型']
#select_attrs = [u'颜色', u'适用季节', u'款式', u'花型', u'类型', u'领型', u'版型', u'图案', u'人群', u'功能', u'闭合方式', u'袖型', u'衣门襟']
select_attrs = [u'颜色',u'材质', u'适用季节', u'款式', u'花型', u'风格', u'类型', u'领型', u'版型', u'图案', u'人群', u'功能', u'闭合方式', u'袖型', u'衣门襟', u'主要材质']


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

    return '|'.join(filter_attrs)

def choose_column(line):
    """
    每一行数据为：
    sku_id, sku_name, item_name, item_desc, barndname_full, cid1, cid2, cid3, slogan, item_type, title, query, image, sku_attrs
    筛选出 cid3, sku_id, image, sku_attrs
    """
    line = line.strip().split('\t')
    cid = '{}|{}|{}'.format(line[5].encode('utf8'),line[6].encode('utf8'),line[7].encode('utf8'))
    if cid in cid_mapping:
        cid3_number = cid_mapping[cid]
    else:
        cid3_number = ''
    return cid3_number, line[0], line[12], line[13]

def map_to_cid3(line):
    """
    每一行数据为：cid3, sku_id, image, sku_attrs
    """
    
    attrs = filter_attrs(line[3])
    return line[0], line[1], line[2], attrs

def join_attr(line):
    sku_all = [line[0], line[1], line[2], line[3]]
    sku_b = [x.encode('utf8') for x in sku_all]
    return ('\t'.join(sku_b))

#sc = SparkContext("master","test")
sc = SparkSession.builder.appName("generate_pairwse_test_data").enableHiveSupport().getOrCreate()

# spark从Input_path里Load原始表
input_rdd = sc.sparkContext.textFile(input_paths)

# 提取在所需cid3中的图片
processed_rdd1 = input_rdd.map(choose_column).filter(lambda x: x[0] != '')

# 变换key为cid3
processed_rdd2 = processed_rdd1.map(map_to_cid3).filter(lambda x: x[3] != '')

processed_rdd3 = processed_rdd2.map(join_attr)
#储存结果
processed_rdd3.repartition(1000).saveAsTextFile(save_path)

