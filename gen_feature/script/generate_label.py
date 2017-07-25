#-*-coding:utf8-*-
import os
import sys
import argparse

result = dict()


# input is category_attribute
for line in sys.stdin:
  attrs = line.strip().split('\t')
  if len(attrs) != 3:
    continue
  [cid, attr, val] = attrs
  vals = val.split('|')
  for it in vals:
    key = '{}|{}|{}'.format(cid, attr, it)
    result.setdefault(key, 0)


idx = 0
for k in result:
  idx += 1
  print '{}\t{}'.format(k, idx)
