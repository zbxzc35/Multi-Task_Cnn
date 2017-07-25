#-*-coding:utf8-*-
import os
import sys

last_key = ''

def dump():
    global last_key
    if last_key == '':
        return
    print last_key

for line in sys.stdin:
    if line.strip() != last_key:
        dump()
        last_key = line.strip()
dump()
