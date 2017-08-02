#-*-coding:utf8-*-
import os
import sys
import numpy as np

def attrs_sort(origin_attrs_labels):
	attrs_dict={}
	select_attr=[]
	for attrs in origin_attrs_labels:
		[cid, attr, attrv] = attrs.strip().split('|')
		
		if attr not in select_attr:
			select_attr.append(attr)
		
		if cid not in attrs_dict:
			attrs_dict[cid] = {attr:[attrv]}
		elif attr not in attrs_dict[cid]:
			attrs_dict[cid][attr] = [attrv]
		elif attrv not in attrs_dict[cid][attr]:
			attrs_dict[cid][attr].append(attrv)

	attrs_labels=[]
	num_splits=[0]
	NUM_CLASSES=0
	
	for attr in select_attr:
		for (cid, value) in attrs_dict.items():
			if attr in value:
				for attrv in value[attr]:
					attrs_labels.append('{}|{}|{}'.format(cid, attr, attrv))
					NUM_CLASSES += 1
		num_splits.append(NUM_CLASSES)
	num_splits = np.diff(num_splits)         

	print NUM_CLASSES

	return attrs_labels, num_splits

def generate_attrs_labels(input_file, min_nimages, num_splits_path):
# generate the labels, such that all labels have more than min_nimages images
	attrs_nimages = {}
	for data_path in input_file:
		with open(data_path, 'r') as f:
			for line in f.readlines():
				[skuid, img, attrs_all] = line.strip().split('\t')
				
				for attrs in attrs_all.split(','):
					if attrs not in attrs_nimages:
						attrs_nimages[attrs] = 1
					else:
						attrs_nimages[attrs] += 1

	attrs_labels=[]
	for attrs, nimages in attrs_nimages.items():
		if nimages>min_nimages:
			attrs_labels.append(attrs)
	
	attrs_labels, num_splits = attrs_sort(attrs_labels)
	with open(num_splits_path,'wb') as f:
		for sp in num_splits:
			f.write('{}\n'.format(sp))

	return attrs_labels

def save_to_meta_path(attrs_labels, meta_path):
	NUM_CLASSES = len(attrs_labels)
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

if __name__ == '__main__':
	abspath = os.path.abspath('..')
	meta_path = os.path.join(abspath, 'data/batches.meta')
	num_splits_path = os.path.join(abspath, 'data/num_splits')

	input_file = [os.path.join(abspath, 'data/part-0000%d' % i) for i in xrange(10)]
	input_file = input_file + [os.path.join(abspath, 'data/part-000%d' % i) for i in xrange(10, 100)]

	min_nimages = 1000 
	attrs_labels = generate_attrs_labels(input_file, min_nimages, num_splits_path)
	
	save_to_meta_path(attrs_labels, meta_path)
