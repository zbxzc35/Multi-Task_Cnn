import os, sys

for line in sys.stdin:
	itemarr = line.strip().split('\t')


	subarr = itemarr[2].split('\001')
	assert(len(subarr) == 2)
	assert(subarr[0] == 'string')
	strlen = len(subarr[-1])

	subarr = itemarr[3].split('\001')
	assert(len(subarr) == 2)
	assert(subarr[0] == 'int')
	int_len = len(subarr[-1].split('|'))
	

	subarr = itemarr[4].split('\001')
	assert(len(subarr) == 2)
	assert(subarr[0] == 'float')
	float_len = len(subarr[-1].split('|'))
	print strlen, int_len, float_len
	#print len(itemarr[-1])
