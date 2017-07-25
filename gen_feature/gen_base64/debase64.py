import os, sys, base64, random
import argparse, time

parser = argparse.ArgumentParser()
parser.add_argument("--save", help = "saving dir of tf-record files")
parser.add_argument("-n", "--num", help = "number of saved tf-record files")
parser.add_argument('--maxin', help = "max input line number")
parser.add_argument('-m', '--maxout', help = "max output line number")
args = parser.parse_args()

save_num =(int)(args.num)
ofst_arr = []
for i in xrange(0, save_num):
	ofst_arr.append(open(os.path.join(args.save, '%s.tfrecord' % i), 'wb'))

outbuf = [[] for x in xrange(0, save_num)]

max_num = (int)(args.maxin) if args.maxin else 1e30
max_out = (int)(args.maxout) if args.maxout else 1e30

print 'max_in_num[%d]' % max_num
print 'max_out_num[%d]' % max_out

def null_selector(itemarr):
	return True

def rand_select(itemarr):
	flag = itemarr[1]
	if flag != '1':
		return True
	else:
		d = random.randint(0, 7)
		if d % 7 == 1:
			return True
		
		return False

i_out = 0
cap = 0
MAX_CAP = 256
ll = 0
start_time = time.time()
last_time = start_time
for line in sys.stdin:
	ll += 1
	if ll > max_num:
		break;
	itemarr = line.strip().split('\t')
	if null_selector(itemarr):
		ss = base64.b64decode(itemarr[-1])
		idx = random.randint(0, save_num - 1)
		ofst_arr[idx].write(ss)
		i_out += 1
		if i_out > max_out:
			break
	
	if ll % 200000 == 0:
		cur_time = time.time()
		duration = cur_time  - start_time
		print 'readline[%d], out_line[%d], takes %.2fsecs, speed %.2f/sec, total speed %.2f/sec' % (ll, i_out, duration, 200000 / (cur_time - last_time), ll * 1.0 / duration)
		last_time = cur_time



for ii in xrange(0, save_num):
	#ofst_arr[ii].write(''.join(outbuf[ii]))
	ofst_arr[ii].close()

duration = time.time() - start_time
print 'll[%s], n[%s], takes %.2fsecs, %.2f/sec' % (ll, i_out, duration, ll * 1.0 / duration)
exit(0)
