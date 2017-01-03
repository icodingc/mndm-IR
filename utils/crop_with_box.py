import tensorflow as tf
import threading
from PIL import Image
import numpy as np
import sys
filename = './Anno/list_bbox_inshop.txt'

def worker(idx,ranges,filelists):
    for a in filelists[ranges[idx][0]:ranges[idx][1]]:
        tmp = a.strip().split(' ')
        pic = tmp[0]
        a,b,c,d = map(float,tmp[-4:])
        Image.open(pic).crop((a,b,c,d)).save(pic)
num = 8
with open(filename,'r') as f:
    filenames = f.readlines()
filenames = filenames[2:]
print 'All %d files' % len(filenames)
spacing = np.linspace(0,len(filenames),num+1).astype(np.int)
ranges = []
threads = []
for i in xrange(len(spacing)-1):
    ranges.append([spacing[i],spacing[i+1]])
print ranges

coord = tf.train.Coordinator()
threads = []
for thread_idx in xrange(len(ranges)):
    args = (thread_idx,ranges,filenames)
    t = threading.Thread(target=worker,args=args)
    t.start()
    threads.append(t)
coord.join(threads)
