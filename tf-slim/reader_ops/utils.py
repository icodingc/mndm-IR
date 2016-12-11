import time,sys
import threading
import time
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from collections import namedtuple
import cPickle as pickle
#from tqdm import tqdm

file_name = '/home/zhangxuesen/data/profile/train.lst'
name_file_ ='matrix/filename.npy'
feat_file_ ='matrix/feature.npy'

def get_names2feat(name_file,feat_file):
    names = np.load(name_file)
    feature = normalize(np.load(feat_file))
    rst = {}
    for i,a in enumerate(names):
        rst[a]=np.squeeze(feature[i])
    return rst
start = time.time()
#####train lst
with open(file_name,'r') as f:
    namelst = [a.strip() for a in f]
#####get name2feat
name2feat=get_names2feat(name_file_,feat_file_)
#####get names by items
vector = {}
for a in namelst:
    fg = a.split('/')[-2]
    if fg not in vector:
        vector[fg] = []
    vector[fg].append(a)
#####item_lst & Sim-Matrix & Iter_mean_vec
def get_mean_feat(imgs):
	shape = name2feat[imgs[0]].shape
	rst = np.zeros(shape)
	cnt = len(imgs)
	for ap in imgs:
		rst+=name2feat[ap]
	return rst/cnt
##Item_lst
item_lsts = sorted(vector.keys())
item_cnt = len(item_lsts)
print "Total",item_cnt,'items...'
item_idxs = list(xrange(item_cnt))
item2idx = dict(zip(item_lsts,item_idxs))
##Item_mean_vec
Item_mean={}
for item in item_lsts:
	image_all = vector[item]
	vector_mean = get_mean_feat(image_all)
	Item_mean[item]=vector_mean
##Item_sim_matrix,In this way,[may be sampling positive!!!]
Item_sim_matrix = []
for item1 in item_lsts:
	#TODO get similarity not distance
	cur_sim_v = np.array([(1.-cosine(Item_mean[item1],Item_mean[other])) for other in item_lsts])
	#TODO Simarity ==> prob
	cur_prob_v = cur_sim_v/np.sum(cur_sim_v)
	Item_sim_matrix.append(cur_prob_v)
np.save('matrix/sim_matrix',np.array(Item_sim_matrix))
end = time.time()
print 'Processing Time: ',end-start
