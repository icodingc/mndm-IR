"""
select negative by random sampling from shops set
select image according Total score in items.
"""
from tqdm import tqdm
import numpy as np
import sys
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from collections import namedtuple
import cPickle as pickle

name_file_ = 'old_/filename.npy'
feat_file_ = 'old_/feature_ap.npy'

def get_names2feat(name_file,feat_file):
    names = np.load(name_file)
    feature = normalize(np.load(feat_file))
    rst = {}
    for i,a in enumerate(names):
        rst[a]=np.squeeze(feature[i])
    return rst
with open('train.lst','r') as f:
    namelst = [a.strip() for a in f]
name2feat = get_names2feat(name_file_,feat_file_)
#####get names by items
vector = {}
for a in namelst:
    fg = a.split('/')[-2]
    if fg not in vector:
        vector[fg] = []
    vector[fg].append(a)
######get Totol Score
def get_total_score(images):
    rst=[]
    for im in images:
        score = 0.0
        for other in images:
            score += cosine(name2feat[im],name2feat[other])
        rst.append(score)
    rst = np.array(rst)
    return rst/np.sum(rst)
########Get Item & Total score
ItemVector={}
Item = namedtuple('Item','Image Score')
for item in vector:
    image_all = vector[item]
    total_score=get_total_score(image_all).tolist()
    ItemVector[item]=Item(image_all,total_score)
print ItemVector.keys()[:5]
##
op = open('../profile/random_triplet_totalscore.lst','w')
for i,pic in tqdm(enumerate(namelst)):
    cur_fg = pic.split('/')[-2]
    op.write(pic+'\t')
    # random uniform
    op.write(np.random.choice(vector[cur_fg])+'\t')
    # select random negatives  per shop per example
    # select item shop
    negs = np.random.choice(vector.keys(),5)
    while cur_fg in negs:
        negs = np.random.choice(vector.keys(),14)
    for neg in negs:
        # random according Total Score
    	op.write(np.random.choice(ItemVector[neg].Image,p=ItemVector[neg].Score)+'\t')
    op.write('\n')
op.close()
