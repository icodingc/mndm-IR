import numpy as np
from sklearn.preprocessing import normalize
#from sklearn.metrics.pairwise import cosine_similarity as cosine
#from scipy.spatial.distance import cosine
def get_p(score):
    pos = np.exp(score[0])
    z = np.sum(np.exp(score))
    return pos/z
def solver(score):
    gamma = [0.5,1.0,4.2,10]
    for g in gamma:
        a = get_p(g*np.array(score))
        print 'gamma {}==>{}'.format(g,a)

name_file_ = '/home/zhangxs/data/In-shop-IR/utils/old_/filename.npy'
feat_file_ = '/home/zhangxs/data/In-shop-IR/utils/old_/feature_ap.npy'
def get_names2feat(name_file,feat_file):
    names = np.load(name_file)
    feature = normalize(np.load(feat_file))
    rst = {}
    for i,a in enumerate(names):
        rst[a]=np.squeeze(feature[i])
    return rst
#name 2 feat
name2feat = get_names2feat(name_file_,feat_file_)
#with open('train_triplet.lst','r') as f:
#    idx = 0
#    for ex in f:
#        idx+=1
#        ex_lst = ex.strip().split('\t')
#        a,p=ex_lst[0],ex_lst[1]
#        ns = ex_lst[2:]
#        score=[cosine(name2feat[a],name2feat[e]) for e in [a]+[p]+ns]
#        print score
#        solver(score)
#        if idx==1:break
