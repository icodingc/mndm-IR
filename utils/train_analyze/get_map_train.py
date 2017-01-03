from annoy import AnnoyIndex
import numpy as np
import sys
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--n', dest='name', type=str, default='filename.npy', help='file name')
parser.add_argument('--f', dest='feature', type=str, default='feature.npy', help='feature ')
parser.add_argument('--d', dest='dim', type=int, default=512, help='dim of per feature')
parser.add_argument('--k', dest='topk', type=int, default=20, help='top map')
parser.add_argument('--p', dest='pca', type=int, default=0, help='0 for pca 1 else')
parser.add_argument('--l', dest='cc', type=str, default='ap', help='0 for pca 1 else')
args = parser.parse_args()


name_file = args.name
feat_file = args.feature
dim = args.dim
k = args.topk
flag = args.pca
cc = args.cc

def get_names2feat(name_file,feat_file):
    names = np.load(name_file)
    feature = normalize(np.load(feat_file))
    print 'src dim %d' % dim
    if flag==1:
        pca = PCA(n_components=dim/2,whiten=True)
        print 'pca to dim %d'% (dim/2)
        feature = pca.fit_transform(feature)
    rst = {}
    for i,a in enumerate(names):
        rst[a]=np.squeeze(feature[i])
    return rst
query_name2feat = get_names2feat('./train_filename.npy','./train_feature_ap.npy')
gallery_name2feat = get_names2feat('./train_filename.npy','./train_feature_ap.npy')

with open('train.lst','r') as f:
    query = [a.strip() for a in f]
with open('train.lst','r') as f:
    gallery = [a.strip() for a in f]


gallery_feat = []
for b in gallery:
    gallery_feat.append(gallery_name2feat[b])
f = dim
if flag==1:f = dim/2
t = AnnoyIndex(f)
for i,a in enumerate(gallery_feat):
    t.add_item(i,a)
t.build(100)

def solve(pic,rst_names):
    fg = pic.split('/')[-2]
    for rp in rst_names:
        fgg = rp.split('/')[-2]
        if fg==fgg:return True
    return False
# get map
# query num = 14218
def get_ap(k=20):
    cnt=0
    for i,pic in enumerate(query):
        cur_feat = query_name2feat[pic]
        rst_idx = t.get_nns_by_vector(cur_feat,n=k)
        rst_names = [gallery[idx] for idx in rst_idx[:]]
        if solve(pic,rst_names):
            cnt+=1
    print('k=%d,map:'% k,cnt/(25882.0))
get_ap(20)
#get_ap(50)
