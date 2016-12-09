from annoy import AnnoyIndex
import numpy as np
import sys
from sklearn.preprocessing import normalize
from tqdm import tqdm
name_file_ = 'old_/filename.npy'
feat_file_ = 'old_/feature_ap.npy'

def get_names2feat(name_file,feat_file):
    names = np.load(name_file)
    feature = normalize(np.load(feat_file))
    rst = {}
    for i,a in enumerate(names):
        rst[a]=np.squeeze(feature[i])
    return rst

with open('train.lst','r') as f:namelst = [a.strip() for a in f]
name2feat = get_names2feat(name_file_,feat_file_)
#####get names by items
vector = {}
for a in namelst:
    fg = a.split('/')[-2]
    if fg not in vector:
        vector[fg] = []
    vector[fg].append(a)
#######################
# build ann           #
#######################
shop_feat = []
for b in namelst:shop_feat.append(name2feat[b])
t = AnnoyIndex(512)
for i,a in enumerate(shop_feat):
    t.add_item(i,a)
t.build(100)

op = open('hard_and_random_triplet.lst','w')

def IsEqualItem(cur_fg,rst_names):
    flag=False
    idx=0
    for i,r_name in enumerate(rst_names):
        rfg = r_name.split('/')[-2]
        if cur_fg==rfg:
            flag=True
            idx=i
            break
    return flag,idx

cnt_hard = 0
cnt_random = 0   
for pic in tqdm(namelst):
    cur_feat = name2feat[pic]
    rst_idx = t.get_nns_by_vector(cur_feat,n=20)
    #TODO should remove self 
    rst_names = [namelst[idx] for idx in rst_idx][1:]
    cur_fg = pic.split('/')[-2]
    pic_idx = 20
    #cur_fg if in Top20
    flags,item_idx = IsEqualItem(cur_fg,rst_names)
    if flags:
        # if cur_fg in rst_names [Random]
        cnt_random+=1
        op.write(pic+'\t')
        op.write(np.random.choice(vector[cur_fg])+'\t')
        negs = np.random.choice(vector.keys(),5)
        while cur_fg in negs:
            negs = np.random.choice(vector.keys(),5)
        for neg in negs:
            op.write(np.random.choice(vector[neg])+'\t')
        op.write('\n')
    else:
        #from items [Hard]
        cnt_hard+=1
        op.write(pic+'\t')
        op.write(np.random.choice(vector[cur_fg])+'\t')
        negs = np.random.choice(rst_names[:pic_idx],5)
        for neg in negs:
            op.write(neg+'\t')
        op.write('\n')
print 'cnt_hard',cnt_hard
print 'cnt_ramdom',cnt_random
op.close()
