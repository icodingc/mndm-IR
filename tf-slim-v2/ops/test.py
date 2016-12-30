import tensorflow as tf
import numpy as np
import triplet_ops as tp
sp = [10,5]
a = np.random.randn(10,5)
p = np.random.randn(10,5)
ns = [np.random.randn(10,5),np.random.randn(10,5)]
#############NP
def sig(x):return 1./(1.+np.exp(-x))
def cos_dis(a,b):
    return np.expand_dims(np.sum(np.multiply(a,b),1),1)
def _loss(a,p,ns):
    rst = [cos_dis(a,n) for n in [p]+ns]
    logits = np.concatenate(rst,1)
    labels = np.array([1.0,0.0,0.0]*10).reshape(10,3)
    all_ones = np.ones_like(labels)
    target = 2*labels-all_ones
    sig_ = sig(np.multiply(logits,target))
    rst_loss = np.mean(-np.log(np.prod(sig_,1)))
    return rst_loss
aa = tf.placeholder(tf.float32,[10,5])
pp = tf.placeholder(tf.float32,[10,5])
nn1 = tf.placeholder(tf.float32,[10,5])
nn2 = tf.placeholder(tf.float32,[10,5])
#############TF
rst = tp._nce_loss(aa,pp,[nn1,nn2])
sess = tf.Session()
print('tf..',sess.run(rst,{aa:a,pp:p,nn1:ns[0],nn2:ns[1]}))
print('np..',_loss(a,p,ns))
