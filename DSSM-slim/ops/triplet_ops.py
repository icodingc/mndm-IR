"""
@@_triplet_multi_loss
@@_dssm_loss
#############
@@_dssm_loss_with_ap[min sim{a,p}]
#############
@@_dssm_loss_with_label[label smoothing]
@@_dssm_loss_with_label_noise[label smoothing with Noise]
#############
@@_dssm_learn_loss[learning similarity]
@@_dssm_loss_one_neg[neg=1 with multi examples]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.framework import deprecated

slim = tf.contrib.slim

#########################
#### contrastive loss
#########################
@deprecated('2016-11-01', 'Please do not use this. Result is bad!!')
def _siamese_loss(anchor,feature,label,alpha,scope=None):
  with tf.name_scope(scope,'SiameseLoss',[anchor,feature]):
  	alpha = tf.convert_to_tensor(alpha,
                                  dtype=anchor.dtype.base_dtype,
                                  name='alpha_margin')
  	dis = tf.reduce_sum(tf.square(tf.sub(anchor,feature)),1)
  	basic_loss = label*dis + (1. - label)*tf.nn.relu(alpha - dis)
  	rst_loss = 0.5*tf.reduce_mean(basic_loss,0)
  	slim.losses.add_loss(rst_loss)
  	return rst_loss
#################################
####Triplet Loss with large margin
#################################
@deprecated('2016-10-27', 'Please use _triplet_multi_loss instead')
def _triplet_loss(anchor,positive,negative,alpha,
                scope=None):
  with tf.name_scope(scope,'TripletLoss',[anchor,positive,negative]):
    alpha = tf.convert_to_tensor(alpha,
                                  dtype=anchor.dtype.base_dtype,
                                  name='alpha_margin')
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor,positive)),1)
    neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor,negative)),1)
    basic_loss = tf.nn.relu(tf.add(tf.sub(pos_dist,neg_dist),alpha),name='tripletloss')
    rst_loss = tf.reduce_mean(basic_loss,0)  
    slim.losses.add_loss(rst_loss)
    return rst_loss
#####################################
#TODO by default,we have two negatives
######################################
@deprecated('2016-10-27', 'Please use _triplet_multi_loss instead')
def _triplet_neg_loss(anchor,positive,negatives,alpha,scope=None):
  with tf.name_scope(scope,'DSSMMargin',[anchor,positive,negatives]):
    alpha = tf.convert_to_tensor(alpha,
                              dtype=anchor.dtype.base_dtype,
                              name='alpha_margin')
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor,positive)),1)
    neg_dist1 = tf.reduce_sum(tf.square(tf.sub(anchor,negatives[0])),1)
    neg_dist2 = tf.reduce_sum(tf.square(tf.sub(anchor,negatives[1])),1)
    basic_loss1 = tf.nn.relu(tf.add(tf.sub(pos_dist,neg_dist1),alpha),name='neg_loss_1')
    basic_loss2 = tf.nn.relu(tf.add(tf.sub(pos_dist,neg_dist2),alpha),name='neg_loss_2')
    rst_loss = tf.add(tf.reduce_mean(basic_loss1,0),tf.reduce_mean(basic_loss2,0)) * 0.5
    slim.losses.add_loss(rst_loss)
    return rst_loss 
##########################################
#multi neg Triplet loss[a form to add example]
##########################################
def _triplet_multi_loss(anchor,positive,negatives,alpha,
                scope=None):
  with tf.name_scope(scope,'TripletMultiLoss',[anchor,positive,negatives]):
    alpha = tf.convert_to_tensor(alpha,
                                  dtype=anchor.dtype.base_dtype,
                                  name='alpha_margin')
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor,positive)),1)
    neg_dists = [tf.reduce_sum(tf.square(tf.sub(anchor,n)),1) for n in negatives]
    basic_loss = [tf.nn.relu(tf.add(tf.sub(pos_dist,neg_dist),alpha)) for neg_dist in neg_dists]
    rst_loss = tf.reduce_mean(tf.concat(0,basic_loss),0)
    slim.losses.add_loss(rst_loss)
    return rst_loss
##################
### cos distance
##################
def _cosine_distance(a,b):
  a.get_shape().assert_is_compatible_with(b.get_shape())
  return tf.expand_dims(tf.reduce_sum(tf.mul(a,b),1),1)
##################
###  L2 distance
##################
def _euclidean_distance(a,b):
  a.get_shape().assert_is_compatible_with(b.get_shape())
  return tf.expand_dims(tf.reduce_sum(tf.square(tf.sub(a,b)),1),1)

#TODO add negative sampling [inspired from word2vec]
@deprecated('2016-11-01', 'I do not know if this is work')
def _nce_loss(anchor,positive,negatives,labels=None,scope=None):
  with tf.name_scope(scope,'NCE',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    labels = tf.constant([[1.0,0.0,0.0]])
    batch_size = logits.get_shape()[0]
    labels = tf.tile(labels,tf.pack([batch_size,1]))
    labels = math_ops.to_float(labels)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())

    all_ones = array_ops.ones_like(labels)
    target = math_ops.sub(2*labels,all_ones) 
    sig = tf.sigmoid(math_ops.mul(logits,target))
    rst_loss = tf.reduce_mean(-tf.log(tf.reduce_prod(sig,1))) 
    slim.losses.add_loss(rst_loss)
    return rst_loss
##############################
# similarity learning with DSSM
###############################
def _dssm_learn_loss(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'dssm_learn',[anchor,positive,negatives]):
    rst_cat = [tf.concat(1,[anchor,n]) for n in [positive]+negatives]
    neg_add = len(rst_cat)
    # batch*[128*2]
    # add similarity learning
    rst_batch = tf.concat(0,rst_cat)
    # [n*batch]*[128*2]
    batch_sim = slim.fully_connected(rst_batch,1, scope='sim')
    # [n*batch]*[1]
    rst_split=tf.split(0,neg_add,batch_sim)
    rst = tf.concat(1,rst_split)
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    rst_loss = tf.reduce_mean(-tf.log(tf.squeeze(tf.slice(p,[0,0],[-1,1])),name='Dssmloss'))
    slim.losses.add_loss(rst_loss)
    return rst_loss
####################################
#Softmax with importance sampling && 
#Deep similar sementic Model   using cosine similarity.
####################################
def _dssm_loss(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    rst_loss = tf.reduce_mean(-tf.log(tf.squeeze(tf.slice(p,[0,0],[-1,1])),name='Dssmloss'))
#    rst_loss = tf.Print(rst_loss,[rst_loss],message='hard loss is')
    slim.losses.add_loss(rst_loss)
    return rst_loss
######################################
#Monte-Carlo approximation of gradient
######################################
@deprecated('2016-12-05', 'I do not know if this is work')
def _dssm_mc_loss(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'dssm_mc',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = gamma*tf.concat(1,rst)
    sim_pos = -1.2*tf.squeeze(tf.slice(logits,[0,0],[-1,1]))
    sim_negs = 0.2*tf.reduce_sum(logits,1)
    print('sim_pos.shape',sim_pos)
    print('sim_negs.shape',sim_negs)
    rst_loss = tf.reduce_mean(sim_pos+sim_negs)
    slim.losses.add_loss(rst_loss)
    return rst_loss
######################################
#Importance sampling with Uniform(Q)
######################################
def _dssm_is_loss(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'dssm_is',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = gamma*tf.concat(1,rst)
    sim_pos = -1.*tf.squeeze(tf.slice(logits,[0,0],[-1,1]))
    sim_negs = tf.log(tf.reduce_sum(tf.exp(tf.slice(logits,[0,1],[-1,-1])),1))
    rst_loss = tf.reduce_mean(sim_pos+sim_negs)
    slim.losses.add_loss(rst_loss)
    return rst_loss 
##################################################
#Importance sampling with Proposal distribution(Q)
##################################################
def _dssm_is_Q_loss(anchor,positive,negatives,gamma,Q,scope=None):
  with tf.name_scope(scope,'dssm_is',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = gamma*tf.concat(1,rst)
    sim_pos = -1.*tf.squeeze(tf.slice(logits,[0,0],[-1,1]))
    #TODO process Q
    sim_negs_ = tf.div(tf.exp(tf.slice(logits,[0,1],[-1,-1])),Q)
    sim_negs = tf.log(tf.reduce_sum(sim_negs_,1))
    rst_loss = tf.reduce_mean(sim_pos+sim_negs)
    slim.losses.add_loss(rst_loss)
    return rst_loss 
#TODO conv to moni gamma
@deprecated('2016-12-05', "it dosn't work")
def _dssm_loss_var_gamma(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]

    gamma = tf.get_variable('gamma',[1],initializer=tf.constant_initializer(10.0))
    logits = tf.concat(1,rst)
    gamma = tf.Print(gamma,[gamma],'This is gamma:',50)
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    rst_loss = tf.reduce_mean(-tf.log(tf.squeeze(tf.slice(p,[0,0],[-1,1])),name='Dssmloss'))
    slim.losses.add_loss(rst_loss)
    return rst_loss 
####################################
#TODO add label smothing
def _dssm_loss_with_label(anchor,positive,negatives,gamma,label_smoothing=0.0,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    batch_size = logits.get_shape()[0]
    num_neg = len(rst)-1
    label_pos = tf.ones(tf.pack([batch_size,1]))
    label_negs = tf.zeros(tf.pack([batch_size,num_neg]))
    labels = tf.concat(1,[label_pos,label_negs])
    if label_smoothing > 0:
    	smooth_positives = 1.0 - label_smoothing
    	smooth_negatives = label_smoothing / len(rst)
    	labels = labels * smooth_positives + smooth_negatives
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    xent_loss = tf.nn.softmax_cross_entropy_with_logits(gamma*logits,labels,name='dssmloss')
    rst_loss = tf.reduce_mean(xent_loss)
    slim.losses.add_loss(rst_loss)
    return rst_loss 
#TODO add a loss which minimum similarity{a,p},with alpha tradeoff
@deprecated('2016-12-05', 'I do not know if this is work')
def _dssm_loss_with_ap(anchor,positive,negatives,
	gamma=4.2,label_smoothing=0.2,alpha_tradeoff=0.1,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    batch_size = logits.get_shape()[0]
    num_neg = len(rst)-1
    label_pos = tf.ones(tf.pack([batch_size,1]))
    label_negs = tf.zeros(tf.pack([batch_size,num_neg]))
    labels = tf.concat(1,[label_pos,label_negs])
    if label_smoothing > 0:
    	smooth_positives = 1.0 - label_smoothing
    	smooth_negatives = label_smoothing / len(rst)
    	labels = labels * smooth_positives + smooth_negatives
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    xent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(gamma*logits,labels,name='dssmloss'))
    ## Add second loss
    dis_loss = alpha_tradeoff * tf.reduce_mean((_euclidean_distance(anchor,positive)))
    ## rst_loss = tf.Print(rst_loss,[rst_loss],message='soft loss is')
    slim.losses.add_loss(xent_loss+dis_loss)
    return xent_loss
# with label smoothing and noise
@deprecated('2016-12-05', 'I do not know if this is work')
def _dssm_loss_with_label_noise(anchor,positive,negatives,gamma,label_smoothing=0.2,
	        alpha=0.8,prob=0.8,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    batch_size = logits.get_shape()[0]
    num_neg = len(rst)-1
    label_pos = tf.ones(tf.pack([batch_size,1]))
    label_negs = tf.zeros(tf.pack([batch_size,num_neg]))
    labels = tf.concat(1,[label_pos,label_negs])
    if label_smoothing > 0:
    	smooth_positives = 1.0 - label_smoothing
    	smooth_negatives = label_smoothing / len(rst)
    	labels = labels * smooth_positives + smooth_negatives
    if prob > 0:
    	# noise and clip
    	noise_tensor = tf.random_normal(tf.shape(labels),stddev=alpha)
    	noise_tensor = tf.clip_by_value(noise_tensor,-0.8,1.0)
    	# prob
    	random_tensor = prob
    	random_tensor += tf.random_uniform(tf.pack([batch_size]))
    	binary_tensor = tf.floor(random_tensor)
    	binary_tensor = tf.tile(tf.expand_dims(binary_tensor,1),[1,num_neg+1])
    	sig_tensor = noise_tensor*binary_tensor
    	# add to label & Relu[label > 0]
    	labels += labels * sig_tensor
    	labels = tf.nn.relu(labels)
    	# Norm ~ P()
    	labels=tf.div(labels,tf.tile(tf.expand_dims(tf.reduce_sum(labels,1),1),[1,num_neg+1]))
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    xent_loss = tf.nn.softmax_cross_entropy_with_logits(gamma*logits,labels,name='dssmloss')
    rst_loss = tf.reduce_mean(xent_loss)
    slim.losses.add_loss(rst_loss)
    return rst_loss 
###################################
#one neg, but like _triplet_multi_loss,reduce computation.
#####################################
def _dssm_loss_one_neg(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    ap = rst[0]
    logits = [tf.concat(1,[ap,an]) for an in rst[1:]]
    logits = tf.concat(0,logits) 
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    rst_loss = tf.reduce_mean(-tf.log(tf.squeeze(tf.slice(p,[0,0],[-1,1])),name='Dssmloss'))
    slim.losses.add_loss(rst_loss)
    return rst_loss 
#TODO there,using -distance.[equal to dssm using similarity]
def _dssm_ste_loss(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'ste',[anchor,positive,negatives]):
    rst = [-_euclidean_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    logits = tf.concat(1,rst)
    p = tf.nn.softmax(gamma*logits)
    rst_loss = tf.reduce_mean(-tf.log(tf.squeeze(tf.slice(p,[0,0],[-1,1])),name='Dssmloss'))
    slim.losses.add_loss(rst_loss)
    return rst_loss
#TODO t-STE
def _dssm_tste_loss(anchor,positive,negatives,alpha,scope=None):
  with tf.name_scope(scope,'t-STE',[anchor,positive,negatives]):
    alpha = tf.convert_to_tensor(alpha,
                              dtype=anchor.dtype.base_dtype,
                              name='degree')
    beta = -(alpha+1.0)/2.0
    rst = [(1.0+_euclidean_distance(anchor,n)/alpha) for n in [positive]+negatives]
    rst2 = [tf.pow(k,beta)for k in rst]
    # batch*(p+ns)
    logits = tf.concat(1,rst2)
    logits_sum = tf.reduce_sum(logits,1)
    logits_p = tf.squeeze(tf.slice(logits,[0,0],[-1,1]))
    #logits_sum.get_shape().assert_is_compatible_with(logits_p.get_shape())
    rst_loss = tf.reduce_mean(-tf.log(tf.div(logits_p,logits_sum)))
    slim.losses.add_loss(rst_loss)
    return rst_loss
