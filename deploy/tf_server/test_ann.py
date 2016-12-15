import tensorflow as tf
import numpy as np
import base64,os
import json
import utils
import vgg
import pyjsonrpc
from annoy import AnnoyIndex
class Feature2():
    def __init__(self,model_path='model/00150000/'):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options = gpu_options,
                allow_soft_placement=True))
        saver = tf.train.import_meta_graph(model_path+'export.meta')
        saver.restore(self.sess,model_path+'export')
        self.img = self.sess.graph.get_tensor_by_name('Placeholder:0')
        self.out = tf.squeeze(self.sess.graph.get_tensor_by_name('vgg_16/avgp5/AvgPool:0'))
    def feat1(self,image_path):
        img_data = np.expand_dims(np.array(open(image_path,'r').read()),0)
        return self.sess.run(self.out,{self.img:img_data})
    def feat2(self,feat_string):
        img_data = np.expand_dims(np.array(feat_string),0)
        return self.sess.run(self.out,{self.img:img_data})
names = np.load('data/name.npy')
if not os.path.exists('model/inshop.ann'):
    feats = np.load('data/feats.npy')
    t = AnnoyIndex(512)
    for i,a in enumerate(feats):
        t.add_item(i,a)
    t.build(200)
    t.save('model/inshop.ann')
else:
    t = AnnoyIndex(512)
    t.load('model/inshop.ann')

worker = Feature2()
str_ = open('./test.jpg','r').read()
feat1 = worker.feat2(str_)
feat2 = np.load('rst2.npy')
print 'Extract Feature:',t.get_nns_by_vector(feat1,20)
print 'Serving Feature:',t.get_nns_by_vector(feat2,20)
