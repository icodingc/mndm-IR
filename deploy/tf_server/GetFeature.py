import numpy as np
import os,sys
import tensorflow as tf
import utils
import vgg
import time

class Feature1():
    def __init__(self,model_path='model/inshop.sgd.adam'):
        self.model_path = model_path
        self.x = tf.placeholder(tf.string,shape=[])
        img = tf.image.decode_jpeg(self.x,channels=3)
        img = tf.expand_dims(utils.preprocess_image(img),0)

        self.feature = vgg.inference(img)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.3)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        assert tf.gfile.Exists(self.model_path)
        saver = tf.train.Saver()
        print('Using model from {}'.format(self.model_path))
        saver.restore(self.sess,self.model_path)
    def feat(self,image_path):
        img_data = open(image_path,'r').read()	
        return np.squeeze(self.sess.run(self.feature,feed_dict={self.x:img_data}))
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
    def feat(self,image_path):
        img_data = np.expand_dims(np.array(open(image_path,'r').read()),0)
        return self.sess.run(self.out,{self.img:img_data})
class Feature3():
    def __init__(self,model_path='model/vgg_serving/00150000/'):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options = gpu_options,
                allow_soft_placement=True))
        saver = tf.train.import_meta_graph(model_path+'export.meta')
        saver.restore(self.sess,model_path+'export')
        self.img = self.sess.graph.get_tensor_by_name('tf_example:0')
        self.out = tf.squeeze(self.sess.graph.get_tensor_by_name('vgg_16/avgp5/AvgPool:0'))
    def feat(self,image_path):
        img_data = np.expand_dims(np.array(open(image_path,'r').read()),0)
        return self.sess.run(self.out,{self.img:img_data})
if __name__ == '__main__':
    t2 = Feature1()
    for i in xrange(2):
        start = time.time()
        rst =t2.feat('./test.jpg')
        print time.time() -start
    print rst.shape
    np.save('F_src2',rst)
