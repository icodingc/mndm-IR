import os
import sys
import threading
import numpy as np
import tensorflow as tf
home_dir = os.path.expanduser('~/data/inshop/')
file_name = home_dir + 'profile/train.lst'
base_dir = home_dir

#####train lst
with open(file_name,'r') as f:
    namelst = [a.strip() for a in f]
vector = {}
for a in namelst:
    fg = a.split('/')[-2]
    if fg not in vector:
        vector[fg] = []
    vector[fg].append(a)
item_lsts = vector.keys()
#TODO make sure that .keys() equal
item_lsts = sorted(item_lsts)
item_cnt = len(item_lsts)
item_idxs = list(xrange(item_cnt))
item2idx = dict(zip(item_lsts,item_idxs))
# item simimarity matrix
Item_sim_matrix=np.load('./reader_ops/matrix/sim_matrix.npy')
print('sim_mat.shape: ',Item_sim_matrix.shape)
#######################
def data_iter():
    #TODO iter items && shuffle
    file_lists = item_lsts
    count_file = len(file_lists)
    file_lists = np.array(file_lists)
    while True:
        idxs = np.arange(0,count_file)
        np.random.shuffle(idxs)
        shuf_lists = file_lists[idxs]

        for batch_idx in xrange(0,len(shuf_lists)):
            ##TODO get anchor item
            item_a = shuf_lists[batch_idx].strip()
            ##TODO get negative items with Q4
            Q4 = np.power(Item_sim_matrix[item2idx[item_a]],4)
            Q = Q4/np.sum(Q4)
            item_negs = np.random.choice(item_lsts,15,p=Q)
            while item_a in item_negs:
                item_negs = np.random.choice(item_lsts,15,p=Q)
            item_negs = item_negs.tolist()
            for sel_item in [item_a]+item_negs:
                imgs = np.random.choice(vector[sel_item],4)
                a1 = open(base_dir+imgs[0],'r').read()
                a2 = open(base_dir+imgs[1],'r').read()
                a3 = open(base_dir+imgs[2],'r').read()
                a4 = open(base_dir+imgs[3],'r').read()
                yield a1,a2,a3,a4

class CustomRunner(object):
    def __init__(self,batch_size=16,shuffle=False,vp=None):
        self.batch_size = batch_size
        self.image_a1 = tf.placeholder(tf.string,[])
        self.image_a2 = tf.placeholder(tf.string,[])
        self.image_a3 = tf.placeholder(tf.string,[])
        self.image_a4 = tf.placeholder(tf.string,[])
        self.vp=vp
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        c_sp=[224,224,3]
        c_shapes=[c_sp]*4
        c_dtypes=[tf.float32]*4
        if shuffle:
            self.queue = tf.RandomShuffleQueue(shapes=c_shapes,
                                  dtypes=c_dtypes,
                                  capacity=35,
                                  min_after_dequeue=20)
        else:
            self.queue = tf.FIFOQueue(shapes=c_shapes,
                                  dtypes=c_dtypes,
                                  capacity=33)
        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        img_a1 = self.transfer(self.image_a1,'a1')
        img_a2 = self.transfer(self.image_a2,'a2')
        img_a3 = self.transfer(self.image_a3,'a3')
        img_a4 = self.transfer(self.image_a4,'a4')
        features=[img_a1,img_a2,img_a3,img_a4]
        self.enqueue_op = self.queue.enqueue_many(features)
    def transfer(self,input_string,prefix):
    	img = tf.cast(tf.image.decode_jpeg(input_string,channels=3),tf.float32)
    	return tf.expand_dims(self.vp(img,224,224,prefix=prefix),0)
    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        #TODO fixed batch_size
        images_batch= self.queue.dequeue_many(self.batch_size)
        return images_batch

    def thread_main(self, sess,coord):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for a1,a2,a3,a4 in data_iter():
            feed_d = {self.image_a1:a1,
                      self.image_a2:a2,
                      self.image_a3:a3,
                      self.image_a4:a4,
                      }
            sess.run(self.enqueue_op, feed_dict=feed_d)
            if coord.should_stop():
                break

    def start_threads(self, sess,coord,n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,coord,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
if __name__ == '__main__':
    # Doing anything with data on the CPU is generally a good idea.
    with tf.device("/cpu:0"):
        custom_runner = CustomRunner(16)
        images_batch = custom_runner.get_inputs()
    print type(images_batch)
    print images_batch
    square_op = tf.concat(0,images_batch)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    my_config = tf.ConfigProto(
            gpu_options = gpu_options,
            allow_soft_placement=True)
    with tf.Session(config=my_config) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        # start the tensorflow QueueRunner's
        tf.train.start_queue_runners(sess=sess)
        # start our custom queue runner's threads
        coord = tf.train.Coordinator()
        threads = custom_runner.start_threads(sess,coord)
        try:
            for a in xrange(10):
                if coord.should_stop():
                    break
                print sess.run(square_op).shape
        except Exception,e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
        print 'finish'
