import time,sys
import threading
import time
import numpy as np
import tensorflow as tf
#import scipy
#from scipy.spatial.distance import cosine
#from sklearn.metrics.pairwise import cosine_similarity as cosine
#from sklearn.preprocessing import normalize
from preprocessing import vgg_preprocessing as vp

file_name = '/raid/home/zhangxuesen/data/In-shop-IR/utils/train.lst'
base_dir = '/raid/home/zhangxuesen/data/In-shop-IR/'
name_file_ = base_dir+'utils/old_/filename.npy'
feat_file_ = base_dir+'utils/old_/feature_ap.npy'

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
#name2feat=get_names2feat(name_file_,feat_file_)
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
item_lsts = vector.keys()
#TODO make sure that .keys() equal
item_lsts = sorted(item_lsts)
np.save('./ops/fixed_item',item_lsts)

item_cnt = len(item_lsts)
item_idxs = list(xrange(item_cnt))
item2idx = dict(zip(item_lsts,item_idxs))
##Item_mean_vec
#Item_mean={}
#for item in item_lsts:
#	image_all = vector[item]
#	vector_mean = get_mean_feat(image_all)
#	Item_mean[item]=vector_mean
##Item_sim_matrix,In this why,[may be sampling positive!!!]
#Item_sim_matrix = []
#for item1 in item_lsts:
	#TODO get similarity not distance
#	cur_sim_v = np.array([np.squeeze(cosine(Item_mean(item1).reshape(1,-1),Item_mean(other).reshape(1,-1))) for other in item_lsts])
	#TODO Simarity ==> prob
#	cur_prob_v = cur_sim_v/np.sum(cur_sim_v)
#	Item_sim_matrix.append(cur_prob_v)
#np.save('sim_matrix',np.array(Item_sim_matrix))
Item_sim_matrix=np.load('./ops/sim_matrix.npy')
print Item_sim_matrix.shape
end = time.time()
print 'Processing Time: ',end-start
#######################
def data_iter():
    file_lists = namelst
    count_file = len(file_lists)
    file_lists = np.array(file_lists)
    while True:
        idxs = np.arange(0,count_file)
        np.random.shuffle(idxs)
        shuf_lists = file_lists[idxs]

        for batch_idx in xrange(0,len(shuf_lists)):
            img_a = shuf_lists[batch_idx].strip()
            #TODO should read to string.
            cur_fg = img_a.split('/')[-2]
            img_p = np.random.choice(vector[cur_fg])
            #TODO sampling negs according similarity
            negs = np.random.choice(item_lsts,6,p=Item_sim_matrix[item2idx[cur_fg]])
            while cur_fg in negs:negs = np.random.choice(item_lsts,6,p=Item_sim_matrix[item2idx[cur_fg]])
            a = open(base_dir+img_a,'r').read()
            p = open(base_dir+img_p,'r').read()
            n1 = open(base_dir+np.random.choice(vector[negs[0]]),'r').read()
            n2 = open(base_dir+np.random.choice(vector[negs[1]]),'r').read()
            n3 = open(base_dir+np.random.choice(vector[negs[2]]),'r').read()
            n4 = open(base_dir+np.random.choice(vector[negs[3]]),'r').read()
            n5 = open(base_dir+np.random.choice(vector[negs[4]]),'r').read()
            n6 = open(base_dir+np.random.choice(vector[negs[5]]),'r').read()
            #TODO Item2idx and idx to P
            pro_Q=[Item_sim_matrix[item2idx[cur_fg]][item2idx[itm]] for itm in negs]
            yield a,p,n1,n2,n3,n4,n5,n6,pro_Q

class CustomRunner(object):
    def __init__(self,batch_size=10,shuffle=False):
        self.batch_size = batch_size
        self.image_a = tf.placeholder(tf.string,[])
        self.image_p = tf.placeholder(tf.string,[])
        self.image_n1 = tf.placeholder(tf.string,[])
        self.image_n2 = tf.placeholder(tf.string,[])
        self.image_n3 = tf.placeholder(tf.string,[])
        self.image_n4 = tf.placeholder(tf.string,[])
        self.image_n5 = tf.placeholder(tf.string,[])
        self.image_n6 = tf.placeholder(tf.string,[])
        self.prob = tf.placeholder(tf.float32,[6])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        c_sp=[224,224,3]
        c_shapes=[c_sp]*8+[[6]]
        c_dtypes=[tf.float32]*(8+1)
        if shuffle:
            self.queue = tf.RandomShuffleQueue(shapes=c_shapes,
                                  dtypes=c_dtypes,
                                  capacity=30,
                                  min_after_dequeue=14)
        else:
            self.queue = tf.FIFOQueue(shapes=c_shapes,
                                  dtypes=c_dtypes,
                                  capacity=30)
        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        img_a = self.transfer(self.image_a,'anchor')
        img_p = self.transfer(self.image_p,'positive')
        img_n1 = self.transfer(self.image_n1,'neg1')
        img_n2 = self.transfer(self.image_n2,'neg2')
        img_n3 = self.transfer(self.image_n3,'neg3')
        img_n4 = self.transfer(self.image_n4,'neg4')
        img_n5 = self.transfer(self.image_n5,'neg5')
        img_n6 = self.transfer(self.image_n6,'neg6')
        features=[img_a,img_p,img_n1,img_n2,img_n3,img_n4,img_n5,img_n6,self.prob]
        self.enqueue_op = self.queue.enqueue_many(features)
    def transfer(self,input_string,prefix):
    	img = tf.cast(tf.image.decode_jpeg(input_string,channels=3),tf.float32)
    	return tf.expand_dims(vp.preprocess_for_train(img,224,224,prefix=prefix),0)
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
        for a,p,n1,n2,n3,n4,n5,n6,Q in data_iter():
            feed_d = {self.image_a:a,
                      self.image_p:p,
                      self.image_n1:n1,
                      self.image_n2:n2,
                      self.image_n3:n3,
                      self.image_n4:n4,
                      self.image_n5:n5,
                      self.image_n6:n6,
                      self.prob:Q,
                      }
            sess.run(self.enqueue_op, feed_dict=feed_d)
            if coord.should_stop():
                break

    def start_threads(self, sess,coord,n_threads=2):
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
        custom_runner = CustomRunner(10)
        images_batch = custom_runner.get_inputs()
    square_op = tf.concat(0,images_batch)
    with tf.Session() as sess:
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
