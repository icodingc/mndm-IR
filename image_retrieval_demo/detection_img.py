# coding=utf-8
#input: image
#return: detection image if can.
import numpy as np
import pyjsonrpc
from PIL import Image
import time
import os,sys
import os.path
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import scipy.misc
from progress.bar import Bar 

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
        
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
def cropImageWithSSD(net,image):
# By : wanghaiyi 
    # set net to batch size of 1
    
    # fileName = fileRoot + image
    # imageTmp = caffe.io.load_image(image)
    
    transformed_image = detection_transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']

    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.8]
    target_indices = [i for i, number in enumerate(det_label) if number == 15]

    # 需要的编号
    target_number  = [val for val in top_indices if val in target_indices]
    
    imagesList = []
    confList = []
    if len(target_number) == 0:
        imagesList.append(image)	
        confList.append(1.0)
        return imagesList
    else:
        
        confList = det_conf[target_number]

        top_xmin = det_xmin[target_number]
        top_ymin = det_ymin[target_number]
        top_xmax = det_xmax[target_number]
        top_ymax = det_ymax[target_number]
    
        for i in xrange(confList.shape[0]):
    
    #   i 应该为判断的那个坐标的数据，如果i为空则返回原图片
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
    
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
    
            imageTmp =image[ymin:ymax,xmin:xmax,:]
        
            imagesList.append(imageTmp)
#             plt.imshow(imageTmp)
        
        
        
    return imagesList,confList
#============init======
caffe.set_device(1)
caffe.set_mode_gpu()

detection_model_def = '/home/wangyh/ssd/caffe/models/VOC0712/SSD_300/deploy.prototxt'
detection_model_weights = '/home/wangyh/ssd/caffe/models/VOC0712/SSD_300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
# load PASCAL VOC labels
voc_labelmap_file = '/home/wangyh/ssd/caffe/data/VOC0712/labelmap_voc.prototxt'
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)
detection_net = caffe.Net(detection_model_def,      # defines the structure of the model
                detection_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
detection_transformer = caffe.io.Transformer({'data': detection_net.blobs['data'].data.shape})
detection_transformer.set_transpose('data', (2, 0, 1))
detection_transformer.set_mean('data', np.array([104,117,123])) # mean pixel
detection_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
detection_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


image_resize = 300
detection_net.blobs['data'].reshape(1,3,image_resize,image_resize)
    #TODO get image
root_path = '/mnt/sdc/pub/project_/'
with open(root_path+'rz256.lst','r') as f:names=f.readlines()
bar = Bar('processing',max=288615)
beg = time.time()
for pic in names:
    image = caffe.io.load_image(root_path+'images/'+pic.strip())
    #images,conf_List = cropImageWithSSD(detection_net,image)
    images = cropImageWithSSD(detection_net,image)
    if not isinstance(images,list):images = images[0]
    img = scipy.misc.toimage(images[0])
    #TODO save path
    img.save(root_path+'det_imgs/'+pic.strip())
    bar.next()
bar.finish()
print time.time()-beg
print "done..."
