# coding=utf-8
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import urllib2, urllib
import random
import traceback
import settings
import requests
import json
import base64
import string
import traceback
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
caffe.set_mode_cpu()
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
#-=====================
def home(request):

    data={}
     
    return render(request, 'home.html',data)

# capture pic image
#http://www.cnblogs.com/linxiyue/p/4038436.html about request.FILES
@csrf_exempt
def recognise(request):
    # only if get 'image' ,run this code?
    f = request.FILES['image']
    #Return a k length list of unique elements chosen from the population sequence
    name=''.join(random.sample(string.ascii_letters + string.digits, 3))+f.name
    # store the file in disk
    with open('static/img/'+name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    image = caffe.io.load_image('static/img/'+name)
    #images,conf_List = cropImageWithSSD(detection_net,image)
    images = cropImageWithSSD(detection_net,image)
    if not isinstance(images,list):
        images = images[0]
    img = scipy.misc.toimage(images[0])
    img.save('static/img/det'+name)
    #print img.shape
    # exmaple.jpg 为需要传给接口的图片，最好缩放到256*256*3 的jpg格式
    # 这句话不是没有道理的。
    # can we 传递 filename
    #content = base64.b64encode(open('static/img/'+name, 'rb').read())
    #content_det = base64.b64encode(open('static/img/det'+name, 'rb').read())

    beg = time.time()
    url = 'http://localhost:9010'
    rpc_client = pyjsonrpc.HttpClient(url,gzipped=True)
    # return a list
    resp = rpc_client.add([name,'det'+name])
    print time.time()-beg
    data={}
    data["image_name"] = name
    data["image_det_name"]='det'+name
    # for src image
    data["result"] = resp[:8]
    # for detection image
    data["result2"] = resp[8:]
    return render(request, 'search.html',data)
    
