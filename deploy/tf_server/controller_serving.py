# coding=utf-8
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import urllib2, urllib
import json
import base64
import string
import random
import traceback
import time
import settings
import requests
import traceback
import time
import settings
import requests
#for tensorflow serving
from grpc.beta import implementations
import tensorflow as tf
import predict_pb2
import prediction_service_pb2
import numpy as np
#for annoy
from annoy import AnnoyIndex

host, port = 'localhost:7004'.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
names = np.load('data/name.npy')
t = AnnoyIndex(512)
t.load('model/inshop.ann')
print 'Load ann...\n'


def home(request):
    data={}
    return render(request, 'home.html',data)

# capture pic image
#http://www.cnblogs.com/linxiyue/p/4038436.html about request.FILES
@csrf_exempt
def recognise(request):
    try:
        f = request.FILES['image']
        #Return a k length list of unique elements chosen from the population sequence
        get_pic_name= f.name + ''.join(random.sample(string.ascii_letters + string.digits, 3))
        # store the file in disk
        with open('static/img/'+get_pic_name, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        print '[INFO]:',request.method
        print '[INFO]:get from upload'
    except:
        get_pic_name = request.GET['image_name']
        print '[INFO]:',request.method
        print '[INFO]:get from click image'
    finally:
        print '[INFO]:Get',get_pic_name,'\n'

        pic_data = open('static/img/'+get_pic_name, 'rb').read()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception'
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(pic_data, shape=[1]))
        result = stub.Predict(request, 10.0)  # 10 secs timeout
        feat = result.outputs['feats'].float_val
        #TODO annoy get result
        idxs = t.get_nns_by_vector(feat,8) 

        data={}
        data["image_name"] = get_pic_name
        data['result'] = names[idxs]
        return render(request, 'search.html',data)
