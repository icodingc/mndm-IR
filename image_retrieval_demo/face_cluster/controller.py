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
import urllib2, urllib
import json
import base64
import string
import random
import traceback
import time
import settings
import requests
import base64
import urllib2
import urllib
import json
import random
from sklearn.neighbors import KDTree
import numpy as np

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
        # exmaple.jpg 为需要传给接口的图片，最好缩放到256*256*3 的jpg格式
        #content = base64.b64encode(open('static/img/'+name, 'rb').read())
        #content_data = urllib.urlencode({'pic':content})
        
        #url = 'http://localhost:9010/service-similar-pic'
        #req = urllib2.Request(url, data = content_data)

        #resp = urllib2.urlopen(req).read()
        #resp = json.loads(resp)
        data={}
        data["image_name"] = get_pic_name
        data['result'] = []
        cur_list = [1,2,4,5,7,12,18,15]
        np.random.shuffle(cur_list)
        for i in cur_list:
    	   data['result'].append(str(i)+'_t.jpg')
        return render(request, 'search.html',data)