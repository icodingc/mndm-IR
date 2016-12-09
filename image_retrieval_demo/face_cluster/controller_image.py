# coding=utf-8
from django.shortcuts import render
from django.http import HttpResponse
import sys
import StringIO
import PIL.Image as Image

def download_img(request):
    image_name = request.GET.get('image_name',"")

    im = Image.open('static/img/' + image_name) 

    output = StringIO.StringIO()
    im.save(output, 'png')
    
    return HttpResponse(output.getvalue())


