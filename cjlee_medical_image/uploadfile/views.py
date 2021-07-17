from django.shortcuts import render,redirect
from .forms import uploadmodelform
from .models import photo

# Create your views here.
def index(request):
    #pic_obj = models.photo.objects.clear()
    # photos = photo.objects.all()  #查詢所有資料
    preview=photo.objects.last() # 查詢最後一筆資料
    form = uploadmodelform()
    if request.method == "POST":
        form = uploadmodelform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    context = {
        #'photos': photos,
        'form': form,
        'preview':preview
    }
    return render(request, 'test2/second.html', context)    

def pred_demo(request):
    return render(request,'test3/unet_pred_demo.html')

def unet_pred(request):
    return render(request,'test3/unet_pred.html')


from from werkzeug.utils import secure_filename
from keras.applications.mobilenet import preprocess_input as preprocess_1
from keras.applications.resnet import preprocess_input as preprocess_2
from keras.applications.densenet import preprocess_input as preprocess_3
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras_unet_collection import utils
import cv2
from PIL import Image
import shutil

