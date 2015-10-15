from django.shortcuts import render
from PIL import Image
from django.http import HttpResponse
import ImageFile
from django.shortcuts import render_to_response
from django.template import loader, Context, RequestContext
from django import forms
import datetime
import Person
import img_classify
import os
import json

def img_rotate(image, angle):
    return image.rotate(angle)

def imageUpload(request):
    print 'imageUpload'
    img_dir = "./tmpl/images/"

    try:
        img = request.FILES.get('img')
        image_name = str(datetime.datetime.now()).replace(' ', '_') + '_' + img.name
        image = Image.open(img)
        img_dir = img_dir + image_name

        if 1:
            #============= image rotate start =====
            if hasattr(image, '_getexif'):
                exifdata = image._getexif()
                if exifdata != None:
                    try:
                        orient = exifdata[0x0112]
                        print 'orient ', orient
                        switch = {
                            '1': lambda: img_rotate(image,0),
                            '2': lambda: img_rotate(image,0),
                            '3': lambda: img_rotate(image,180),
                            '4': lambda: img_rotate(image,180),
                            '5': lambda: img_rotate(image,90),
                            '6': lambda: img_rotate(image,270),
                            '7': lambda: img_rotate(image,270),
                            '8': lambda: img_rotate(image,90)
                        }
                        image = switch[str(orient)]()
                    except:
                        None
            #============= image rotate end =======

        image.save(img_dir)
        print 'image process'
        number, gender, age, char, image_url = face_predict.Classify(img_dir)
        print 'face predict success'
        final = []
        c = 0
        if gender != None:
            final.append(1)
            final.append(image_url)
            if number == 1:
                string = char[0].split(";")
                for s in string:
                    final.append(s)
            else:
                for g in gender:
                    final.append('No.' + str(c+1) + ': ')
                    string = char[c].split(";")
                    for s in string:
                        final.append(s)
                    c += 1
        else:
            final.append(0)
            final.append(image_name)
        print 'test end'
        return HttpResponse(json.dumps(final), content_type='application/jason')

    except:
        final = [0, 0]
        return HttpResponse(json.dumps(final), content_type='application/jason')

def ajax_test1(request):
    per = [1, 2, 3]
    print 'ajax_test 11'
    return HttpResponse(json.dumps(per), content_type='application/jason')

def facePredict(request):
    global face_predict
    face_predict = Person.MyClassify()
    return render(request, 'facePredict.html')

def phoneFacePredict(request):
    global face_predict
    face_predict = Person.MyClassify()
    return render(request, 'phone_facePredict.html')

def imageClassify(request):
    global img_predict
    img_predict = img_classify.imgClassify()
    return render(request, 'images.html')

def Classify(request):
    try:
        print 'try'
        img_predict = img_classify.imgClassify()
        img_dir = './tmpl/images/'
        img = request.FILES.get('img')

        image_name = str(datetime.datetime.now()).replace(' ', '_') + '_' + img.name
        image = Image.open(img)
        img_dir = img_dir + image_name

        if 1:
            #============= image rotate start =====
            if hasattr(image, '_getexif'):
                exifdata = image._getexif()
                if exifdata != None:
                    try:
                        orient = exifdata[0x0112]
                        switch = {
                            '1': lambda: img_rotate(image,0),
                            '2': lambda: img_rotate(image,0),
                            '3': lambda: img_rotate(image,180),
                            '4': lambda: img_rotate(image,180),
                            '5': lambda: img_rotate(image,90),
                            '6': lambda: img_rotate(image,270),
                            '7': lambda: img_rotate(image,270),
                            '8': lambda: img_rotate(image,90)
                        }
                        image = switch[str(orient)]()
                    except:
                        None
            #============= image rotate end =======

        image.save(img_dir)
        results = img_predict.classify(img_dir)

        final = []
        final.append(1)
        final.append(image_name)
        final.append(results)
        print 'return'
        return HttpResponse(json.dumps(final), content_type='application/jason')

    except:
        print 'except'
        final = [0,0]
        return HttpResponse(json.dumps(final), content_type='application/jason')

# Create your views here.
