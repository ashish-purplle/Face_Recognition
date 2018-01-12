from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tornado.ioloop
import tornado.web
#import datalayer as dtl
import os
import base64
import  numpy as np
import cv2
import  time
import json
from timeit import default_timer as timer
from detection import preprocess as prep
from recognition import compare as cmp
import servicelayer as sl

def detectImageAndAssignClasses(obj,step):
    image_64_decode = decodeImage(obj)
    print("-------Detection Started-----")
    start_detect = timer()
    detected_imgs = prep.init(image_64_decode)
    end_detect = timer()
    print("-------Detection Complete-----")
    print("-------Total time in Detection-----",(end_detect-start_detect))
    print("-------Recognition Started-----")
    rec_start = timer()
    model = 'model/lightened_cnn/lightened_cnn'
    para = cmp.loadModel(model, 0)
    for index in range(len(detected_imgs)):
        inputPath = '/Users/admin/Work/Face_Recognition/out'
        response = cmp.compare_two_face(obj,para, inputPath, detected_imgs[index],step,image_64_decode,createRandomImageId("main_img"))
    rec_end = timer()
    print("-------Recognition Complete-----")
    print("-------Total time in Recognition-----",(rec_end-rec_start))
    return response

def decodeImage(obj):
    image_stream = obj.request.files['img'][0]
    image_64_encode = base64.encodestring(image_stream['body'])
    image_64_decode = base64.decodestring(image_64_encode)
    return  image_64_decode

def createRandomFolderId(param):
    return param.format(int(time.time() * 100000))

def createRandomImageId(param):
    return param+"_{}.jpeg".format(int(time.time() * 100000))

class savefaces(tornado.web.RequestHandler):
    def post(self):
        response = detectImageAndAssignClasses(self,1)
        print(response)
        print(json.dumps(response))
        self.write(json.dumps(response))


class predict(tornado.web.RequestHandler):
    def post(self):
        response = detectImageAndAssignClasses(self,2)
        self.write(json.dumps(response))

class makecollage(tornado.web.RequestHandler):
    def post(self):
        response = detectImageAndAssignClasses(self,3)


