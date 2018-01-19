from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tornado.web
from tornado import gen
import os
import base64
import  time
import json
from timeit import default_timer as timer
from detection import preprocess as prep
from recognition import compare as cmp
import collage as col;
import shutil
import urllib

import threading
from tornado.concurrent import return_future

@return_future
def detectImageAndAssignClasses(obj,step,callback=None):
    image_64_decode = decodeImage(obj)

    print("***************-------Detection Started FROM CAMERA-----*******************",step)

    detected_imgs = prep.init(image_64_decode)
    model = 'model/lightened_cnn/lightened_cnn'
    para = cmp.loadModel(model, 0)

    for index in range(len(detected_imgs)):
        response = cmp.compare_two_face(obj,para, os.environ.get("DETECTED_FACES_STORE_PATH"), detected_imgs[index],step,image_64_decode,createRandomImageId("main_img"))

    print("***************-------Detection And Recognition Complete FROM CAMERA-----*******************", step)

    callback(response)
    #return response

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
    @tornado.gen.coroutine
    def post(self):
        print("1")
        response = yield(detectImageAndAssignClasses(self,1))
        print("2")
        print("**********Camera 1 Output***************",json.dumps(response))
        self.write(json.dumps(response))
        self.finish()


class predict(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        response = yield(detectImageAndAssignClasses(self,2))
        print("**********Camera 2 Output***************",json.dumps(response))
        self.write(json.dumps(response))

class makecollage(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        response = yield(detectImageAndAssignClasses(self,3))
        if(response['status'] != 'error'):
            if not os.path.exists(response['folderPath']):
                response = {}
                response['status'] = 'error'
                response['message'] = 'No Class Found'
            else:
                #col.generateCollage(folderPath=response['folderPath'],width=800,height=250,shuffle=True,classid=response['classId'])
                #shutil.rmtree(response['folderPath'])
                #shutil.rmtree(os.path.join(os.environ.get("DETECTED_FACES_STORE_PATH"),response['classId']))
                s3url = 'http://'+self.request.host+"/collage/"+response['classId']+".jpg"
                response ={}
                response['status'] = 'success'
                response['image_url'] = s3url
        print("**********Camera 3 Output***************",json.dumps(response))
        self.write(json.dumps(response))



