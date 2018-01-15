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
import collage as col;
import shutil



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
    print("detected Img",len(detected_imgs))
    for index in range(len(detected_imgs)):
        print("for",detected_imgs[index])
        response = cmp.compare_two_face(obj,para, os.environ.get("DETECTED_FACES_STORE_PATH"), detected_imgs[index],step,image_64_decode,createRandomImageId("main_img"))
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
        print(response )
        if(response['status'] != 'error'):
            if not os.path.exists(response['folderPath']):
                response = {}
                response['status'] = 'error'
                response['message'] = 'No Class Found'
            else:
                s3url = col.generateCollage(folderPath=response['folderPath'],width=800,height=250,shuffle=True,classid=response['classId'])
                shutil.rmtree(response['folderPath'])
                shutil.rmtree(os.path.join(os.environ.get("DETECTED_FACES_STORE_PATH"),response['classId']))
                response ={}
                response['image_url'] = s3url
        print(response)
        self.write(json.dumps(response))



