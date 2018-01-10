from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tornado.ioloop
import tornado.web
#import datalayer as dtl
import os
import base64
import  numpy as np
import cv2
from detection import preprocess as prep
from recognition import compare as cmp

class RecognizeByCompareHandler(tornado.web.RequestHandler):
    def get(self):
        from recognition import compare as cmp
        model='model/lightened_cnn/lightened_cnn'
        para = cmp.loadModel(model,0)
        print ("similar",cmp.compare_two_face(para,"/Users/admin/Work/Face_Recognition/output","151497675637119.jpg","151377336968480.jpg"))
        #cmp.compare_two_face(para, "/Users/admin/Work/facerecognition/out", "151426961291577.jpg")
        print("detect start")
        self.write("recognize")
    def post(self):
        image_stream = self.request.files['img'][0]
        image_64_encode = base64.encodestring(image_stream['body'])
        image_64_decode = base64.decodestring(image_64_encode)
        print("-------Detection Started-----")
        detected_imgs = prep.init(image_64_decode)
        print("-------Detection Complete-----")
        print("No. of Detected Images ",len(detected_imgs))
        print("-------Recognition Started-----")
        model = 'model/lightened_cnn/lightened_cnn'
        para = cmp.loadModel(model, 0)
        for index in range(len(detected_imgs)):
            inputPath = '/Users/admin/Work/Face_Recognition/out'
            print("Recognition started for Face ",index+1 )
            cmp.compare_two_face(para, inputPath,detected_imgs[index])
        print("-------Recognition Complete-----")