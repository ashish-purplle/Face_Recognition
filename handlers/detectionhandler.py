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

class DetectionHandler(tornado.web.RequestHandler):
    def get(self):
        print("detect start")
        img = self.get_argument('img', True)
        print(img)
        from detection import preprocess as prep
        prep.init(img)
        self.write('detection done')