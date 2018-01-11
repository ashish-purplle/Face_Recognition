from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tornado.ioloop
import tornado.web
import os
import base64
import  numpy as np
import cv2
from detection import preprocess as prep
from recognition import compare as cmp
from handlers import recognizebycomparehandler as rbch
from handlers import recognizebymodelhandler as rbmh
from handlers import detectionhandler as dh

def make_app():
    return tornado.web.Application([
        (r"/detect", dh.DetectionHandler),
        (r"/jio/recognition/compareface/savefaces/v1", rbch.savefaces),
        (r"/jio/recognition/compareface/predict/v1", rbch.predict),
        (r"/jio/recognition/compareface/makecollage/v1", rbch.makecollage),
        (r"/jio/recognition/v1/searchface", rbmh.RecognizeByModelHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("server started at port 8888")
    tornado.ioloop.IOLoop.current().start()
