from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tornado.ioloop
import tornado.web
import os

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

class DetectHandler(tornado.web.RequestHandler):
    def get(self):
        print("detect start")
        img = self.get_argument('img', True)
        print(img)
        from detection import preprocess as prep
        prep.init(img)
        self.write('detection done')

class RecognizeHandler(tornado.web.RequestHandler):
    def get(self):
        from recognition import compare as cmp
        model='model/lightened_cnn/lightened_cnn'
        para = cmp.loadModel(model,0)
        print ("similar",cmp.compare_two_face(para,"/Users/admin/Work/Face_Recognition/output","151497675637119.jpg","151377336968480.jpg"))
        #cmp.compare_two_face(para, "/Users/admin/Work/facerecognition/out", "151426961291577.jpg")
        print("detect start")
        self.write("recognize")
    def post(self):
        #self.write(self.request.body)
        img_ba = self.get_argument("img", default=None, strip=False)
        self.write(img_ba)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/detect", DetectHandler),
        (r"/jio/recognition/v1/compareface", RecognizeHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("server started at port 8888")
    tornado.ioloop.IOLoop.current().start()
