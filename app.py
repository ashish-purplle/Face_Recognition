from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import tornado.ioloop
import tornado.web

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
    server = tornado.httpserver.HTTPServer(app)
    server.bind(8888)
    server.start(8)  # autodetect number of cores and fork a process for each
    print("server started at port 8888")
    tornado.ioloop.IOLoop.instance().start()


    #app = make_app()
    # app.listen(8888)
    # server.bind(8888)
    # server.start(0)  # Forks multiple sub-processes
    # IOLoop.current().start()

    # tornado.ioloop.IOLoop.current().start()
