import logging
from logging import handlers

class AppFilter(logging.Filter):

    def filter(self, record):
        try:
            record.userid = record.userid
        except Exception as e:
            record.userid = ""
        try:
            record.devicekey = record.devicekey
        except Exception:
            record.devicekey = ""
        try:
            record.hash = record.hash
        except Exception:
            record.hash = ""

        return True


logger = logging.getLogger('__logger__')
logger.setLevel(logging.INFO)
logger.addFilter(AppFilter())


<<<<<<< HEAD
fh = logging.handlers.RotatingFileHandler(filename="image-recognition.log", mode='a', maxBytes=5*1024*1024,
=======
fh = logging.handlers.RotatingFileHandler(filename="text-classification.log", mode='a', maxBytes=5*1024*1024,
>>>>>>> 242ded8a5099a7b3c1af2630bd8e980c71b6c9f6
                                 backupCount=20, encoding=None, delay=0)
fh.setLevel(logging.INFO)


formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(userid)s] [%(devicekey)s] [%(hash)s] [%(message)s] ',datefmt='%d-%m-%Y %I:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)




