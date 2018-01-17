
FROM chetandev/ubuntu16-python2-opencv3-dlib

RUN pip install --upgrade pip
RUN git clone https://github.com/ashish-purplle/Face_Recognition.git \
&& pipreqs /Face_Recognition \
&& cat /Face_Recognition/requirements.txt \
&& pip install -r /Face_Recognition/requirements.txt

RUN wget  -O /Face_Recognition/model.zip  https://www.dropbox.com/s/0lh12ljmtb7qb4c/model.zip \
&& wget  -O /Face_Recognition/detection/mxnet-face-fr50-symbol.json  https://www.dropbox.com/s/i6a51twcr8mp95m/mxnet-face-fr50-symbol.json \
&& wget  -O /Face_Recognition/detection/mxnet-face-fr50-0000.params  https://www.dropbox.com/s/ec7dfffzbbfrzis/mxnet-face-fr50-0000.params \
&& wget  -O /Face_Recognition/.env https://www.dropbox.com/s/j4jbqjvnhwm614h/.env \
&& unzip /Face_Recognition/model.zip  -d /Face_Recognition/

WORKDIR  /Face_Recognition
CMD python app.py

EXPOSE 8888