FROM python:3.6-buster

RUN pip install uwsgi

ADD requirements.txt /ds9/requirements.txt
RUN cd /ds9 && pip install -r requirements.txt

ADD . /ds9
RUN cd /ds9 && pip install -e .
