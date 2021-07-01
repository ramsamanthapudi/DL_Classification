FROM python:3.8-slim-buster
WORKDIR /puvvulu
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["python3","-m","flask","run","--host=0.0.0.0"]
