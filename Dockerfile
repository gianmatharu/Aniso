FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .
EXPOSE 8050

CMD [ "python3", "run_app.py"]
