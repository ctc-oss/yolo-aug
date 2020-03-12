FROM python:3.6

ENV PYTHONPATH=/opt/src

WORKDIR /opt/src
COPY . /opt/src

RUN python -m pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-m", "yolo_aug"]
