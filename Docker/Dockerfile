FROM openvino/ubuntu22_runtime:2024.6.0

USER root
RUN apt -y update && apt -y install libopencv-dev

RUN python3 -m pip install -U pip && \
    pip3 install -U setuptools wheel && \
    pip3 install opencv-python

RUN useradd -ms /usr/bin/bash ovuser
USER ovuser

WORKDIR /app
COPY . /app

CMD [ "python3", "openvino_app.py"]
