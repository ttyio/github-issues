FROM nvcr.io/nvidia/tensorrt:20.12-py3
RUN python3 -m pip install tensorflow-gpu==2.4
RUN python3 -m pip install tf2onnx
COPY test.py /test.py
WORKDIR /
