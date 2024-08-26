FROM nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && apt-get -y install sudo

RUN sudo apt-get install -y python3 python3-pip

RUN sudo apt-get install -y python3-venv

RUN mkdir /vectordb && cd /vectordb && \
    python3 -m venv myenv

WORKDIR /vectordb

RUN . myenv/bin/activate && \
    pip install torch transformers uvicorn fastapi

COPY . /vectordb

ENTRYPOINT ["myenv/bin/python3", "testapi.py"]
