# Vectorizer

This project aims to build an optimized inference in a containerized environment. It utilizes the `all-mpnet-base-v2` sentence vectorization model, which can be found [here](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).

The base image used for running CUDA and PyTorch is `nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04`.

## Setup

Follow these steps to run the server inside a container:

1. Build the Docker image:
    ```sh
    docker build -t hawkeye/vectorizer:v1 .
    ```

2. Run the Docker container:
    ```sh
    docker run -it --rm --gpus all hawkeye/vectorizer:v1
    ```