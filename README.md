# Vectorizer
This project aims to build an optimized inference in a containerized environment.

## Setup
Perform the following steps to run the server inside container
```
docker build -t hawkeye/vectorizer:v1 .
docker run -it  --rm --gpus all <image>
```