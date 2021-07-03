#!/bin/bash
poetry install
poetry run python python/export_saved_model.py -o savedmodel/001
docker pull tensorflow/serving
docker run -it --rm -p 8500:8500 -p 8501:8501 \
    -v "$(pwd)/savedmodel:/models/fibonacci" -e MODEL_NAME=fibonacci \
    tensorflow/serving

