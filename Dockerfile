# AnyView-DVS container: CUDA + PyTorch base plus the release's pip requirements.
#
# Build (from the release root):   docker build -t anyview .
# Run: see README.md, Installation (the checkout is mounted at /workspace/anyview).
# The image name used throughout the docs is "anyview".
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/anyview
COPY requirements.txt /workspace/anyview/requirements.txt
RUN pip install --no-cache-dir -r /workspace/anyview/requirements.txt
COPY . /workspace/anyview
ENV PYTHONPATH=/workspace/anyview
