# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel as base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y software-properties-common && apt-get install -y git-lfs nano
# RUN pip install uv
RUN python3 -m pip install transformers==4.45.0 tokenizers==0.20.0
RUN python3 -m pip install wandb evaluate tensorboard accelerate sentence-transformers notebook scipy soundfile librosa
RUN python3 -m pip install git+https://github.com/huggingface/datasets
RUN python3 -m pip install flash-attn --no-build-isolation

RUN git config --global credential.helper software
RUN git config --global --add safe.directory /exp/


WORKDIR /exp

CMD ["bash"]