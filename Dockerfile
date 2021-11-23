FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get -y upgrade && \
    apt-get -y install \
    python2.7 \
    python3.7 \
    python3-pip \
    nano \
    wget \
    curl \
    git \
    unzip \
    sudo \
    zsh \
    gcc \
    g++ \
    make \
    libsm6 libxrender1

COPY requirements.txt .

RUN pip3 install -U pip && pip install -r requirements.txt