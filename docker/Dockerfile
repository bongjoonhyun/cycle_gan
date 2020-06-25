FROM nvidia/cuda:10.2-devel-ubuntu18.04
MAINTAINER bongjoon.hyun@gmail.com

COPY PA2_Skeleton/requirements.txt /cycle_gan

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get clean
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y tmux
RUN apt-get install -y wget
RUN apt-get install -y vim

# Install Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

ENV PATH=/root/anaconda3/bin:$PATH

RUN conda create -y --name py36 python=3.6.10

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/root/anaconda3/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN pip install opencv-python
RUN pip install numpy

# Install requirements.txt
RUN pip install -r /cycle_gan/requirements.txt

# Install PyTorch
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

WORKDIR /cycle_gan