FROM nvidia/cuda:10.2-devel-ubuntu18.04
MAINTAINER bongjoon.hyun@gmail.com

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

RUN pip install --upgrade pip

RUN pip install opencv-python
RUN pip install numpy

# Install requirements.txt
RUN pip install attrs==19.3.0
RUN pip install certifi==2020.4.5.1
RUN pip install cffi==1.14.0
RUN pip install chardet==3.0.4
RUN pip install click==7.1.2
RUN pip install clickclick==1.2.2
RUN pip install connexion==2.7.0
RUN pip install cycler==0.10.0
RUN pip install decorator==4.4.2
RUN pip install dominate==2.5.1
RUN pip install Flask==1.1.2
RUN pip install future==0.18.2
RUN pip install idna==2.9
RUN pip install imageio==2.8.0
RUN pip install importlib-metadata==1.6.1
RUN pip install inflection==0.5.0
RUN pip install itsdangerous==1.1.0
RUN pip install Jinja2==2.11.2
RUN pip install jsonpatch==1.25
RUN pip install jsonpointer==2.0
RUN pip install jsonschema==3.2.0
RUN pip install kiwisolver==1.2.0
RUN pip install MarkupSafe==1.1.1
RUN pip install matplotlib==3.2.1
RUN conda install mkl-service==2.3.0
RUN pip install networkx==2.4
RUN pip install numpy==1.18.4
RUN pip install olefile==0.46
RUN pip install openapi-spec-validator==0.2.8
RUN pip install Pillow==7.1.2
RUN pip install pycparser==2.20
RUN pip install pyparsing==2.4.7
RUN pip install pyrsistent==0.16.0
RUN pip install python-dateutil==2.8.1
RUN pip install PyWavelets==1.1.1
RUN pip install PyYAML==5.3.1
RUN pip install pyzmq==19.0.1
RUN pip install requests==2.23.0
RUN pip install scikit-image==0.17.2
RUN pip install scipy==1.1.0
RUN pip install six==1.15.0
#RUN pip install TBB==0.1
RUN pip install tifffile==2020.6.3
#RUN pip install torch==1.5.0
#RUN pip install torchfile==0.1.0
#RUN pip install torchvision==0.6.0
RUN pip install tornado==6.0.4
RUN pip install tqdm==4.46.1
RUN pip install urllib3==1.25.9
RUN pip install visdom==0.1.8.9
RUN pip install websocket-client==0.57.0
RUN pip install Werkzeug==1.0.1
RUN pip install zipp==3.1.0

# Install PyTorch
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

WORKDIR /cycle_gan
