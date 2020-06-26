#!/bin/bash

xhost +

nvidia-docker run \
  -it \
  --runtime=nvidia \
  --ipc=host \
  -v ~/cycle_gan/PA2_Skeleton:/cycle_gan \
  bongjoonhyun/cycle_gan:latest \
  /bin/bash

