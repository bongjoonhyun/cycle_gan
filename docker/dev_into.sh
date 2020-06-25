#!/bin/bash

xhost +

docker run \
  -it \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  -v ~/cycle_gan/PA2_Skeleton:/cycle_gan \
  bongjoonhyun/cycle_gan:latest \
  /bin/bash

