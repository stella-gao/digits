#!/bin/sh

CUDA_DIR=/opt/cuda-7.5
PATH=$CUDA_DIR/bin:$PATH
LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
THEANO_FLAGS=floatX=float32,device=gpu,force_device=True,nvcc.fastmath=True python $1
