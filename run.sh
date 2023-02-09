#!/bin/bash
# env LD_LIBRARY_PATH=/mnt/vepfs/users/yaosk/code/deepmd_on_pytorch_dev/source/build/lib:/mnt/vepfs/users/yaosk/code/deepmd_on_pytorch_dev/source/build/lib/src/cuda/ python3 deepmd_pt/se_a.py
export LD_LIBRARY_PATH="/mnt/vepfs/users/yaosk/code/deepmd_on_pytorch_dev/source/build/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/vepfs/users/yaosk/code/deepmd_on_pytorch_dev/source/build/lib/src/cuda/:$LD_LIBRARY_PATH"
env USE_CUDA=1 python deepmd_pt/main.py train tests/water/se_e2_a.json