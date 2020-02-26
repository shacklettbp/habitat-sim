#!/bin/bash

CMAKE_ARGS="-DUSE_SYSTEM_ASSIMP=Yes -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc" python setup.py develop --with-cuda --headless
