#!/bin/bash

. ./setup.sh

eval $TEST_CXX OMP_NUM_THREADS=1 ./omp_gpu_monitor
