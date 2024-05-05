#!/bin/sh


# This script takes two OPTIONAL environment variables:
# 1) LIBAIMS_PATH will be added to LD_LIBRARY_PATH
# 2) TESTING_PYTHON should be a command for python3 interpreter used for tests launch
# Particular behaviour depends on test.py in specific testcase

echo $LIBAIMS_PATH
TESTING_PYTHON=${TESTING_PYTHON:-'python3'}
echo $TESTING_PYTHON
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBAIMS_PATH
ldd $LIBAIMS_PATH/lib*.so
mpirun -n 2 $TESTING_PYTHON -u test.py > stdout.log 2>&1

