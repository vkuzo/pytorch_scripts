#!/bin/bash

# terminate on error
set -e

# hardcode for now
LLVM_PATH=/home/vasiliy/local/llvm-project/build/bin

# compile with clang, passing in the right llvm configs
# clang++ apfloat_example.cpp -o apfloat_example $(llvm-config --cxxflags --ldflags --libs) -lLLVM
$LLVM_PATH/clang++ apfloat_example.cpp -o apfloat_example.out $($LLVM_PATH/llvm-config --cxxflags --ldflags --libs) -lLLVM

# run and capture stdout
./apfloat_example.out 2>&1 | tee results.txt
