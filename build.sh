#!/bin/bash
if [ ! -d "./build" ]; then
  mkdir build
fi

rm -rf ./build/*

# BUILD
cd build
cmake ..
make -j12