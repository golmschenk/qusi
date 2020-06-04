#!/bin/bash

export march=armv8-a
export ARROW_ARMV8_ARCH=armv8-a

sudo apt-get install -y libjemalloc-dev libboost-dev libboost-filesystem-dev libboost-system-dev libboost-regex-dev autoconf flex bison virtualenv
sudo apt-get install -y openssl 
pip3 install hypothesis-5.1.0 sortedcontainers-2.1.0
sudo apt-get install -y cmake
pip3 install cmake
sudo apt-get install -y clang-tidy clang-format
sudo apt-get install -y zip unzip


##snappy
sudo wget https://github.com/google/snappy/archive/master.zip
sudo mkdir snappy-src
chmod -R a+rwx *
unzip master.zip -d snappy-src 
sudo chmod +777 --recursive snappy-src
cd snappy-src/snappy-master
sudo mkdir build
chmod -R a+rwx *
cd build 
cmake ../
make
cd ~/

sudo wget https://github.com/google/double-conversion/archive/master.zip -O double-conversion-master.zip
sudo mkdir double-conversion-src
chmod -R a+rwx *
unzip double-conversion-master.zip -d double-conversion-src 
sudo chmod +777 --recursive double-conversion-src
cd double-conversion-src/double-conversion-master
cmake . -DBUILD_TESTING=ON
make
test/cctest/cctest --list | tr -d '<' | xargs test/cctest/cctest
cd ~/


pip3 install boost brotli 

#gflags
sudo apt-get install -y libgflags-dev


#glog
sudo wget https://github.com/google/glog/archive/master.zip -O glog-master.zip
sudo mkdir glog-src
chmod -R a+rwx *
unzip glog-master.zip -d glog-src
sudo chmod +777 --recursive glog-src
cd glog-src/glog-master
cmake -H. -Bbuild -G "Unix Makefiles"
cmake --build build
cmake --build build --target test
cmake --build build --target install
cd ..
cd ..

cd ~/
#gtest
sudo apt-get install -y libgtest-dev

pip3 install thrift
#-0.13.0

##benchmark
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
chmod -R a+rwx *
cd benchmark
mkdir build && cd build
cmake ../
make
cd ~/

pip3 install rapidjson


#zlib
sudo apt-get install -y zlib1g-dev bzip2
pip3 install lz4 zstd 

#re
sudo mkdir re2-src
sudo wget https://github.com/google/re2/archive/master.zip -O re2-src.zip
chmod -R a+rwx *
unzip re2-src.zip -d re2-src
sudo chmod +777 --recursive re2-src
cd re2-src/re2-master
make
make test
make testinstall
make install
cd ~/
#libffi-dev 13.2.1-8 is required to install cffi
sudo apt-get install -y libffi-dev
pip3 install cffi
#==1.13.2

#py.org cares is inside of pycares as of 1.0.0
pip3 install pycares
#==3.1.0

#grpc
sudo apt-get install -y clang libc++-dev
sudo apt-get install -y build-essential autoconf libtool pkg-config
sudo apt-get install -y curl
git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc
cd grpc
git submodule update --init
make
cd ..
cd ~/
sudo pip3 install -U pip

sudo mkdir ~/arrow
#sudo chmod -R a+rwx *
cd arrow
pip3 install -U thrift #now at 0.14.0
pip3 install retrying #new from RAMjET's May 2020 Version
pip3 install pytest psutil # new to arrow 0.17.0
git clone https://github.com/apache/arrow.git #now at 0.17.1
chmod -R a+rwx *
cd ~/arrow/arrow/cpp
sudo mkdir release
#sudo mkdir debug
chmod -R a+rwx *
sudo sed -i 's|set(ARROW_ALTIVEC_FLAG "-maltivec")|set(ARROW_ALTIVEC_FLAG "-mglibc")|g' ~/arrow/arrow/cpp/cmake_modules/SetupCxxFlags.cmake
sudo sed -i 's|set(ARROW_ARMV8_ARCH_FLAG "-march=${ARROW_ARMV8_ARCH}")|set(ARROW_ARMV8_ARCH_FLAG "-march=armv8-a")|g' ~/arrow/arrow/cpp/cmake_modules/SetupCxxFlags.cmake
sudo sed -i 's|set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")|set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")|g' ~/arrow/arrow/cpp/cmake_modules/SetupCxxFlags.cmake

export march=armv8-a
export ARROW_ARMV8_ARCH=armv8-a
export CC=gcc
export CXX=g++
export ARROW_BUILD_TYPE=release
export ARROW_HOME=~/arrow/arrow/cpp/release
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH

sudo cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME -DCMAKE_INSTALL_LIBDIR=$ARROW_HOME/lib -DARROW_PYTHON=ON -DARROW_CUDA=OFF -DARROW_BUILD_EXAMPLES=OFF -DARROW_TENSORFLOW=OFF -DARROW_BUILD_INTEGRATION=ON -DARROW_BUILD_TESTS=OFF -DPYTHON_EXECUTABLE=/usr/bin/python3
make -j4
make install 
cd ..
sudo chmod -R a+rwx *
cd ~/arrow/arrow/python
sudo sed -i 's|numpy==1.14.5|numpy==1.16.1|g' requirements-wheel-build.txt
pip3 install -r requirements-wheel-build.txt
sudo sed -i 's|numpy==1.14.5|numpy==1.16.1|g' requirements-wheel-test.txt
pip3 install -r requirements-wheel-test.txt
export ARROW_HOME=~/arrow/arrow/cpp/release
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH
export march=armv8-a
export ARROW_ARMV8_ARCH=armv8-a
sudo -E python3 setup.py install
echo " "
echo " "
echo " (Check above for errors) "
echo " "
echo " "
export ARROW_HOME=~/arrow/arrow/cpp/release
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH
sudo ldconfig
sudo cp ~/arrow/arrow/cpp/release/lib/* /usr/lib/python3.6/config-3.6m-aarch64-linux-gnu
sudo cp ~/arrow/arrow/cpp/release/lib/* /usr/local/lib
export ARROW_HOME=~/arrow/arrow/cpp/release
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH
sudo ldconfig
echo " "
echo " "
echo "                     Check for Errors"
echo " (Three warnings about items not being symlinks is fine) "
echo " "
echo " "

###
#
# CCX FLAGS cause problems because the L4T doesn't have an arch property somewhere within
#  it set correctly.  No one has speculated as to which one. 
#
#  The most helpful document from apache is: 
#    https://arrow.apache.org/docs/developers/cpp/building.html#individual-dependency-resolution
#
#  The two "solutions" come from these two arrow issues.  No solution is actually written.
#    https://github.com/apache/arrow/issues/1125
#    https://github.com/apache/arrow/issues/6049
#
#  However, the solution is to literally change some things which make the flags happen. 
#   Edit ~/repos/arrow/cpp/cmake_modules/SetupCxxFlags.cmake and DefineOptions.cmake
#   by removing the routines setting these flags and exporting the arch for it. 
#
#  In DefineOptions.cmake:
#    Line 108: This gives us the ARMv8 options that are acceptable to
#                 ARROW_ARMV8_ARCH = "armv8-a" or "armv8-a+crc+crypto"
#
#  In SetupCxxFlags.cmake:
#    Line  54: This was a suggested change from github
#                 from: set(ARROW_ALTIVEC_FLAG "-maltivec")
#                 to:   set(ARROW_ALTIVEC_FLAG "-mglibc")
#    Line  58: This is what ends up not working... ever. ever. 
#                 Chesterton's Fence.  Don't worry about the courseness of this solution.
#                 from: set(ARROW_ARMV8_ARCH_FLAG "-march=${ARROW_ARMV8_ARCH}")
#                 to:   set(ARROW_ARMV8_ARCH_FLAG "-march=armv8-a")
#    Line 100: This was a suggestion from github
#                 from: set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")
#                 to:   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")
#
#  In the Python Setup Requirement-*.txt, it really wants us to use numpy==1.14.5.
#   If you let it install this, it will break the rest of RAMjET,
#   specifically astropy 4.1.dev532+gde9db8be6 and tensorflow-gpu 2.0.0+nv19.11.tf2
#
#  In requirements-wheel-build.txt, change this line: numpy==1.14.5; python_version < "3.8"
#  In requirements-wheel-test.txt,  change this line: numpy==1.14.5; python_version < "3.8"
#
#  The change is to make numpy==1.16.1
#
#
###
