




cmake_minimum_required(VERSION 3.10)
project(ImageCompressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



add_executable(new_encoder
    new_encoder/src/main.cpp
    encoder_mod/src/ImageConverter_mod.cpp
    encoder_mod/include/GradientReconstruction_mod.hpp
)

target_include_directories(new_encoder PRIVATE 
    encoder_mod/include 
    encoder_mod/lib 
    util
)