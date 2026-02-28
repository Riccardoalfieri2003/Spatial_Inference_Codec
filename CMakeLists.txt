




cmake_minimum_required(VERSION 3.10)
project(ImageCompressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encoder mod executable
add_executable(encoder_mod
    encoder_mod/src/main.cpp
    encoder_mod/src/ImageConverter_mod.cpp
)

target_include_directories(encoder_mod PRIVATE 
    encoder_mod/include 
    encoder_mod/lib 
    util
)