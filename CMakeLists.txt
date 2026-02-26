cmake_minimum_required(VERSION 3.10)
project(ImageCompressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(encoder/include)
include_directories(encoder/lib)

# Encoder executable
add_executable(encoder 
    encoder/src/main.cpp 
    encoder/src/ImageConverter.cpp
    encoder/src/FileHandler.cpp
)

# Decoder executable
add_executable(decoder 
    decoder/src/main_decoder.cpp
)
target_include_directories(encoder PRIVATE encoder/include encoder/lib util)
target_include_directories(decoder PRIVATE decoder/include encoder/include decoder/lib util)