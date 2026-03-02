




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


add_executable(decoder_mod
    decoder_mod/src/main_decoder_mod.cpp
    encoder_mod/src/ImageConverter_mod.cpp
)

target_include_directories(decoder_mod PRIVATE 
    decoder_mod/include 
    decoder_mod/lib 
    util
)


add_executable(channel_encoder
    channel_encoder/src/main.cpp
    encoder_mod/src/ImageConverter_mod.cpp
)

target_include_directories(channel_encoder PRIVATE 
    encoder_mod/include 
    encoder_mod/lib 
    util
)


add_executable(channel_decoder
    channel_decoder/src/main.cpp
    encoder_mod/src/ImageConverter_mod.cpp
    encoder_mod/include/GradientReconstruction_mod.hpp
)

target_include_directories(channel_decoder PRIVATE 
    decoder_mod/include 
    decoder_mod/lib 
    encoder_mod/include
    util
)