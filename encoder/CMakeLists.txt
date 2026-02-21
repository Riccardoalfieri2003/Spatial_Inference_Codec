cmake_minimum_required(VERSION 3.10)
project(ImageCompressor)

# Use C++17 for modern features and efficiency
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake where your .h files are
include_directories(include)
include_directories(lib)

# Create the executable "encoder" from your source file
add_executable(encoder src/main.cpp src/ImageConverter.cpp)

# Optional: If you use math libraries later, you'd link them here