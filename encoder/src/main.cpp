#include <iostream>

// This macro tells the library to create the implementation here
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

int main() {
    // Path to your test image in the data folder
    const char* filename = "../data/images/Lenna.png";

    int width, height, channels;
    
    // Load the image: 
    // we ask for 0 'desired_channels' to get the original format
    unsigned char* imgData = stbi_load(filename, &width, &height, &channels, 0);

    if (imgData == NULL) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
        return 1;
    }

    // Precision check: Print the metadata
    std::cout << "--- Image Metadata ---" << std::endl;
    std::cout << "Width:    " << width << " px" << std::endl;
    std::cout << "Height:   " << height << " px" << std::endl;
    std::cout << "Channels: " << channels << " (3=RGB, 4=RGBA)" << std::endl;

    // Example: Accessing the very first pixel (Top-Left)
    // imgData[0] = Red, imgData[1] = Green, imgData[2] = Blue
    std::cout << "Top-left Pixel RGB: (" 
              << (int)imgData[0] << ", " 
              << (int)imgData[1] << ", " 
              << (int)imgData[2] << ")" << std::endl;

    // IMPORTANT: Always free the memory to prevent leaks
    stbi_image_free(imgData);

    return 0;
}