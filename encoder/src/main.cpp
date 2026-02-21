#include <iostream>
#include <vector>
#include "ImageConverter.hpp" // Include our new converter

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Note: CMake handles the path now!

int main() {
    const char* filename = "../data/images/Lenna.png";
    int width, height, channels;
    unsigned char* imgData = stbi_load(filename, &width, &height, &channels, 0);

    if (!imgData) return 1;

    // This loop visits every single pixel once
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            
            // Calculate the position in the raw 1D array
            int pixelOffset = (y * width + x) * channels;

            // 1. Convert to Lab (Reading from memory)
            LabPixel lab = ImageConverter::convertPixelRGBtoLab(
                imgData[pixelOffset], 
                imgData[pixelOffset + 1], 
                imgData[pixelOffset + 2]
            );

            // 2. YOUR WORKSPACE
            // This is where you will add your Spatial Inference logic!
            // Example: For now, we just do nothing or small math
            // lab.L *= 1.1f; // Just an example of processing on the fly
            
            // 3. To keep it fast, we don't print inside the loop 
            // because printing is extremely slow!
        }
    }

    std::cout << "Successfully processed " << width * height << " pixels in a single pass!" << std::endl;

    stbi_image_free(imgData);
    return 0;
}