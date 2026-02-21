#include <iostream>
#include <vector>
#include "ImageConverter.hpp" // Include our new converter
#include "VoxelGrid.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Note: CMake handles the path now!

int main() {
    const char* filename = "../data/images/Lenna.png";
    int width, height, channels;
    unsigned char* imgData = stbi_load(filename, &width, &height, &channels, 0);

    if (!imgData) return 1;

    float epsilon = 1.0f; // This is your quantization "strength"
    VoxelMap grid;

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

            // 2. Map to Voxel Coordinates
            // Formula: index = floor(value / epsilon)
            VoxelCoord coord = {
                static_cast<int>(std::floor(lab.L / epsilon)),
                static_cast<int>(std::floor(lab.a / epsilon)),
                static_cast<int>(std::floor(lab.b / epsilon))
            };

            // This now increments the count for THAT specific color in THAT specific voxel
            grid[coord].addPixel(lab);


        }
    }


    std::cout << "\n--- Voxel Grid Inspection ---" << std::endl;
    std::cout << "Total Non-Empty Voxels: " << grid.size() << std::endl;

    

    // Iterate through the map
    for (auto const& [coord, data] : grid) {
        std::cout << "Voxel Index: [" << coord.l_idx << ", " << coord.a_idx << ", " << coord.b_idx << "]" << std::endl;
        std::cout << "  Total Pixels in Voxel: " << data.totalPixelCount << std::endl;
        std::cout << "  Unique Colors inside: " << data.colorFrequencies.size() << std::endl;

        // Iterate through the colors inside this specific voxel
        for (auto const& [color, count] : data.colorFrequencies) {
            std::cout << "    - Color (L:" << color.L << ", a:" << color.a << ", b:" << color.b 
                    << ") -> Count: " << count << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
    }

    std::cout << "Original Pixels: " << width * height << std::endl;
    std::cout << "Non-empty Voxels: " << grid.size() << std::endl;
    std::cout << "Compression Ratio (Color space): " << (float)(width * height) / grid.size() << ":1" << std::endl;

    

    stbi_image_free(imgData);
    return 0;
}