#include <iostream>
#include <vector>
#include "ImageConverter.hpp" // Include our new converter
#include "VoxelGrid.hpp"
#include "Cluster.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Note: CMake handles the path now!

#include <iostream>
#include <string> // For std::stof and std::stoi


int main(int argc, char* argv[]) {
    // Check if the user provided enough arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <epsilon> <max_steps>" << std::endl;
        std::cout << "Example: " << argv[0] << " 5.0 2" << std::endl;
        return 1;
    }

    // Convert command line strings to float and int
    // argv[0] is the program name, so we start at index 1
    float epsilon = std::stof(argv[1]);
    int maxStepsFromRoot = std::stoi(argv[2]);

    std::cout << "Running Encoder with Epsilon: " << epsilon 
              << " and Max Steps: " << maxStepsFromRoot << std::endl;

    // ... your image loading code ...
    const char* filename = "../data/images/Lenna.png";
    int width, height, channels;
    unsigned char* imgData = stbi_load(filename, &width, &height, &channels, 0);

    if (!imgData) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

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

    


    std::vector<Cluster> finalClusters = Clusterer::run(grid, maxStepsFromRoot);
    std::cout << "Clusters generated: " << finalClusters.size() << std::endl;


    std::cout << "\n--- Final Cluster Analysis ---" << std::endl;
    for (size_t i = 0; i < finalClusters.size(); ++i) {
        Cluster::ClusterStats stats = finalClusters[i].getStats();
        
        // Only print interesting clusters or the first few to keep the terminal clean
        if (i < 5) {
            std::cout << "Cluster #" << i << ":" << std::endl;
            std::cout << "  Pixels: " << finalClusters[i].totalPixels << std::endl;
            std::cout << "  Centroid Lab: (" << stats.centroid.L << ", " 
                    << stats.centroid.a << ", " << stats.centroid.b << ")" << std::endl;
            std::cout << "  Max Error: " << stats.maxError << std::endl;
            std::cout << "-----------------------" << std::endl;
        }
}

        

    stbi_image_free(imgData);
    return 0;
}