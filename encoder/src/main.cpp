#include <iostream>
#include <vector>
#include "ImageConverter.hpp" // Include our new converter
#include "VoxelGrid.hpp"
#include "Cluster.hpp"
#include "FileHandler.hpp"

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
    const char* filename = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\encoder\\data\\images\\Lenna.png";
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

    
    // 1. Run the modified Clusterer
    ClusteringResult result = Clusterer::run(grid, maxStepsFromRoot);

    // 2. Build the Palette [L, a, b, e]
    std::vector<PaletteEntry> palette;

    for (const auto& cluster : result.clusters) {
        auto stats = cluster.getStats();
        palette.push_back({stats.centroid.L, stats.centroid.a, stats.centroid.b, stats.maxError});
    }

    // 3. Build the Index Matrix
    // Using a 1D vector to represent the 2D image matrix
    std::vector<int> indexMatrix(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixelOffset = (y * width + x) * channels;

            // Convert current pixel to its voxel coord again
            LabPixel lab = ImageConverter::convertPixelRGBtoLab(
                imgData[pixelOffset], imgData[pixelOffset+1], imgData[pixelOffset+2]
            );
            
            VoxelCoord coord = {
                static_cast<int>(std::floor(lab.L / epsilon)),
                static_cast<int>(std::floor(lab.a / epsilon)),
                static_cast<int>(std::floor(lab.b / epsilon))
            };

            // Assign the cluster index to this pixel
            indexMatrix[y * width + x] = result.voxelToClusterIdx[coord];
        }
    }

    std::cout << "\n--- Compression Summary ---" << std::endl;
    std::cout << "Original Pixels:    " << width * height << std::endl;
    std::cout << "Number of Clusters: " << palette.size() << std::endl;

    // Calculate "Reduction Ratio" in terms of unique colors
    // A standard image can have up to 16 million colors; we reduced it to palette.size()
    float reduction = (float)(width * height) / palette.size();
    std::cout << "Color Reduction:    " << reduction << ":1" << std::endl;

    // --- Detailed Palette & Error Analysis ---
    float avgMaxError = 0.0f;
    float absoluteMaxError = 0.0f;

    for (const auto& entry : palette) {
        avgMaxError += entry.error;
        if (entry.error > absoluteMaxError) absoluteMaxError = entry.error;
    }
    avgMaxError /= palette.size();

    std::cout << "\n--- Error Metrics (Lab Space) ---" << std::endl;
    std::cout << "Average Cluster Max-Error: " << avgMaxError << " (Perceptual distance)" << std::endl;
    std::cout << "Worst-Case Cluster Error:  " << absoluteMaxError << std::endl;

    // --- Index Matrix Verification ---
    // Let's check the first few indices to make sure they aren't all the same
    std::cout << "\n--- Index Matrix Sample (Top-Left 5x1) ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Pixel [" << i << "]: Cluster Index " << indexMatrix[i] << std::endl;
    }

    std::cout << "\nReady for file serialization!" << std::endl;

    saveSIF_claude("output_claude.sif", width, height, palette, indexMatrix);
        

    stbi_image_free(imgData);
    return 0;
}