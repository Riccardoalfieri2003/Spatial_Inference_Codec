#include <iostream>
#include <vector>
#include "ImageConverter_mod.hpp" // Include our new converter
#include "VoxelGrid_mod.hpp"
#include "Cluster_mod.hpp"
#include "FileHandler_mod.hpp"
#include "GradientEncoder_mod.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Note: CMake handles the path now!

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Note: CMake handles the path now!

#include <iostream>
#include <string> // For std::stof and std::stoi

#include "GradientReconstruction_mod.hpp"
#include "PaletteEntry.hpp"


int main(int argc, char* argv[]) {
    // Check if the user provided enough arguments
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                << " <epsilon> <max_steps> <epsilon_res> <max_steps_res>" << std::endl;
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
    //const char* filename = "C:\\Users\\rical\\OneDrive\\Desktop\\Wallpaper\\Napoli.png";
    

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


    // ── Build flat Lab array (needed by gradient encoder) ────────────────────
    std::vector<LabPixelFlat> imgLabFlat(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixelOffset = (y * width + x) * channels;
            LabPixel lab = ImageConverter::convertPixelRGBtoLab(
                imgData[pixelOffset], imgData[pixelOffset+1], imgData[pixelOffset+2]
            );
            imgLabFlat[y * width + x] = {lab.L, lab.a, lab.b};
        }
    }

    // ── Encode gradients ─────────────────────────────────────────────────────
    // Choose precision: BITS_2, BITS_4, or BITS_6
    GradientData gradients = encodeGradients(
        indexMatrix,
        imgLabFlat,
        width, height,
        GradientPrecision::BITS_4,
        1.0f,   // changeThreshold — lower = more sensitive to gradient changes
        64       // segmentSize — smaller = more frequent updates along boundaries
    );


    

    // ── Build interleaved flat Lab arrays ────────────────────────────────────────
    // originalLab:   the real Lab value of every pixel
    // quantizedLab:  the palette centroid Lab value assigned to every pixel
    std::vector<float> originalLab(width * height * 3);
    std::vector<float> quantizedLab(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixelOffset = (y * width + x) * channels;
            int idx = y * width + x;

            // Original pixel in Lab
            LabPixel lab = ImageConverter::convertPixelRGBtoLab(
                imgData[pixelOffset], imgData[pixelOffset+1], imgData[pixelOffset+2]);
            originalLab[idx * 3 + 0] = lab.L;
            originalLab[idx * 3 + 1] = lab.a;
            originalLab[idx * 3 + 2] = lab.b;

            // Quantized: just the palette centroid for this pixel's cluster
            int clusterIdx = indexMatrix[idx];
            quantizedLab[idx * 3 + 0] = palette[clusterIdx].L;
            quantizedLab[idx * 3 + 1] = palette[clusterIdx].a;
            quantizedLab[idx * 3 + 2] = palette[clusterIdx].b;
        }
    }










    // ── Parse residual encoding parameters from command line ──────────────────
    float epsilonRes      = std::stof(argv[3]);
    int   maxStepsRes     = std::stoi(argv[4]);

    std::cout << "\nRunning Residual Encoder with Epsilon: " << epsilonRes
            << " and Max Steps: " << maxStepsRes << std::endl;

    // ── Step 1: Reconstruct the quantized+gradient image in Lab ──────────────
    // Start from quantized palette values, then apply gradient blending on top.
    std::vector<LabF> reconstructed(width * height);
    for (int i = 0; i < width * height; i++) {
        int cIdx = indexMatrix[i];
        reconstructed[i] = { palette[cIdx].L, palette[cIdx].a, palette[cIdx].b };
    }

    applyGradients(reconstructed, indexMatrix, palette, gradients, width, height);

    // ── Step 2: Compute residual (original Lab − reconstructed Lab) ───────────
    // Values are signed floats — voxelization with std::floor handles negatives correctly.
    std::vector<LabPixelFlat> residualLab(width * height);
    for (int i = 0; i < width * height; i++) {
        residualLab[i] = {
            imgLabFlat[i].L - reconstructed[i].L,
            imgLabFlat[i].a - reconstructed[i].a,
            imgLabFlat[i].b - reconstructed[i].b
        };
    }

    // ── Step 3: Voxelize the residual ─────────────────────────────────────────
    VoxelMap residualGrid;
    for (int i = 0; i < width * height; i++) {
        VoxelCoord coord = {
            static_cast<int>(std::floor(residualLab[i].L / epsilonRes)),
            static_cast<int>(std::floor(residualLab[i].a / epsilonRes)),
            static_cast<int>(std::floor(residualLab[i].b / epsilonRes))
        };
        // addPixel expects a LabPixel — bridge from LabPixelFlat
        LabPixel labPx = { residualLab[i].L, residualLab[i].a, residualLab[i].b };
        residualGrid[coord].addPixel(labPx);
    }

    std::cout << "\n--- Residual Voxel Grid ---" << std::endl;
    std::cout << "Total Non-Empty Voxels: " << residualGrid.size() << std::endl;

    // ── Step 4: Cluster the residual voxels ───────────────────────────────────
    ClusteringResult residualResult = Clusterer::run(residualGrid, maxStepsRes);

    // ── Step 5: Build residual palette ────────────────────────────────────────
    std::vector<PaletteEntry> residualPalette;
    for (const auto& cluster : residualResult.clusters) {
        auto stats = cluster.getStats();
        residualPalette.push_back({ stats.centroid.L, stats.centroid.a, stats.centroid.b, stats.maxError });
    }

    // ── Step 6: Build residual index matrix ───────────────────────────────────
    std::vector<int> residualIndexMatrix(width * height);
    for (int i = 0; i < width * height; i++) {
        VoxelCoord coord = {
            static_cast<int>(std::floor(residualLab[i].L / epsilonRes)),
            static_cast<int>(std::floor(residualLab[i].a / epsilonRes)),
            static_cast<int>(std::floor(residualLab[i].b / epsilonRes))
        };
        residualIndexMatrix[i] = residualResult.voxelToClusterIdx[coord];
    }

    std::cout << "\n--- Residual Compression Summary ---" << std::endl;
    std::cout << "Residual Clusters: " << residualPalette.size() << std::endl;

    float residualAvgError = 0.0f, residualMaxError = 0.0f;
    for (const auto& entry : residualPalette) {
        residualAvgError += entry.error;
        if (entry.error > residualMaxError) residualMaxError = entry.error;
    }
    residualAvgError /= residualPalette.size();
    std::cout << "Avg Residual Cluster Error: " << residualAvgError << std::endl;
    std::cout << "Max Residual Cluster Error: " << residualMaxError << std::endl;






    // ── Visualize: original, quantized, residual ──────────────────────────────
    {
        std::vector<unsigned char> imgOrigRGB(width * height * 3);
        std::vector<unsigned char> imgQuantRGB(width * height * 3);
        std::vector<unsigned char> imgResidualRGB(width * height * 3);

        for (int i = 0; i < width * height; i++) {
            // Original
            ImageConverter::convertPixelLabToRGB(
                imgLabFlat[i].L, imgLabFlat[i].a, imgLabFlat[i].b,
                imgOrigRGB[i*3], imgOrigRGB[i*3+1], imgOrigRGB[i*3+2]);

            // Quantized (reconstructed = palette + gradients)
            ImageConverter::convertPixelLabToRGB(
                reconstructed[i].L, reconstructed[i].a, reconstructed[i].b,
                imgQuantRGB[i*3], imgQuantRGB[i*3+1], imgQuantRGB[i*3+2]);

            // Residual — shift by 128 so zero residual = mid-grey
            float resL = residualLab[i].L + 50.0f;   // L is [0,100], center at 50
            float resA = residualLab[i].a + 0.0f;
            float resB = residualLab[i].b + 0.0f;
            ImageConverter::convertPixelLabToRGB(
                resL, resA, resB,
                imgResidualRGB[i*3], imgResidualRGB[i*3+1], imgResidualRGB[i*3+2]);
        }

        stbi_write_png("out_original.png",  width, height, 3, imgOrigRGB.data(),     width * 3);
        stbi_write_png("out_quantized.png", width, height, 3, imgQuantRGB.data(),    width * 3);
        stbi_write_png("out_residual.png",  width, height, 3, imgResidualRGB.data(), width * 3);

        // Open all three side by side on Windows
        system("start out_original.png");
        system("start out_quantized.png");
        system("start out_residual.png");

        std::cout << "\nImages saved and opened: out_original.png, out_quantized.png, out_residual.png\n";
    }

    // Quantized residual (after clustering)
    std::vector<unsigned char> imgResidualQuantRGB(width * height * 3);

    for (int i = 0; i < width * height; i++) {
        int cIdx = residualIndexMatrix[i];
        float resL = residualPalette[cIdx].L + 50.0f;
        float resA = residualPalette[cIdx].a;
        float resB = residualPalette[cIdx].b;
        ImageConverter::convertPixelLabToRGB(
            resL, resA, resB,
            imgResidualQuantRGB[i*3], imgResidualQuantRGB[i*3+1], imgResidualQuantRGB[i*3+2]);
    }

    stbi_write_png("out_residual_quantized.png", width, height, 3, imgResidualQuantRGB.data(), width * 3);
    system("start out_residual_quantized.png");


    saveSIF_v2("output_claude.sif", width, height, palette, indexMatrix, gradients, residualPalette, residualIndexMatrix);

    

    /*
    bool reduceMatrix=true;

    if (reduceMatrix){

        // Subsample the index matrix before saving
        int subW, subH;
        std::vector<int> subMatrix = subsampleIndexMatrix(indexMatrix, width, height, subW, subH);

        // Pass subMatrix instead of indexMatrix, and store subW/subH in the header
        saveSIF_claude_reduce("output_claude.sif", 
                subW, subH,        // ← subsampled dims go as w, h
                width, height,     // ← full dims go as origW, origH
                palette, subMatrix, gradients, residual);

    }
    else{

        // ── Save ─────────────────────────────────────────────────────────────────
        saveSIF_claude("output_claude.sif", width, height, palette, indexMatrix, gradients, residual);

    }
        */
    

        

    stbi_image_free(imgData);
    return 0;
}