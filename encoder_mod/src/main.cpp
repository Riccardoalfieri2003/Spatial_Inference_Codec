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

// ── Simulate float16 palette quantization (mirrors decoder precision) ─────
auto quantizePalette = [](std::vector<PaletteEntry>& pal) {
    for (auto& p : pal) {
        p.L = halfToFloat(floatToHalf(p.L));
        p.a = halfToFloat(floatToHalf(p.a));
        p.b = halfToFloat(floatToHalf(p.b));
    }
};


int main(int argc, char* argv[]) {
    // Check if the user provided enough arguments
    if (argc < 9) {
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

    // After main palette is built:
    quantizePalette(palette);
    

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

    for (int i = 0; i < 5; i++)
    std::cout << "ENC palette[" << i << "] L=" << palette[i].L 
              << " a=" << palette[i].a << " b=" << palette[i].b << "\n";
    std::cout << "ENC indexMatrix[0]=" << indexMatrix[0] << "\n";

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

    // DEBUG: save raw quantized image before any gradient application
    {
        std::vector<unsigned char> buf(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            ImageConverter::convertPixelLabToRGB(
                palette[indexMatrix[i]].L, palette[indexMatrix[i]].a, palette[indexMatrix[i]].b,
                buf[i*3], buf[i*3+1], buf[i*3+2]);
        }
        //stbi_write_png("debug_raw_quantized_NO_GRAD_encoder.png", width, height, 3, buf.data(), width * 3);
    }

    


    // ── Encode gradients ─────────────────────────────────────────────────────
    // Choose precision: BITS_2, BITS_4, or BITS_6
    GradientData gradients = encodeGradients(
        indexMatrix,
        imgLabFlat,
        width, height,
        GradientPrecision::BITS_2,
        1.0f,   // changeThreshold — lower = more sensitive to gradient changes
        32       // segmentSize — smaller = more frequent updates along boundaries
    );
    // After encodeGradients for residualGradients:
    for (auto& desc : gradients.queue) desc.shape = 1;

    // Add this after applyGradients in BOTH encoder and decoder
    for (int i = 0; i < 5; i++) {
        std::cout << "pixel[" << i << "] L=" << imgLabFlat[i].L  // or reconstructed[i] in encoder
                << " a=" << imgLabFlat[i].a
                << " b=" << imgLabFlat[i].b << "\n";
    }


    

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







    // ══════════════════════════════════════════════════════════════════════════
    // RESIDUAL 1
    // ══════════════════════════════════════════════════════════════════════════

    float epsilonRes      = std::stof(argv[3]);
    int   maxStepsRes     = std::stoi(argv[4]);
    float residualScale   = 1.0f;  // ← tune this: higher = finer clustering of residuals

    std::cout << "\nRunning Residual 1 Encoder with Epsilon: " << epsilonRes
              << " Max Steps: " << maxStepsRes 
              << " Scale: " << residualScale << std::endl;

    std::vector<LabF> reconstructed(width * height);
    for (int i = 0; i < width * height; i++) {
        int cIdx = indexMatrix[i];
        reconstructed[i] = { palette[cIdx].L, palette[cIdx].a, palette[cIdx].b };
    }


    // Print indexMatrix values around pixel 0 in a 5x5 region
    std::cout << "indexMatrix patch around (0,0):\n";
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++)
            std::cout << indexMatrix[y * width + x] << " ";  // or data.indexMatrix
        std::cout << "\n";
    }


    applyGradients(reconstructed, indexMatrix, palette, gradients, width, height);

    std::vector<LabPixelFlat> residualLab(width * height);
    for (int i = 0; i < width * height; i++) {
        residualLab[i] = {
            imgLabFlat[i].L - reconstructed[i].L,
            imgLabFlat[i].a - reconstructed[i].a,
            imgLabFlat[i].b - reconstructed[i].b
        };
    }

    // ── Voxelize residual 1 in scaled space ───────────────────────────────────
    VoxelMap residualGrid;
    for (int i = 0; i < width * height; i++) {
        float sL = residualLab[i].L * residualScale;
        float sA = residualLab[i].a * residualScale;
        float sB = residualLab[i].b * residualScale;
        VoxelCoord coord = {
            static_cast<int>(std::floor(sL / epsilonRes)),
            static_cast<int>(std::floor(sA / epsilonRes)),
            static_cast<int>(std::floor(sB / epsilonRes))
        };
        // Add the scaled values so the centroid is in scaled space
        residualGrid[coord].addPixel({ sL, sA, sB });
    }
    std::cout << "\n--- Residual 1 Voxel Grid: " << residualGrid.size() << " voxels ---" << std::endl;

    ClusteringResult residualResult = Clusterer::run(residualGrid, maxStepsRes);

    // ── Build palette: scale centroids back to original space ─────────────────
    std::vector<PaletteEntry> residualPalette;
    for (const auto& cluster : residualResult.clusters) {
        auto stats = cluster.getStats();
        residualPalette.push_back({
            stats.centroid.L / residualScale,
            stats.centroid.a / residualScale,
            stats.centroid.b / residualScale,
            stats.maxError   / residualScale
        });
    }

    // After main palette is built:
    quantizePalette(residualPalette);

    // Encoder — after quantizePalette(residualPalette):
    for (int i = 0; i < 5; i++)
        std::cout << "ENC resPalette[" << i << "] L=" << residualPalette[i].L 
                << " a=" << residualPalette[i].a << " b=" << residualPalette[i].b << "\n";

    // ── Build index matrix using scaled coords ────────────────────────────────
    std::vector<int> residualIndexMatrix(width * height);
    for (int i = 0; i < width * height; i++) {
        float sL = residualLab[i].L * residualScale;
        float sA = residualLab[i].a * residualScale;
        float sB = residualLab[i].b * residualScale;
        VoxelCoord coord = {
            static_cast<int>(std::floor(sL / epsilonRes)),
            static_cast<int>(std::floor(sA / epsilonRes)),
            static_cast<int>(std::floor(sB / epsilonRes))
        };
        residualIndexMatrix[i] = residualResult.voxelToClusterIdx[coord];
    }

    std::cout << "Residual 1 Clusters: " << residualPalette.size() << std::endl;

    GradientData residualGradients = encodeGradients(
        residualIndexMatrix, residualLab, width, height,
        GradientPrecision::BITS_2, 0.25f, 16
    );
    // After encodeGradients for residualGradients2:
    for (auto& desc : residualGradients.queue) desc.shape = 1;




    // ══════════════════════════════════════════════════════════════════════════
    // RESIDUAL 2
    // ══════════════════════════════════════════════════════════════════════════

    float epsilonRes2    = std::stof(argv[5]);
    int   maxStepsRes2   = std::stoi(argv[6]);
    // float residualScale2 = residualScale * 2.0f;  // scale even more for second residual
    float residualScale2 = residualScale * 1.0f;  // scale even more for second residual

    std::cout << "\nRunning Residual 2 Encoder with Epsilon: " << epsilonRes2
              << " Max Steps: " << maxStepsRes2
              << " Scale: " << residualScale2 << std::endl;

    std::vector<LabF> reconstructedRes1(width * height);
    for (int i = 0; i < width * height; i++) {
        int cIdx = residualIndexMatrix[i];
        reconstructedRes1[i] = { residualPalette[cIdx].L, residualPalette[cIdx].a, residualPalette[cIdx].b };
    }
    applyGradients(reconstructedRes1, residualIndexMatrix, residualPalette, residualGradients, width, height);

    std::vector<LabPixelFlat> residualLab2(width * height);
    for (int i = 0; i < width * height; i++) {
        float reconL = reconstructed[i].L + reconstructedRes1[i].L;
        float reconA = reconstructed[i].a + reconstructedRes1[i].a;
        float reconB = reconstructed[i].b + reconstructedRes1[i].b;
        residualLab2[i] = {
            imgLabFlat[i].L - reconL,
            imgLabFlat[i].a - reconA,
            imgLabFlat[i].b - reconB
        };
    }

    // ── Voxelize residual 2 in scaled space ───────────────────────────────────
    VoxelMap residualGrid2;
    for (int i = 0; i < width * height; i++) {
        float sL = residualLab2[i].L * residualScale2;
        float sA = residualLab2[i].a * residualScale2;
        float sB = residualLab2[i].b * residualScale2;
        VoxelCoord coord = {
            static_cast<int>(std::floor(sL / epsilonRes2)),
            static_cast<int>(std::floor(sA / epsilonRes2)),
            static_cast<int>(std::floor(sB / epsilonRes2))
        };
        residualGrid2[coord].addPixel({ sL, sA, sB });
    }
    std::cout << "\n--- Residual 2 Voxel Grid: " << residualGrid2.size() << " voxels ---" << std::endl;

    ClusteringResult residualResult2 = Clusterer::run(residualGrid2, maxStepsRes2);

    // ── Build palette: scale centroids back to original space ─────────────────
    std::vector<PaletteEntry> residualPalette2;
    for (const auto& cluster : residualResult2.clusters) {
        auto stats = cluster.getStats();
        residualPalette2.push_back({
            stats.centroid.L / residualScale2,
            stats.centroid.a / residualScale2,
            stats.centroid.b / residualScale2,
            stats.maxError   / residualScale2
        });
    }

    // ── Build index matrix using scaled coords ────────────────────────────────
    std::vector<int> residualIndexMatrix2(width * height);
    for (int i = 0; i < width * height; i++) {
        float sL = residualLab2[i].L * residualScale2;
        float sA = residualLab2[i].a * residualScale2;
        float sB = residualLab2[i].b * residualScale2;
        VoxelCoord coord = {
            static_cast<int>(std::floor(sL / epsilonRes2)),
            static_cast<int>(std::floor(sA / epsilonRes2)),
            static_cast<int>(std::floor(sB / epsilonRes2))
        };
        residualIndexMatrix2[i] = residualResult2.voxelToClusterIdx[coord];
    }

    // After main palette is built:
    quantizePalette(residualPalette2);

    // Encoder — after quantizePalette(residualPalette):
for (int i = 0; i < 5; i++)
    std::cout << "ENC resPalette2[" << i << "] L=" << residualPalette2[i].L 
              << " a=" << residualPalette2[i].a << " b=" << residualPalette2[i].b << "\n";

    std::cout << "Residual 2 Clusters: " << residualPalette2.size() << std::endl;

    GradientData residualGradients2 = encodeGradients(
        residualIndexMatrix2, residualLab2, width, height,
        GradientPrecision::BITS_2, 0.25f, 16
    );
    // After encodeGradients for residualGradients3:
    for (auto& desc : residualGradients2.queue) desc.shape = 1; 

    std::vector<LabF> reconstructedRes2(width * height);
    for (int i = 0; i < width * height; i++) {
        int cIdx = residualIndexMatrix2[i];
        reconstructedRes2[i] = { residualPalette2[cIdx].L, residualPalette2[cIdx].a, residualPalette2[cIdx].b };
    }
    applyGradients(reconstructedRes2, residualIndexMatrix2, residualPalette2, residualGradients2, width, height);




    // ══════════════════════════════════════════════════════════════════════════
    // RESIDUAL 3
    // ══════════════════════════════════════════════════════════════════════════

    float epsilonRes3    = std::stof(argv[7]);
    int   maxStepsRes3   = std::stoi(argv[8]);
    float residualScale3 = residualScale2 * 1.0f;  // 16x scaling for third residual

    std::cout << "\nRunning Residual 3 Encoder with Epsilon: " << epsilonRes3
              << " Max Steps: " << maxStepsRes3
              << " Scale: " << residualScale3 << std::endl;

    // ── Reconstruct residual2+gradient image ──────────────────────────────────
    std::vector<LabF> reconstructedRes3(width * height);
    for (int i = 0; i < width * height; i++) {
        int cIdx = residualIndexMatrix2[i];
        reconstructedRes3[i] = { residualPalette2[cIdx].L, residualPalette2[cIdx].a, residualPalette2[cIdx].b };
    }
    applyGradients(reconstructedRes3, residualIndexMatrix2, residualPalette2, residualGradients2, width, height);

    // ── Compute residual 3 ────────────────────────────────────────────────────
    std::vector<LabPixelFlat> residualLab3(width * height);
    for (int i = 0; i < width * height; i++) {
        float reconL = reconstructed[i].L + reconstructedRes1[i].L + reconstructedRes2[i].L;
        float reconA = reconstructed[i].a + reconstructedRes1[i].a + reconstructedRes2[i].L;
        float reconB = reconstructed[i].b + reconstructedRes1[i].b + reconstructedRes2[i].L;
        residualLab3[i] = {
            imgLabFlat[i].L - reconL,
            imgLabFlat[i].a - reconA,
            imgLabFlat[i].b - reconB
        };
    }

    // ── Voxelize residual 3 in scaled space ───────────────────────────────────
    VoxelMap residualGrid3;
    for (int i = 0; i < width * height; i++) {
        float sL = residualLab3[i].L * residualScale3;
        float sA = residualLab3[i].a * residualScale3;
        float sB = residualLab3[i].b * residualScale3;
        VoxelCoord coord = {
            static_cast<int>(std::floor(sL / epsilonRes3)),
            static_cast<int>(std::floor(sA / epsilonRes3)),
            static_cast<int>(std::floor(sB / epsilonRes3))
        };
        residualGrid3[coord].addPixel({ sL, sA, sB });
    }
    std::cout << "\n--- Residual 3 Voxel Grid: " << residualGrid3.size() << " voxels ---" << std::endl;

    ClusteringResult residualResult3 = Clusterer::run(residualGrid3, maxStepsRes3);

    // ── Build palette: scale centroids back to original space ─────────────────
    std::vector<PaletteEntry> residualPalette3;
    for (const auto& cluster : residualResult3.clusters) {
        auto stats = cluster.getStats();
        residualPalette3.push_back({
            stats.centroid.L / residualScale3,
            stats.centroid.a / residualScale3,
            stats.centroid.b / residualScale3,
            stats.maxError   / residualScale3
        });
    }

    // ── Build index matrix using scaled coords ────────────────────────────────
    std::vector<int> residualIndexMatrix3(width * height);
    for (int i = 0; i < width * height; i++) {
        float sL = residualLab3[i].L * residualScale3;
        float sA = residualLab3[i].a * residualScale3;
        float sB = residualLab3[i].b * residualScale3;
        VoxelCoord coord = {
            static_cast<int>(std::floor(sL / epsilonRes3)),
            static_cast<int>(std::floor(sA / epsilonRes3)),
            static_cast<int>(std::floor(sB / epsilonRes3))
        };
        residualIndexMatrix3[i] = residualResult3.voxelToClusterIdx[coord];
    }

    // After main palette is built:
    quantizePalette(residualPalette3);

    // Encoder — after quantizePalette(residualPalette):
for (int i = 0; i < 5; i++)
    std::cout << "ENC resPalette3[" << i << "] L=" << residualPalette3[i].L 
              << " a=" << residualPalette3[i].a << " b=" << residualPalette3[i].b << "\n";

    std::cout << "Residual 3 Clusters: " << residualPalette3.size() << std::endl;

    GradientData residualGradients3 = encodeGradients(
        residualIndexMatrix3, residualLab3, width, height,
        GradientPrecision::BITS_2, 0.25f, 16
    );

    // After encodeGradients for residualGradients3:
    for (auto& desc : residualGradients3.queue) desc.shape = 1; 

    // ── Reconstruct residual 3 for visualization ──────────────────────────────
    std::vector<LabF> reconstructedRes3Final(width * height);
    for (int i = 0; i < width * height; i++) {
        int cIdx = residualIndexMatrix3[i];
        reconstructedRes3Final[i] = { residualPalette3[cIdx].L, residualPalette3[cIdx].a, residualPalette3[cIdx].b };
    }
    applyGradients(reconstructedRes3Final, residualIndexMatrix3, residualPalette3, residualGradients3, width, height);


    /*
    // ── Update fullRecon to include residual 3 ────────────────────────────────
    for (int i = 0; i < width * height; i++) {
        fullRecon[i] = {
            reconstructed[i].L + reconstructedRes1[i].L + reconstructedRes2[i].L + reconstructedRes3Final[i].L,
            reconstructed[i].a + reconstructedRes1[i].a + reconstructedRes2[i].a + reconstructedRes3Final[i].a,
            reconstructed[i].b + reconstructedRes1[i].b + reconstructedRes2[i].b + reconstructedRes3Final[i].b
        };
    }
    */

    std::cout << "\n--- ENCODER FINAL STATE ---\n";
    for (int i = 0; i < 5; i++)
        std::cout << "ENC final palette[" << i << "] L=" << palette[i].L 
                << " a=" << palette[i].a << " b=" << palette[i].b << "\n";
    std::cout << "ENC gradients queue[0]: shape=" << (int)gradients.queue[0].shape
            << " width=" << (int)gradients.queue[0].width << "\n";
    std::cout << "ENC reconstructed[0]: L=" << reconstructed[0].L
            << " a=" << reconstructed[0].a << " b=" << reconstructed[0].b << "\n";

    // ══════════════════════════════════════════════════════════════════════════
    // VISUALIZATION
    // ══════════════════════════════════════════════════════════════════════════

    // Helper lambda: Lab image → PNG file + optional open
    auto saveLabImage = [&](
        const std::vector<LabF>& img,
        const std::string& filename,
        float shiftL = 0.0f)
    {
        std::vector<unsigned char> buf(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            ImageConverter::convertPixelLabToRGB(
                img[i].L + shiftL, img[i].a, img[i].b,
                buf[i*3], buf[i*3+1], buf[i*3+2]);
        }
        stbi_write_png(filename.c_str(), width, height, 3, buf.data(), width * 3);
        std::cout << "Saved: " << filename << "\n";
    };

    auto saveLabFlatImage = [&](
        const std::vector<LabPixelFlat>& img,
        const std::string& filename,
        float shiftL = 0.0f)
    {
        std::vector<unsigned char> buf(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            ImageConverter::convertPixelLabToRGB(
                img[i].L + shiftL, img[i].a, img[i].b,
                buf[i*3], buf[i*3+1], buf[i*3+2]);
        }
        //stbi_write_png(filename.c_str(), width, height, 3, buf.data(), width * 3);
        std::cout << "Saved: " << filename << "\n";
    };

    // Build full reconstruction for visualization
    std::vector<LabF> fullRecon(width * height);
    for (int i = 0; i < width * height; i++) {
        fullRecon[i] = {
            reconstructed[i].L + reconstructedRes1[i].L + reconstructedRes2[i].L + reconstructedRes3Final[i].L,
            reconstructed[i].a + reconstructedRes1[i].a + reconstructedRes2[i].a + reconstructedRes3Final[i].a,
            reconstructed[i].b + reconstructedRes1[i].b + reconstructedRes2[i].b + reconstructedRes3Final[i].b,
        };
    }

    // Convert original imgLabFlat to LabF for saveLabImage
    std::vector<LabF> originalLabF(width * height);
    for (int i = 0; i < width * height; i++)
        originalLabF[i] = { imgLabFlat[i].L, imgLabFlat[i].a, imgLabFlat[i].b };

    //saveLabImage(originalLabF,   "out_original.png");
    //saveLabImage(reconstructed,  "out_quantized_grad.png");
    //saveLabFlatImage(residualLab,  "out_residual1.png", 50.0f);
    //saveLabFlatImage(residualLab2, "out_residual2.png", 50.0f);
    saveLabImage(fullRecon,      "out_full_reconstruction.png");

    // ══════════════════════════════════════════════════════════════════════════
    // SAVE
    // ══════════════════════════════════════════════════════════════════════════

    saveSIF_v2("output_claude.sif",
               width, height,
               palette, indexMatrix, gradients,
               residualPalette,  residualIndexMatrix,  residualGradients,
               residualPalette2, residualIndexMatrix2, residualGradients2,
               residualPalette3, residualIndexMatrix3, residualGradients3);





        

    stbi_image_free(imgData);
    return 0;
}