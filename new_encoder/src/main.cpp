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




#include <vector>
#include <cmath>
#include <iostream>



// ── Simulate float16 palette quantization (mirrors decoder precision) ─────
auto quantizePalette = [](std::vector<PaletteEntry>& pal) {
    for (auto& p : pal) {
        p.L = halfToFloat(floatToHalf(p.L));
        p.a = halfToFloat(floatToHalf(p.a));
        p.b = halfToFloat(floatToHalf(p.b));
    }       
};



struct ChannelQuantization {
    std::vector<float> palette;
    std::vector<int> indexMatrix;
};

struct ChannelResult {
    std::vector<float> palette;
    std::vector<int> indexMatrix;
    GradientData gradients;
};
ChannelQuantization quantizeChannel(
    const std::vector<float>& channelData, 
    int width, int height, 
    float epsilon, int maxSteps,
    const std::string& label = "")
{
    VoxelMap grid;
    
    for (float val : channelData) {
        VoxelCoord coord = { static_cast<int>(std::floor(val / epsilon)), 0, 0 };
        grid[coord].addPixel({ val, 0.0f, 0.0f });
    }

    ClusteringResult result = Clusterer::run(grid, maxSteps);

    std::vector<float> palette;
    for (const auto& cluster : result.clusters) {
        palette.push_back(cluster.getStats().centroid.L);
    }

    std::vector<int> indices(width * height);
    for (int i = 0; i < width * height; ++i) {
        VoxelCoord coord = { static_cast<int>(std::floor(channelData[i] / epsilon)), 0, 0 };
        indices[i] = result.voxelToClusterIdx[coord];
    }

    // ── Debug ─────────────────────────────────────────────────────────────────
    float minVal = *std::min_element(channelData.begin(), channelData.end());
    float maxVal = *std::max_element(channelData.begin(), channelData.end());
    float minPal = *std::min_element(palette.begin(), palette.end());
    float maxPal = *std::max_element(palette.begin(), palette.end());
    std::cout << "[quantizeChannel] " << label << "\n"
              << "  Input range   : [" << minVal << ", " << maxVal << "]\n"
              << "  Epsilon       : " << epsilon << "  MaxSteps: " << maxSteps << "\n"
              << "  Palette size  : " << palette.size() << "\n"
              << "  Palette range : [" << minPal << ", " << maxPal << "]\n";
    // ─────────────────────────────────────────────────────────────────────────

    return { palette, indices };
}

GradientData encodeChannelGradients(
    const std::vector<int>& indices, 
    const std::vector<float>& originalChannel, 
    int width, int height,
    const std::string& label = "")
{
    std::vector<LabPixelFlat> tempLab(width * height);
    for(int i=0; i < width*height; ++i) tempLab[i] = {originalChannel[i], 0, 0};

    GradientData grads = encodeGradients(indices, tempLab, width, height, GradientPrecision::BITS_2, 1.0f, 64);
    for (auto& desc : grads.queue) desc.shape = 1;

    // ── Debug ─────────────────────────────────────────────────────────────────
    std::cout << "[encodeChannelGradients] " << label << "\n"
              << "  Gradient descriptors: " << grads.queue.size() << "\n";
    // ─────────────────────────────────────────────────────────────────────────

    return grads;
}

ChannelResult encodeChannelResiduals(
    const std::vector<float>& originalChannel,
    const ChannelQuantization& base,
    const GradientData& baseGrads,
    int width, int height,
    float epsilonRes, int maxStepsRes,
    const std::string& label = "")
{
    // 1. Reconstruct
    std::vector<LabF> reconstructed(width * height);
    std::vector<PaletteEntry> tempPal; 
    for(float p : base.palette) tempPal.push_back({p, 0, 0, 0});

    for (int i = 0; i < width * height; i++)
        reconstructed[i] = { base.palette[base.indexMatrix[i]], 0, 0 };
    applyGradients(reconstructed, base.indexMatrix, tempPal, baseGrads, width, height);

    // 2. Residuals
    std::vector<float> residuals(width * height);
    for (int i = 0; i < width * height; i++)
        residuals[i] = originalChannel[i] - reconstructed[i].L;

    // ── Debug ─────────────────────────────────────────────────────────────────
    float minRes = *std::min_element(residuals.begin(), residuals.end());
    float maxRes = *std::max_element(residuals.begin(), residuals.end());
    float sumAbsRes = 0.0f;
    for (float r : residuals) sumAbsRes += std::abs(r);
    float meanAbsRes = sumAbsRes / residuals.size();

    float minRecon = reconstructed[0].L, maxRecon = reconstructed[0].L;
    for (const auto& r : reconstructed) {
        minRecon = std::min(minRecon, r.L);
        maxRecon = std::max(maxRecon, r.L);
    }

    std::cout << "[encodeChannelResiduals] " << label << "\n"
              << "  Reconstructed range : [" << minRecon << ", " << maxRecon << "]\n"
              << "  Residual range      : [" << minRes   << ", " << maxRes   << "]\n"
              << "  Residual mean |err| : " << meanAbsRes << "\n";
    // ─────────────────────────────────────────────────────────────────────────

    // 3. Quantize residuals
    auto resQuant = quantizeChannel(residuals, width, height, epsilonRes, maxStepsRes, label + "_residual");
    
    // 4. Encode residual gradients
    auto resGrads = encodeChannelGradients(resQuant.indexMatrix, residuals, width, height, label + "_residual");

    return { resQuant.palette, resQuant.indexMatrix, resGrads };
}











void reconstructFinalImage(
    int width, int height,
    const ChannelQuantization& baseL, const GradientData& gradsL, const ChannelResult& resL,
    const ChannelQuantization& baseA, const GradientData& gradsA, const ChannelResult& resA,
    const ChannelQuantization& baseB, const GradientData& gradsB, const ChannelResult& resB,
    unsigned char* outRgbData) 
{
    int totalPixels = width * height;
    
    // We'll process each channel (L, A, B)
    auto reconstructChannel = [&](const ChannelQuantization& base, 
                                  const GradientData& bGrads, 
                                  const ChannelResult& residual) 
    {
        std::vector<LabF> channel(totalPixels);
        
        // 1. Build Base from Index Matrix
        for (int i = 0; i < totalPixels; i++) {
            channel[i] = { base.palette[base.indexMatrix[i]], 0.0f, 0.0f };
        }

        // 2. Apply Base Gradients
        // We need a dummy palette for the applyGradients function
        std::vector<PaletteEntry> tempPal;
        for (float p : base.palette) tempPal.push_back({p, 0, 0, 0});
        applyGradients(channel, base.indexMatrix, tempPal, bGrads, width, height);

        // 3. Add Residuals (Residual Palette + Residual Gradients)
        // First, reconstruct the residual matrix itself
        std::vector<LabF> resRecon(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            resRecon[i] = { residual.palette[residual.indexMatrix[i]], 0.0f, 0.0f };
        }
        
        std::vector<PaletteEntry> tempResPal;
        for (float p : residual.palette) tempResPal.push_back({p, 0, 0, 0});
        applyGradients(resRecon, residual.indexMatrix, tempResPal, residual.gradients, width, height);

        // Final Combine: Base + Residual
        std::vector<float> finalChannel(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            finalChannel[i] = channel[i].L + resRecon[i].L;
        }
        return finalChannel;
    };

    // Reconstruct individual channels
    std::vector<float> finalL = reconstructChannel(baseL, gradsL, resL);
    std::vector<float> finalA = reconstructChannel(baseA, gradsA, resA);
    std::vector<float> finalB = reconstructChannel(baseB, gradsB, resB);

    // ── Convert back to RGB ──────────────────────────────────────────────────
    for (int i = 0; i < totalPixels; i++) {
        ImageConverter::convertPixelLabToRGB(
            finalL[i], finalA[i], finalB[i],
            outRgbData[i * 3], outRgbData[i * 3 + 1], outRgbData[i * 3 + 2]
        );
    }
}



void reconstructFinalImage_noResiduals(
    int width, int height,
    const ChannelQuantization& baseL, const GradientData& gradsL,
    const ChannelQuantization& baseA, const GradientData& gradsA,
    const ChannelQuantization& baseB, const GradientData& gradsB,
    unsigned char* outRgbData) 
{
    int totalPixels = width * height;

    auto reconstructChannel = [&](const ChannelQuantization& base, const GradientData& bGrads)
    {
        // 1. Build from index matrix
        std::vector<LabF> channel(totalPixels);
        for (int i = 0; i < totalPixels; i++)
            channel[i] = { base.palette[base.indexMatrix[i]], 0.0f, 0.0f };

        // 2. Apply gradients
        std::vector<PaletteEntry> tempPal;
        for (float p : base.palette) tempPal.push_back({p, 0, 0, 0});
        applyGradients(channel, base.indexMatrix, tempPal, bGrads, width, height);

        // 3. Extract the single channel values
        std::vector<float> result(totalPixels);
        for (int i = 0; i < totalPixels; i++)
            result[i] = channel[i].L;
        return result;
    };

    std::vector<float> finalL = reconstructChannel(baseL, gradsL);
    std::vector<float> finalA = reconstructChannel(baseA, gradsA);
    std::vector<float> finalB = reconstructChannel(baseB, gradsB);

    for (int i = 0; i < totalPixels; i++) {
        ImageConverter::convertPixelLabToRGB(
            finalL[i], finalA[i], finalB[i],
            outRgbData[i * 3], outRgbData[i * 3 + 1], outRgbData[i * 3 + 2]
        );
    }
}






int main(int argc, char* argv[]) {
   

    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <epsilonL> <epsilonA> <epsilonB> <maxStepsL> <maxStepsA> <maxStepsB>\n";
        return 1;
    }

    float epsilonL  = std::stof(argv[1]);
    float epsilonA  = std::stof(argv[2]);
    float epsilonB  = std::stof(argv[3]);
    int   maxStepsL = std::stoi(argv[4]);
    int   maxStepsA = std::stoi(argv[5]);
    int   maxStepsB = std::stoi(argv[6]);

    const char* filename = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\encoder\\data\\images\\Lenna.png";

    int width, height, channels;
    unsigned char* imgData = stbi_load(filename, &width, &height, &channels, 0);
    if (!imgData) { std::cerr << "Failed to load image.\n"; return 1; }

    int totalPixels = width * height;
    bool subsample = true;

    // ── Convert to Lab ────────────────────────────────────────────────────────
    std::vector<float> channelL(totalPixels);
    std::vector<float> channelA(totalPixels);
    std::vector<float> channelB(totalPixels);

    for (int i = 0; i < totalPixels; i++) {
        int pixelOffset = i * channels;
        LabPixel lab = ImageConverter::convertPixelRGBtoLab(
            imgData[pixelOffset], imgData[pixelOffset+1], imgData[pixelOffset+2]);
        channelL[i] = lab.L;
        channelA[i] = lab.a;
        channelB[i] = lab.b;
    }



    // --- Process L (Luminance) ---
    auto baseL = quantizeChannel(channelL, width, height, epsilonL, maxStepsL, "L_base");
    auto gradsL = encodeChannelGradients(baseL.indexMatrix, channelL, width, height, "L_base");
    auto residualL = encodeChannelResiduals(channelL, baseL, gradsL, width, height, epsilonL, maxStepsL);

    std::cout<<"\n\n";

    // --- Process A (Luminance) ---
    auto baseA = quantizeChannel(channelA, width, height, epsilonA, maxStepsA, "A_base");
    auto gradsA = encodeChannelGradients(baseA.indexMatrix, channelA, width, height, "A_base");
    auto residualA = encodeChannelResiduals(channelA, baseA, gradsA, width, height, epsilonA, maxStepsA);

    std::cout<<"\n\n";
    
    // --- Process L (Luminance) ---
    auto baseB = quantizeChannel(channelB, width, height, epsilonB, maxStepsB, "B_base");
    auto gradsB = encodeChannelGradients(baseB.indexMatrix, channelB, width, height, "B_base");
    auto residualB = encodeChannelResiduals(channelB, baseB, gradsB, width, height, epsilonB, maxStepsB);

    std::cout<<"\n\n";





    // Create a buffer for the output
    std::vector<unsigned char> outputRgb_noRes(width * height * 3);

    // Call the reconstruction
    reconstructFinalImage_noResiduals(
        width, height,
        baseL, gradsL,
        baseA, gradsA,
        baseB, gradsB,
        outputRgb_noRes.data()
    );

    // Save to disk to check results
    stbi_write_png("reconstructed_result_noRes.png", width, height, 3, outputRgb_noRes.data(), width * 3);




    // Create a buffer for the output
    std::vector<unsigned char> outputRgb(width * height * 3);

    // Call the reconstruction
    reconstructFinalImage(
        width, height,
        baseL, gradsL, residualL,
        baseA, gradsA, residualA,
        baseB, gradsB, residualB,
        outputRgb.data()
    );

    // Save to disk to check results
    stbi_write_png("reconstructed_result.png", width, height, 3, outputRgb.data(), width * 3);




















    /*

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

    */




        

    stbi_image_free(imgData);
    return 0;
}