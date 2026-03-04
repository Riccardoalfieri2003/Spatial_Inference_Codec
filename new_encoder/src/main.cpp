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



ChannelResult encodeChannelResiduals_pro(
    const std::vector<float>& originalChannel,
    const std::vector<float>& reconstructed,
    int width, int height,
    float epsilonRes, int maxStepsRes,
    const std::string& label = "")
{
    // 1. Residuals — reconstruction already done externally
    std::vector<float> residuals(width * height);
    for (int i = 0; i < width * height; i++)
        residuals[i] = originalChannel[i] - reconstructed[i];

    // ── Debug ─────────────────────────────────────────────────────────────────
    float minRes = *std::min_element(residuals.begin(), residuals.end());
    float maxRes = *std::max_element(residuals.begin(), residuals.end());
    float sumAbsRes = 0.0f;
    for (float r : residuals) sumAbsRes += std::abs(r);
    float meanAbsRes = sumAbsRes / residuals.size();

    float minRecon = *std::min_element(reconstructed.begin(), reconstructed.end());
    float maxRecon = *std::max_element(reconstructed.begin(), reconstructed.end());

    std::cout << "[encodeChannelResiduals] " << label << "\n"
              << "  Reconstructed range : [" << minRecon << ", " << maxRecon << "]\n"
              << "  Residual range      : [" << minRes   << ", " << maxRes   << "]\n"
              << "  Residual mean |err| : " << meanAbsRes << "\n";
    // ─────────────────────────────────────────────────────────────────────────

    // 2. Quantize residuals
    auto resQuant = quantizeChannel(residuals, width, height, epsilonRes, maxStepsRes, label + "_residual");

    // 3. Encode residual gradients
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














std::vector<int> applyMED_encode(
    const std::vector<int>& indexMatrix,
    int width, int height,
    int paletteSize)
{
    std::vector<int> residuals(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int val = indexMatrix[idx];
            int prediction;
            if (x == 0 && y == 0) {
                prediction = 0;
            } else if (y == 0) {
                prediction = indexMatrix[idx - 1];
            } else if (x == 0) {
                prediction = indexMatrix[(y-1) * width];
            } else {
                int W  = indexMatrix[idx - 1];
                int N  = indexMatrix[(y-1) * width + x];
                int NW = indexMatrix[(y-1) * width + (x-1)];
                if      (NW >= std::max(W, N)) prediction = std::min(W, N);
                else if (NW <= std::min(W, N)) prediction = std::max(W, N);
                else                           prediction = W + N - NW;
            }
            residuals[idx] = val - prediction + paletteSize;
        }
    }
    return residuals;
}


void saveSIF_perChannel(const std::string& path,
                        int width, int height,
                        // Base channels
                        const ChannelQuantization& baseL, const GradientData& gradsL,
                        const ChannelQuantization& baseA, const GradientData& gradsA,
                        const ChannelQuantization& baseB, const GradientData& gradsB,
                        // Residual channels
                        const ChannelResult& resL,
                        const ChannelResult& resA,
                        const ChannelResult& resB)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    struct RLESymbol { int value; int runLength; };

    // ─────────────────────────────────────────────────────────────────────────
    // encodeSection: palette + MED + RLE + Huffman
    // ─────────────────────────────────────────────────────────────────────────
    auto encodeSection = [&](
        const std::vector<float>& pal,
        const std::vector<int>& idxMatrix,
        std::vector<RLESymbol>& rleStream,
        std::map<int, std::pair<uint32_t,int>>& huffTable,
        int& maxRun, int& bitsPerIndex)
    {
        // Palette
        uint16_t palSize = (uint16_t)pal.size();
        file.write((char*)&palSize, 2);
        for (float p : pal) {
            uint16_t val = floatToHalf(p);
            file.write((char*)&val, 2);
        }

        bitsPerIndex = 1;
        while ((1 << bitsPerIndex) < (int)palSize) bitsPerIndex++;

        // MED prediction
        std::vector<int> encoded = applyMED_encode(idxMatrix, width, height, (int)palSize);
        int medRange = 2 * (int)palSize;

        // RLE
        if (!encoded.empty()) {
            int cur = encoded[0], run = 1;
            for (size_t i = 1; i < encoded.size(); i++) {
                if (encoded[i] == cur) { run++; }
                else { rleStream.push_back({cur, run}); cur = encoded[i]; run = 1; }
            }
            rleStream.push_back({cur, run});
        }

        maxRun = 1;
        for (auto& s : rleStream) maxRun = std::max(maxRun, s.runLength);

        auto pairKey = [&](int value, int run) {
            return value * (maxRun + 1) + (run - 1);
        };

        // Huffman
        std::map<int,int> freq;
        for (auto& s : rleStream) freq[pairKey(s.value, s.runLength)]++;

        if (freq.size() == 1) {
            huffTable[freq.begin()->first] = {0, 1};
        } else {
            std::priority_queue<Node*, std::vector<Node*>, Compare> pq;
            for (auto const& [id, f] : freq) pq.push(new Node(id, f));
            while (pq.size() > 1) {
                Node* left  = pq.top(); pq.pop();
                Node* right = pq.top(); pq.pop();
                Node* top   = new Node(-1, left->freq + right->freq);
                top->left = left; top->right = right;
                pq.push(top);
            }
            buildCodes(pq.top(), "", huffTable);
        }

        // Write metadata + bitstream
        uint16_t medRangeU = (uint16_t)medRange;
        file.write((char*)&medRangeU, 2);
        file.write((char*)&bitsPerIndex, 1);
        file.write((char*)&maxRun, 4);

        uint16_t tableEntries = (uint16_t)huffTable.size();
        file.write((char*)&tableEntries, 2);
        for (auto const& [key, code] : huffTable) {
            uint8_t len = (uint8_t)code.second;
            file.write((char*)&key,        4);
            file.write((char*)&len,        1);
            file.write((char*)&code.first, 4);
        }

        uint32_t rleCount = (uint32_t)rleStream.size();
        file.write((char*)&rleCount, 4);

        size_t totalBits = 0;
        for (auto& s : rleStream)
            totalBits += huffTable.at(pairKey(s.value, s.runLength)).second;
        uint32_t byteCount = (uint32_t)((totalBits + 7) / 8);
        file.write((char*)&byteCount, 4);

        BitWriter bw(file);
        for (auto& s : rleStream) {
            auto& [code, len] = huffTable[pairKey(s.value, s.runLength)];
            bw.write(code, len);
        }
        bw.flush();
    };

    // ─────────────────────────────────────────────────────────────────────────
    // encodeGradientSection: unchanged
    // ─────────────────────────────────────────────────────────────────────────
    auto encodeGradientSection = [&](const GradientData& grad) {
        int     precBits = (int)grad.precision;
        uint8_t precByte = (uint8_t)grad.precision;
        file.write((char*)&precByte, 1);

        uint32_t queueSize = (uint32_t)grad.queue.size();
        file.write((char*)&queueSize, 4);

        uint32_t gradByteCount = (uint32_t)(((uint64_t)queueSize * precBits + 7) / 8);
        file.write((char*)&gradByteCount, 4);

        BitWriter bwGrad(file);
        for (const auto& desc : grad.queue)
            bwGrad.write(desc.pack(grad.precision), precBits);
        bwGrad.flush();

        uint32_t cpCount = (uint32_t)grad.changePoints.size();
        file.write((char*)&cpCount, 4);
        for (const auto& cp : grad.changePoints) {
            file.write((char*)&cp.x,        2);
            file.write((char*)&cp.y,        2);
            file.write((char*)&cp.queueIdx, 4);
        }
    };

    struct ChannelStats {
        std::vector<RLESymbol> rle;
        std::map<int, std::pair<uint32_t,int>> huff;
        int maxRun = 1, bitsPerIndex = 1;
    };

    // Single helper: encodes palette+indices+gradients for any layer
    auto encodeLayer = [&](
        uint8_t magic,
        const std::vector<float>& pal,
        const std::vector<int>& idx,
        const GradientData& grad,
        ChannelStats& stats)
    {
        file.write((char*)&magic, 1);
        encodeSection(pal, idx, stats.rle, stats.huff, stats.maxRun, stats.bitsPerIndex);
        encodeGradientSection(grad);
    };

    // ── Header ────────────────────────────────────────────────────────────────
    uint32_t w = (uint32_t)width;
    uint32_t h = (uint32_t)height;
    file.write((char*)&w, 4);
    file.write((char*)&h, 4);

    // ── Base layers (0xC1 / 0xC2 / 0xC3) ─────────────────────────────────────
    ChannelStats statsL, statsA, statsB;
    encodeLayer(0xC1, baseL.palette, baseL.indexMatrix, gradsL, statsL);
    encodeLayer(0xC2, baseA.palette, baseA.indexMatrix, gradsA, statsA);
    encodeLayer(0xC3, baseB.palette, baseB.indexMatrix, gradsB, statsB);

    // ── Residual layers (0xD1 / 0xD2 / 0xD3) ─────────────────────────────────
    ChannelStats resStatsL, resStatsA, resStatsB;
    encodeLayer(0xD1, resL.palette, resL.indexMatrix, resL.gradients, resStatsL);
    encodeLayer(0xD2, resA.palette, resA.indexMatrix, resA.gradients, resStatsA);
    encodeLayer(0xD3, resB.palette, resB.indexMatrix, resB.gradients, resStatsB);

    file.close();

    // ─────────────────────────────────────────────────────────────────────────
    // Statistics
    // ─────────────────────────────────────────────────────────────────────────
    size_t fileSize    = std::filesystem::file_size(path);
    int    totalPixels = width * height;

    auto gradBytes = [&](const GradientData& grad) -> size_t {
        int pb = (int)grad.precision;
        return 1 + 4 + 4
               + ((grad.queue.size() * pb + 7) / 8)
               + 4 + grad.changePoints.size() * (2 + 2 + 4);
    };
    auto pairKeyFn = [](int value, int run, int maxRun) {
        return value * (maxRun + 1) + (run - 1);
    };
    auto rleBits = [&](const std::vector<RLESymbol>& rle,
                       const std::map<int,std::pair<uint32_t,int>>& huff,
                       int mRun) -> size_t {
        size_t bits = 0;
        for (auto& s : rle)
            bits += huff.at(pairKeyFn(s.value, s.runLength, mRun)).second;
        return bits;
    };
    auto palBytes1D = [](const std::vector<float>& pal) -> size_t {
        return 2 + pal.size() * 2;
    };

    size_t lBits    = rleBits(statsL.rle,    statsL.huff,    statsL.maxRun);
    size_t aBits    = rleBits(statsA.rle,    statsA.huff,    statsA.maxRun);
    size_t bBits    = rleBits(statsB.rle,    statsB.huff,    statsB.maxRun);
    size_t resLBits = rleBits(resStatsL.rle, resStatsL.huff, resStatsL.maxRun);
    size_t resABits = rleBits(resStatsA.rle, resStatsA.huff, resStatsA.maxRun);
    size_t resBBits = rleBits(resStatsB.rle, resStatsB.huff, resStatsB.maxRun);

    float bpp = (float)(fileSize * 8) / (float)totalPixels;

    auto pct       = [&](size_t bytes) { return (float)(bytes * 8) * 100.0f / (float)(fileSize * 8); };
    auto bitsPerPx = [&](size_t bytes) { return (float)(bytes * 8) / (float)totalPixels; };
    auto row = [&](const std::string& name, size_t bytes) {
        std::cout << "| " << std::left  << std::setw(22) << name
                  << "| " << std::right << std::setw(7)  << bytes
                  << " | "              << std::setw(5)  << std::fixed
                                        << std::setprecision(3) << bitsPerPx(bytes)
                  << "  | "             << std::setw(6)  << std::setprecision(2)
                                        << pct(bytes) << "%  |\n";
    };

    size_t lBytes    = 1 + palBytes1D(baseL.palette) + (lBits+7)/8    + gradBytes(gradsL);
    size_t aBytes    = 1 + palBytes1D(baseA.palette) + (aBits+7)/8    + gradBytes(gradsA);
    size_t bBytes    = 1 + palBytes1D(baseB.palette) + (bBits+7)/8    + gradBytes(gradsB);
    size_t resLBytes = 1 + palBytes1D(resL.palette)  + (resLBits+7)/8 + gradBytes(resL.gradients);
    size_t resABytes = 1 + palBytes1D(resA.palette)  + (resABits+7)/8 + gradBytes(resA.gradients);
    size_t resBBytes = 1 + palBytes1D(resB.palette)  + (resBBits+7)/8 + gradBytes(resB.gradients);

    size_t baseTotal   = lBytes + aBytes + bBytes;
    size_t resTotal    = resLBytes + resABytes + resBBytes;
    size_t headerBytes = 4 + 4;

    std::cout << "\n|-----------------------------------------------------------|\n";
    std::cout << "|             SIF Per-Channel File Breakdown               |\n";
    std::cout << "|-----------------------------------------------------------|\n";
    std::cout << "| Section               | Bytes   |  bpp   |    % file  |\n";
    std::cout << "|-----------------------------------------------------------|\n";
    row("Header",               headerBytes);
    std::cout << "|-----------------------------------------------------------|\n";
    row("L channel (base)",     lBytes);
    row("  - palette",          palBytes1D(baseL.palette));
    row("  - RLE stream",       (lBits + 7) / 8);
    row("  - gradients",        gradBytes(gradsL));
    std::cout << "|-----------------------------------------------------------|\n";
    row("A channel (base)",     aBytes);
    row("  - palette",          palBytes1D(baseA.palette));
    row("  - RLE stream",       (aBits + 7) / 8);
    row("  - gradients",        gradBytes(gradsA));
    std::cout << "|-----------------------------------------------------------|\n";
    row("B channel (base)",     bBytes);
    row("  - palette",          palBytes1D(baseB.palette));
    row("  - RLE stream",       (bBits + 7) / 8);
    row("  - gradients",        gradBytes(gradsB));
    std::cout << "|-----------------------------------------------------------|\n";
    row("BASE TOTAL",           baseTotal);
    std::cout << "|-----------------------------------------------------------|\n";
    row("Residual L",           resLBytes);
    row("  - palette",          palBytes1D(resL.palette));
    row("  - RLE stream",       (resLBits + 7) / 8);
    row("  - gradients",        gradBytes(resL.gradients));
    std::cout << "|-----------------------------------------------------------|\n";
    row("Residual A",           resABytes);
    row("  - palette",          palBytes1D(resA.palette));
    row("  - RLE stream",       (resABits + 7) / 8);
    row("  - gradients",        gradBytes(resA.gradients));
    std::cout << "|-----------------------------------------------------------|\n";
    row("Residual B",           resBBytes);
    row("  - palette",          palBytes1D(resB.palette));
    row("  - RLE stream",       (resBBits + 7) / 8);
    row("  - gradients",        gradBytes(resB.gradients));
    std::cout << "|-----------------------------------------------------------|\n";
    row("RESIDUAL TOTAL",       resTotal);
    std::cout << "|-----------------------------------------------------------|\n";
    row("TOTAL",                fileSize);
    std::cout << "|-----------------------------------------------------------|\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Image:           " << width << "x" << height
              << " (" << totalPixels << " pixels)\n";
    std::cout << "Original RGB:    " << (totalPixels * 3) / 1024 << " KB\n";
    std::cout << "SIF File:        " << fileSize / 1024          << " KB\n";
    std::cout << "Bits Per Pixel:  " << std::fixed << std::setprecision(3) << bpp << " bpp\n";
    std::cout << "Compression:     " << std::setprecision(2)
              << (float)(totalPixels * 24) / (float)(fileSize * 8) << ":1\n";
    std::cout << "Base  overhead:  " << std::setprecision(1)
              << (float)baseTotal * 100.0f / (float)fileSize << "% of file\n";
    std::cout << "Resid overhead:  "
              << (float)resTotal  * 100.0f / (float)fileSize << "% of file\n";
    std::cout << "Location: "
              << std::filesystem::absolute(path) << "\n";
}







// ─────────────────────────────────────────────────────────────────────────────
// subsampleMatrix
//
// Keeps only odd rows and odd columns (0-indexed), i.e. rows/cols 1, 3, 5, ...
// The original width and height are needed by the decoder to reconstruct.
//
// Input:  matrix[row * width + col],  size = width * height
// Output: matrix[row * outWidth + col], size = outWidth * outHeight
//
// outWidth  = width  / 2
// outHeight = height / 2
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> subsampleMatrix(
    const std::vector<float>& matrix,
    int width, int height,
    int& outWidth, int& outHeight)
{
    outWidth  = width  / 2;
    outHeight = height / 2;

    std::vector<float> result(outWidth * outHeight);

    for (int row = 0; row < outHeight; ++row) {
        for (int col = 0; col < outWidth; ++col) {
            float srcRow = row * 2 + 1; // odd rows: 1, 3, 5, ...
            float srcCol = col * 2 + 1; // odd cols: 1, 3, 5, ...
            result[row * outWidth + col] = matrix[srcRow * width + srcCol];
        }
    }

    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// upsampleMatrix
//
// Reconstructs a full matrix from a subsampled one.
// Even rows and even columns are filled by nearest-neighbour from the
// closest odd neighbour (the one that was kept during subsampling).
//
// origWidth / origHeight: the dimensions of the matrix BEFORE subsampling.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> upsampleMatrix(
    const std::vector<float>& subMatrix,
    int subWidth, int subHeight,
    int origWidth, int origHeight)
{
    std::vector<float> result(origWidth * origHeight);

    for (int row = 0; row < origHeight; ++row) {
        for (int col = 0; col < origWidth; ++col) {

            // Map back to the nearest odd source position
            int srcRow = (row - 1) / 2; // nearest odd row index in subMatrix
            int srcCol = (col - 1) / 2; // nearest odd col index in subMatrix

            // Clamp to valid subMatrix bounds
            srcRow = std::max(0, std::min(srcRow, subHeight - 1));
            srcCol = std::max(0, std::min(srcCol, subWidth  - 1));

            result[row * origWidth + col] = subMatrix[srcRow * subWidth + srcCol];
        }
    }

    return result;
}






std::vector<float> reconstructChannel(
    const ChannelQuantization& base,
    const GradientData& grads,
    const ChannelResult& residual,
    int width, int height)
{
    int totalPixels = width * height;

    // Base + gradients
    std::vector<LabF> recon(totalPixels);
    std::vector<PaletteEntry> tempPal;
    for (float p : base.palette) tempPal.push_back({p, 0, 0, 0});
    for (int i = 0; i < totalPixels; ++i)
        recon[i] = { base.palette[base.indexMatrix[i]], 0.0f, 0.0f };
    applyGradients(recon, base.indexMatrix, tempPal, grads, width, height);

    // Residuals
    std::vector<LabF> reconRes(totalPixels);
    std::vector<PaletteEntry> tempResPal;
    for (float p : residual.palette) tempResPal.push_back({p, 0, 0, 0});
    for (int i = 0; i < totalPixels; ++i)
        reconRes[i] = { residual.palette[residual.indexMatrix[i]], 0.0f, 0.0f };
    applyGradients(reconRes, residual.indexMatrix, tempResPal, residual.gradients, width, height);

    // Combine
    std::vector<float> result(totalPixels);
    for (int i = 0; i < totalPixels; ++i)
        result[i] = recon[i].L + reconRes[i].L;

    return result;
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


    // LAYER 1

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




    // LAYER 2

    // --- Process L (Luminance) ---

    // 1. Reconstruct previous channel
    auto reconL1 = reconstructChannel(baseL, gradsL, residualL, width, height);
    
    // 1. Subsample the base index matrix
    int subW, subH;
    auto subIdxL = subsampleMatrix(reconL1, width, height, subW, subH);

    // 2. Quantized subsampled matrix
    auto baseL2 = quantizeChannel(subIdxL, subW, subH, epsilonL, maxStepsL, "L_base2");

    // 3. Encode gradients on quantized sub matrix vs original sub channel values
    auto gradsL2 = encodeChannelGradients(baseL2.indexMatrix, subIdxL, subW, subH, "L_base2");

    // 4. Reconstruct at sub resolution (quantized + gradients) - stays as LabF for applyGradients
    std::vector<LabF> reconSubL(subW * subH);
    std::vector<PaletteEntry> tempPalL2;
    for (float p : baseL2.palette) tempPalL2.push_back({p, 0, 0, 0});
    for (int i = 0; i < subW * subH; ++i)
        reconSubL[i] = { baseL2.palette[baseL2.indexMatrix[i]], 0.0f, 0.0f };
    applyGradients(reconSubL, baseL2.indexMatrix, tempPalL2, gradsL2, subW, subH);

    // 4.5 Extract LabF::L into a flat float vector for upsampling
    std::vector<float> flatReconSubL(subW * subH);
    for (int i = 0; i < subW * subH; ++i) {
        flatReconSubL[i] = reconSubL[i].L;
    }

    // 5. Upsample the flat float vector
    auto upsampledIdxL2 = upsampleMatrix(flatReconSubL, subW, subH, width, height);

    auto residualL2 = encodeChannelResiduals_pro(reconL1, upsampledIdxL2, width, height, epsilonL, maxStepsL, "L");

    std::cout<<"Siamo qui ora"<<std::endl;

    auto reconL2 = reconstructChannel(baseL2, gradsL2 , residualL2, width, height);


    for (int i = 0; i < width * height; ++i)
        std::cout << reconL2[i] << "\n";





    







    



    /*

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






    saveSIF_perChannel("output.sif", width, height,
    baseL, gradsL,
    baseA, gradsA,
    baseB, gradsB,
    residualL, residualA, residualB);


    */










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