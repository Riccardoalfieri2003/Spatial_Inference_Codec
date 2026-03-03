#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include "sif_decoder_mod.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <random>
#include <ImageConverter_mod.hpp>
#include "PaletteEntry.hpp"
#include <GradientReconstruction_mod.hpp>


// ── Upsample: reconstruct missing rows and columns via averaging ──────────
std::vector<int> upsampleChannelMatrix(
    const std::vector<int>& sub,
    int subW, int subH,
    int origW, int origH,
    const std::vector<PaletteEntry>& palette)
{
    std::vector<int> full(origW * origH, 0);

    // Step 1: place known pixels at even positions
    for (int y = 0; y < subH; y++)
        for (int x = 0; x < subW; x++)
            full[(y * 2) * origW + (x * 2)] = sub[y * subW + x];

    // Helper: find nearest palette index to average of two palette values
    auto blendIdx = [&](int idxA, int idxB) -> int {
        float avg = (palette[idxA].L + palette[idxB].L) * 0.5f;
        int best = 0;
        float bestDist = std::abs(palette[0].L - avg);
        for (int i = 1; i < (int)palette.size(); i++) {
            float dist = std::abs(palette[i].L - avg);
            if (dist < bestDist) { bestDist = dist; best = i; }
        }
        return best;
    };

    // Step 2: fill horizontal gaps (odd columns, even rows)
    for (int y = 0; y < origH; y += 2) {
        for (int x = 1; x < origW - 1; x += 2) {
            int left  = full[y * origW + (x - 1)];
            int right = (x + 1 < origW) ? full[y * origW + (x + 1)] : left;
            full[y * origW + x] = blendIdx(left, right);
        }
        // Handle last column if origW is odd
        if (origW % 2 == 1)
            full[y * origW + (origW - 1)] = full[y * origW + (origW - 2)];
    }

    // Step 3: fill vertical gaps (all columns, odd rows)
    for (int y = 1; y < origH - 1; y += 2) {
        for (int x = 0; x < origW; x++) {
            int top    = full[(y - 1) * origW + x];
            int bottom = (y + 1 < origH) ? full[(y + 1) * origW + x] : top;
            full[y * origW + x] = blendIdx(top, bottom);
        }
    }
    // Handle last row if origH is odd
    if (origH % 2 == 1) {
        int y = origH - 1;
        for (int x = 0; x < origW; x++)
            full[y * origW + x] = full[(y - 1) * origW + x];
    }

    return full;
}


std::vector<int> applyMED_decode(
    const std::vector<int>& residuals,
    int width, int height,
    int paletteSize)
{
    std::vector<int> indexMatrix(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

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

                if (NW >= std::max(W, N))
                    prediction = std::min(W, N);
                else if (NW <= std::min(W, N))
                    prediction = std::max(W, N);
                else
                    prediction = W + N - NW;
            }

            indexMatrix[idx] = residuals[idx] - paletteSize + prediction;
        }
    }
    return indexMatrix;
}


// ── Load per-channel SIF ──────────────────────────────────────────────────
struct PerChannelData {
    int width = 0, height = 0;
    int subW  = 0, subH  = 0;
    std::vector<PaletteEntry> paletteL, paletteA, paletteB;
    std::vector<int> indexMatrixL, indexMatrixA, indexMatrixB;
    GradientData gradientsL, gradientsA, gradientsB;
    bool valid = false;
};

PerChannelData loadSIF_perChannel(const std::string& path, bool subsample) {
    PerChannelData result;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return result;
    }

    auto decodeSection = [&](
        std::vector<PaletteEntry>& pal,
        std::vector<int>& idxMatrix,
        int matW, int matH)
    {
        int totalPixels = matW * matH;

        uint16_t palSize = 0;
        file.read((char*)&palSize, 2);
        pal.resize(palSize);
        for (auto& p : pal) {
            uint16_t val;
            file.read((char*)&val, 2);
            p.L = halfToFloat(val);
            p.a = 0.0f; p.b = 0.0f; p.error = 0.0f;
        }

        // Read MED range
        uint16_t medRange = 0;
        file.read((char*)&medRange, 2);

        uint8_t  bitsPerIndex = 0;
        uint32_t maxRun       = 0;
        file.read((char*)&bitsPerIndex, 1);
        file.read((char*)&maxRun,       4);

        uint16_t tableEntries = 0;
        file.read((char*)&tableEntries, 2);

        DecodeNode* root = new DecodeNode();
        for (int i = 0; i < tableEntries; i++) {
            int32_t  key  = 0;
            uint8_t  len  = 0;
            uint32_t code = 0;
            file.read((char*)&key,  4);
            file.read((char*)&len,  1);
            file.read((char*)&code, 4);
            insertCode(root, code, (int)len, key);
        }

        uint32_t rleCount  = 0;
        uint32_t byteCount = 0;
        file.read((char*)&rleCount,  4);
        file.read((char*)&byteCount, 4);

        auto streamStart = file.tellg();

        // Decode RLE into MED residuals
        std::vector<int> medResiduals;
        medResiduals.reserve(totalPixels);
        BitReader br(file);
        for (uint32_t i = 0; i < rleCount; i++) {
            int pairKey = decodeNext(root, br);
            if (pairKey < 0) break;
            int value     =  pairKey / (int)(maxRun + 1);
            int runLength = (pairKey % (int)(maxRun + 1)) + 1;
            for (int r = 0; r < runLength; r++)
                medResiduals.push_back(value);
        }

        file.seekg(streamStart + (std::streamoff)byteCount);
        freeTree(root);

        // Reconstruct indices using MED decode
        idxMatrix = applyMED_decode(medResiduals, matW, matH, (int)(medRange / 2));
    };

    auto decodeGradientSection = [&](GradientData& gradients) {
        uint8_t precByte = 0;
        file.read((char*)&precByte, 1);
        gradients.precision = (GradientPrecision)precByte;
        int precBits = (int)gradients.precision;

        uint32_t queueSize = 0;
        uint32_t byteCount = 0;
        file.read((char*)&queueSize, 4);
        file.read((char*)&byteCount, 4);

        auto streamStart = file.tellg();

        BitReader brGrad(file);
        gradients.queue.reserve(queueSize);
        for (uint32_t i = 0; i < queueSize; i++) {
            uint8_t packed = (uint8_t)brGrad.read(precBits);
            gradients.queue.push_back(
                GradientDescriptor::unpack(packed, gradients.precision));
        }

        file.seekg(streamStart + (std::streamoff)byteCount);

        uint32_t cpCount = 0;
        file.read((char*)&cpCount, 4);
        gradients.changePoints.reserve(cpCount);
        for (uint32_t i = 0; i < cpCount; i++) {
            ChangePoint cp;
            file.read((char*)&cp.x,        2);
            file.read((char*)&cp.y,        2);
            file.read((char*)&cp.queueIdx, 4);
            gradients.changePoints.push_back(cp);
        }
        gradients.valid = true;
    };

    // ── 1. Header ─────────────────────────────────────────────────────────
    uint32_t w = 0, h = 0;
    file.read((char*)&w, 4);
    file.read((char*)&h, 4);
    result.width  = (int)w;
    result.height = (int)h;

    if (subsample) {
        uint32_t sw = 0, sh = 0;
        file.read((char*)&sw, 4);
        file.read((char*)&sh, 4);
        result.subW = (int)sw;
        result.subH = (int)sh;
        // ── Read the subsample flag byte the encoder wrote ─────────────────
        uint8_t subsampleFlag = 0;
        file.read((char*)&subsampleFlag, 1);
    } else {
        result.subW = result.width;
        result.subH = result.height;
    }

    int matW = result.subW;
    int matH = result.subH;

    std::cout << "Header: " << result.width << "x" << result.height
              << " sub: " << matW << "x" << matH << "\n";

    // ── 2. Three channel layers ────────────────────────────────────────────
    auto decodeChannelLayer = [&](
        uint8_t expectedMagic,
        std::vector<PaletteEntry>& pal,
        std::vector<int>& idxMatrix,
        GradientData& gradients)
    {
        uint8_t magic = 0;
        file.read((char*)&magic, 1);
        if (magic != expectedMagic) {
            std::cerr << "Expected magic 0x" << std::hex << (int)expectedMagic
                      << " got 0x" << (int)magic << std::dec << "\n";
            return;
        }
        decodeSection(pal, idxMatrix, matW, matH);
        decodeGradientSection(gradients);
        std::cout << "Channel decoded: palette=" << pal.size()
                  << " indices=" << idxMatrix.size() << "\n";
    };

    decodeChannelLayer(0xC1, result.paletteL, result.indexMatrixL, result.gradientsL);
    decodeChannelLayer(0xC2, result.paletteA, result.indexMatrixA, result.gradientsA);
    decodeChannelLayer(0xC3, result.paletteB, result.indexMatrixB, result.gradientsB);

    file.close();

    // ── 3. Upsample if subsampled ──────────────────────────────────────────
    if (subsample) {
        std::cout << "Upsampling channels...\n";
        result.indexMatrixL = upsampleChannelMatrix(
            result.indexMatrixL, matW, matH,
            result.width, result.height, result.paletteL);
        result.indexMatrixA = upsampleChannelMatrix(
            result.indexMatrixA, matW, matH,
            result.width, result.height, result.paletteA);
        result.indexMatrixB = upsampleChannelMatrix(
            result.indexMatrixB, matW, matH,
            result.width, result.height, result.paletteB);
    }

    result.valid = true;
    return result;
}


// ── Decoder main ──────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    std::string filePath = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_per_channel.sif";
    if (argc > 1) filePath = argv[1];

    bool subsample = false;  // ← toggle manually to match encoder setting

    std::cout << "Loading: " << filePath << "\n";

    PerChannelData data = loadSIF_perChannel(filePath, subsample);
    if (!data.valid) {
        std::cerr << "Failed to decode SIF file.\n";
        return 1;
    }

    int totalPixels = data.width * data.height;

    // ── Apply gradients per channel ───────────────────────────────────────
    // Build LabF images — each channel only uses the .L field
    std::vector<LabF> reconL(totalPixels), reconA(totalPixels), reconB(totalPixels);

    for (int i = 0; i < totalPixels; i++) {
        reconL[i] = { data.paletteL[data.indexMatrixL[i]].L, 0.0f, 0.0f };
        reconA[i] = { data.paletteA[data.indexMatrixA[i]].L, 0.0f, 0.0f };
        reconB[i] = { data.paletteB[data.indexMatrixB[i]].L, 0.0f, 0.0f };
    }

    applyGradients(reconL, data.indexMatrixL, data.paletteL,
                   data.gradientsL, data.width, data.height);
    applyGradients(reconA, data.indexMatrixA, data.paletteA,
                   data.gradientsA, data.width, data.height);
    applyGradients(reconB, data.indexMatrixB, data.paletteB,
                   data.gradientsB, data.width, data.height);

    // ── Combine channels ──────────────────────────────────────────────────
    std::vector<LabF> finalImage(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
        finalImage[i] = { reconL[i].L, reconA[i].L, reconB[i].L };
    }

    // ── Convert Lab → RGB and save ────────────────────────────────────────
    std::vector<unsigned char> pixels(totalPixels * 3);
    for (int i = 0; i < totalPixels; i++) {
        ImageConverter::convertPixelLabToRGB(
            finalImage[i].L, finalImage[i].a, finalImage[i].b,
            pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
    }

    std::string outPath = filePath + "_reconstructed.png";
    int success = stbi_write_png(outPath.c_str(),
                                 data.width, data.height, 3,
                                 pixels.data(), data.width * 3);
    if (success)
        std::cout << "\n[SUCCESS] Saved to: " << outPath << "\n";
    else {
        std::cerr << "Failed to write PNG.\n";
        return 1;
    }

    return 0;
}