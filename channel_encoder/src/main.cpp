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


// ── Median Edge Detector (MED/LOCO-I predictor) ───────────────────────────
// For each pixel, predicts its value from west (W), north (N), northwest (NW)
// neighbors, then returns the prediction error (residual).
// Encoding: store residuals instead of raw values
// Decoding: reconstruct by adding prediction back to residual

// ── Subsample L channel: keep even rows and columns (0,2,4...) ───────────
std::vector<int> subsampleChannelMatrix_even(
    const std::vector<int>& indexMatrix,
    int width, int height,
    int& subW, int& subH)
{
    subW = width  / 2;
    subH = height / 2;

    std::vector<int> sub(subW * subH);
    for (int y = 0; y < subH; y++)
        for (int x = 0; x < subW; x++)
            sub[y * subW + x] = indexMatrix[(y * 2) * width + (x * 2)];

    return sub;
}

// ── Subsample A/B channels: keep odd rows and columns (1,3,5...) ──────────
std::vector<int> subsampleChannelMatrix_odd(
    const std::vector<int>& indexMatrix,
    int width, int height,
    int& subW, int& subH)
{
    subW = (width  - 1) / 2;
    subH = (height - 1) / 2;

    std::vector<int> sub(subW * subH);
    for (int y = 0; y < subH; y++)
        for (int x = 0; x < subW; x++)
            sub[y * subW + x] = indexMatrix[(y * 2 + 1) * width + (x * 2 + 1)];

    return sub;
}

// ── Upsample L channel (was even-sampled) ────────────────────────────────
std::vector<int> upsampleChannelMatrix_even(
    const std::vector<int>& sub,
    int subW, int subH,
    int origW, int origH,
    const std::vector<PaletteEntry>& palette)
{
    std::vector<int> full(origW * origH, 0);

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

    // Place known pixels at even positions
    for (int y = 0; y < subH; y++)
        for (int x = 0; x < subW; x++)
            full[(y * 2) * origW + (x * 2)] = sub[y * subW + x];

    // Fill horizontal gaps (odd columns, even rows)
    for (int y = 0; y < origH; y += 2) {
        for (int x = 1; x < origW - 1; x += 2) {
            int left  = full[y * origW + (x - 1)];
            int right = full[y * origW + std::min(x + 1, origW - 1)];
            full[y * origW + x] = blendIdx(left, right);
        }
        if (origW % 2 == 0)
            full[y * origW + (origW - 1)] = full[y * origW + (origW - 2)];
    }

    // Fill vertical gaps (odd rows)
    for (int y = 1; y < origH - 1; y += 2) {
        for (int x = 0; x < origW; x++) {
            int top    = full[(y - 1) * origW + x];
            int bottom = full[std::min(y + 1, origH - 1) * origW + x];
            full[y * origW + x] = blendIdx(top, bottom);
        }
    }
    if (origH % 2 == 0)
        for (int x = 0; x < origW; x++)
            full[(origH - 1) * origW + x] = full[(origH - 2) * origW + x];

    return full;
}

// ── Upsample A/B channels (was odd-sampled) ──────────────────────────────
std::vector<int> upsampleChannelMatrix_odd(
    const std::vector<int>& sub,
    int subW, int subH,
    int origW, int origH,
    const std::vector<PaletteEntry>& palette)
{
    std::vector<int> full(origW * origH, 0);

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

    // Place known pixels at odd positions
    for (int y = 0; y < subH; y++)
        for (int x = 0; x < subW; x++)
            full[(y * 2 + 1) * origW + (x * 2 + 1)] = sub[y * subW + x];

    // Fill horizontal gaps (even columns, odd rows)
    for (int y = 1; y < origH; y += 2) {
        // First column (x=0): copy from x=1 if available
        full[y * origW + 0] = (origW > 1) ? full[y * origW + 1] : 0;
        for (int x = 2; x < origW - 1; x += 2) {
            int left  = full[y * origW + (x - 1)];
            int right = full[y * origW + std::min(x + 1, origW - 1)];
            full[y * origW + x] = blendIdx(left, right);
        }
        if (origW % 2 == 1)
            full[y * origW + (origW - 1)] = full[y * origW + (origW - 2)];
    }

    // Fill vertical gaps (even rows)
    for (int y = 0; y < origH; y += 2) {
        for (int x = 0; x < origW; x++) {
            int top    = (y > 0)         ? full[(y - 1) * origW + x] : full[(y + 1) * origW + x];
            int bottom = (y + 1 < origH) ? full[(y + 1) * origW + x] : top;
            full[y * origW + x] = blendIdx(top, bottom);
        }
    }

    return full;
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
                if      (NW >= std::max(W, N)) prediction = std::min(W, N);
                else if (NW <= std::min(W, N)) prediction = std::max(W, N);
                else                           prediction = W + N - NW;
            }
            indexMatrix[idx] = residuals[idx] - paletteSize + prediction;
        }
    }
    return indexMatrix;
}

void saveSIF_perChannel(const std::string& path,
                        int width, int height,
                        const std::vector<PaletteEntry>& paletteL,
                        const std::vector<int>& indexMatrixL,
                        const GradientData& gradientsL,
                        const std::vector<PaletteEntry>& paletteA,
                        const std::vector<int>& indexMatrixA,
                        const GradientData& gradientsA,
                        const std::vector<PaletteEntry>& paletteB,
                        const std::vector<int>& indexMatrixB,
                        const GradientData& gradientsB,
                        const std::vector<PaletteEntry>& resPaletteL,
                        const std::vector<int>& resIdxL,
                        const std::vector<PaletteEntry>& resPaletteA,
                        const std::vector<int>& resIdxA,
                        const std::vector<PaletteEntry>& resPaletteB,
                        const std::vector<int>& resIdxB,
                        bool subsample)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    // ── Subsample internally ──────────────────────────────────────────────────
    int subWL = width, subHL = height;
    int subWAB = width, subHAB = height;
    std::vector<int> idxL, idxA, idxB;

    if (subsample) {
        idxL = subsampleChannelMatrix_odd (indexMatrixL, width, height, subWL,  subHL);
        idxA = subsampleChannelMatrix_even(indexMatrixA, width, height, subWAB, subHAB);
        idxB = subsampleChannelMatrix_even(indexMatrixB, width, height, subWAB, subHAB);
        std::cout << "L  (odd):  " << width << "x" << height << " -> " << subWL  << "x" << subHL  << "\n";
        std::cout << "AB (even): " << width << "x" << height << " -> " << subWAB << "x" << subHAB << "\n";
    } else {
        idxL = indexMatrixL;
        idxA = indexMatrixA;
        idxB = indexMatrixB;
    }

    int matWL  = subsample ? subWL  : width;
    int matHL  = subsample ? subHL  : height;
    int matWAB = subsample ? subWAB : width;
    int matHAB = subsample ? subHAB : height;

    struct RLESymbol { int value; int runLength; };

    auto encodeSection = [&](
        const std::vector<PaletteEntry>& pal,
        const std::vector<int>& idxMatrix,
        std::vector<RLESymbol>& rleStream,
        std::map<int, std::pair<uint32_t,int>>& huffTable,
        int& maxRun, int& bitsPerIndex,
        int matW, int matH)
    {
        uint16_t palSize = (uint16_t)pal.size();
        file.write((char*)&palSize, 2);
        for (const auto& p : pal) {
            uint16_t val = floatToHalf(p.L);
            file.write((char*)&val, 2);
        }

        bitsPerIndex = 1;
        while ((1 << bitsPerIndex) < palSize) bitsPerIndex++;

        std::vector<int> encoded = applyMED_encode(idxMatrix, matW, matH, (int)palSize);
        int medRange = 2 * (int)palSize;

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

        uint16_t medRangeU = (uint16_t)medRange;
        file.write((char*)&medRangeU, 2);
        file.write((char*)&bitsPerIndex, 1);
        file.write((char*)&maxRun, 4);

        uint16_t tableEntries = (uint16_t)huffTable.size();
        file.write((char*)&tableEntries, 2);
        for (auto const& [key, code] : huffTable) {
            uint8_t len = (uint8_t)code.second;
            file.write((char*)&key, 4);
            file.write((char*)&len, 1);
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
            int key = pairKey(s.value, s.runLength);
            auto& [code, len] = huffTable[key];
            bw.write(code, len);
        }
        bw.flush();
    };

    auto encodeGradientSection = [&](const GradientData& grad) {
        int precBits = (int)grad.precision;
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
            file.write((char*)&cp.x, 2);
            file.write((char*)&cp.y, 2);
            file.write((char*)&cp.queueIdx, 4);
        }
    };

    struct ChannelStats {
        std::vector<RLESymbol> rle;
        std::map<int, std::pair<uint32_t,int>> huff;
        int maxRun = 1, bitsPerIndex = 1;
    };

    auto encodeChannelLayer = [&](
        uint8_t magic,
        const std::vector<PaletteEntry>& pal,
        const std::vector<int>& idx,
        const GradientData& grad,
        ChannelStats& stats,
        int matW, int matH)
    {
        file.write((char*)&magic, 1);
        encodeSection(pal, idx, stats.rle, stats.huff,
                      stats.maxRun, stats.bitsPerIndex, matW, matH);
        encodeGradientSection(grad);
    };

    auto encodeResidualLayer = [&](
        uint8_t magic,
        const std::vector<PaletteEntry>& pal,
        const std::vector<int>& idx,
        ChannelStats& stats)
    {
        file.write((char*)&magic, 1);
        encodeSection(pal, idx, stats.rle, stats.huff,
                      stats.maxRun, stats.bitsPerIndex, width, height);
    };

    // ── Header ────────────────────────────────────────────────────────────────
    uint32_t w = width, h = height;
    file.write((char*)&w, 4);
    file.write((char*)&h, 4);
    if (subsample) {
        uint32_t swL  = subWL,  shL  = subHL;
        uint32_t swAB = subWAB, shAB = subHAB;
        file.write((char*)&swL,  4);
        file.write((char*)&shL,  4);
        file.write((char*)&swAB, 4);
        file.write((char*)&shAB, 4);
    }

    // ── Main channel layers ───────────────────────────────────────────────────
    ChannelStats statsL, statsA, statsB;
    encodeChannelLayer(0xC1, paletteL, idxL, gradientsL, statsL, matWL,  matHL);
    encodeChannelLayer(0xC2, paletteA, idxA, gradientsA, statsA, matWAB, matHAB);
    encodeChannelLayer(0xC3, paletteB, idxB, gradientsB, statsB, matWAB, matHAB);

    // ── Residual layers (full resolution, no gradients) ───────────────────────
    ChannelStats resStatsL, resStatsA, resStatsB;
    encodeResidualLayer(0xD1, resPaletteL, resIdxL, resStatsL);
    encodeResidualLayer(0xD2, resPaletteA, resIdxA, resStatsA);
    encodeResidualLayer(0xD3, resPaletteB, resIdxB, resStatsB);

    file.close();

    // ── Statistics ────────────────────────────────────────────────────────────
    size_t fileSize    = std::filesystem::file_size(path);
    int    totalPixels = width * height;

    auto gradBytes = [&](const GradientData& grad) -> size_t {
        int pb = (int)grad.precision;
        return 1 + 4 + 4 + ((grad.queue.size() * pb + 7) / 8)
                 + 4 + grad.changePoints.size() * (2+2+4);
    };
    auto pairKeyFn = [](int value, int run, int maxRun) {
        return value * (maxRun + 1) + (run - 1);
    };
    auto rleBits = [&](const std::vector<RLESymbol>& rle,
                       const std::map<int,std::pair<uint32_t,int>>& huff,
                       int maxRun) -> size_t {
        size_t bits = 0;
        for (auto& s : rle)
            bits += huff.at(pairKeyFn(s.value, s.runLength, maxRun)).second;
        return bits;
    };
    auto palBytes1D = [](const std::vector<PaletteEntry>& pal) -> size_t {
        return 2 + pal.size() * 2;
    };

    size_t lBits    = rleBits(statsL.rle,    statsL.huff,    statsL.maxRun);
    size_t aBits    = rleBits(statsA.rle,    statsA.huff,    statsA.maxRun);
    size_t bBits    = rleBits(statsB.rle,    statsB.huff,    statsB.maxRun);
    size_t resLBits = rleBits(resStatsL.rle, resStatsL.huff, resStatsL.maxRun);
    size_t resABits = rleBits(resStatsA.rle, resStatsA.huff, resStatsA.maxRun);
    size_t resBBits = rleBits(resStatsB.rle, resStatsB.huff, resStatsB.maxRun);

    float bpp = (float)(fileSize * 8) / totalPixels;
    auto pct       = [&](size_t b) { return (float)b * 100.0f / (float)fileSize; };
    auto bitsPerPx = [&](size_t b) { return (float)(b * 8) / (float)totalPixels; };
    auto row = [&](const std::string& name, size_t bytes) {
        std::cout << "| " << std::left  << std::setw(19) << name
                  << "| "  << std::right << std::setw(7)  << bytes
                  << " | "              << std::setw(5)   << std::fixed
                                        << std::setprecision(3) << bitsPerPx(bytes)
                  << "  | "             << std::setw(7)   << std::setprecision(2)
                                        << pct(bytes) << "%  |\n";
    };

    size_t headerBytes = 4 + 4 + (subsample ? 16 : 0);

    std::cout << "\n|------------------------------------------------------|\n";
    std::cout << "|          SIF Per-Channel File Breakdown             |\n";
    std::cout << "|------------------------------------------------------|\n";
    std::cout << "| Section            | Bytes   |  bpp   |   % file  |\n";
    std::cout << "|------------------------------------------------------|\n";
    row("Header",             headerBytes);
    row("L channel",          1 + palBytes1D(paletteL) + (lBits+7)/8    + gradBytes(gradientsL));
    row("  - palette",        palBytes1D(paletteL));
    row("  - RLE stream",     (lBits + 7) / 8);
    row("  - gradients",      gradBytes(gradientsL));
    row("A channel",          1 + palBytes1D(paletteA) + (aBits+7)/8    + gradBytes(gradientsA));
    row("  - palette",        palBytes1D(paletteA));
    row("  - RLE stream",     (aBits + 7) / 8);
    row("  - gradients",      gradBytes(gradientsA));
    row("B channel",          1 + palBytes1D(paletteB) + (bBits+7)/8    + gradBytes(gradientsB));
    row("  - palette",        palBytes1D(paletteB));
    row("  - RLE stream",     (bBits + 7) / 8);
    row("  - gradients",      gradBytes(gradientsB));
    row("Residual L",         1 + palBytes1D(resPaletteL) + (resLBits+7)/8);
    row("  - palette",        palBytes1D(resPaletteL));
    row("  - RLE stream",     (resLBits + 7) / 8);
    row("Residual A",         1 + palBytes1D(resPaletteA) + (resABits+7)/8);
    row("  - palette",        palBytes1D(resPaletteA));
    row("  - RLE stream",     (resABits + 7) / 8);
    row("Residual B",         1 + palBytes1D(resPaletteB) + (resBBits+7)/8);
    row("  - palette",        palBytes1D(resPaletteB));
    row("  - RLE stream",     (resBBits + 7) / 8);
    std::cout << "|------------------------------------------------------|\n";
    row("TOTAL",              fileSize);
    std::cout << "|------------------------------------------------------|\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Image:           " << width << "x" << height << " (" << totalPixels << " pixels)\n";
    std::cout << "Original RGB:    " << (totalPixels * 3) / 1024 << " KB\n";
    std::cout << "SIF File:        " << fileSize / 1024          << " KB\n";
    std::cout << "Bits Per Pixel:  " << bpp                      << " bpp\n";
    std::cout << "Compression:     " << (float)(totalPixels*24) / (fileSize*8) << ":1\n";
    std::cout << "Location: "        << std::filesystem::absolute(path)        << "\n";
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

    // ── Helpers ───────────────────────────────────────────────────────────────
    struct ChannelPaletteEntry { float value; float error; };

    auto quantizeChannel = [&](
        const std::vector<float>& channel,
        float epsilon,
        std::vector<ChannelPaletteEntry>& palette,
        std::vector<int>& indexMatrix)
    {
        struct VoxelAccum { float sum = 0; int count = 0; float minV = 1e9, maxV = -1e9; };
        std::map<int, VoxelAccum> voxels;
        for (int i = 0; i < totalPixels; i++) {
            int key = static_cast<int>(std::floor(channel[i] / epsilon));
            auto& v = voxels[key];
            v.sum += channel[i]; v.count++;
            v.minV = std::min(v.minV, channel[i]);
            v.maxV = std::max(v.maxV, channel[i]);
        }
        std::map<int, int> voxelToIdx;
        int idx = 0;
        for (auto& [key, accum] : voxels) {
            float centroid = accum.sum / accum.count;
            float error    = std::max(centroid - accum.minV, accum.maxV - centroid);
            palette.push_back({ centroid, error });
            voxelToIdx[key] = idx++;
        }
        indexMatrix.resize(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            int key = static_cast<int>(std::floor(channel[i] / epsilon));
            indexMatrix[i] = voxelToIdx[key];
        }
        std::cout << "  Clusters: " << palette.size() << "\n";
    };

    auto channelPalToPaletteEntry = [](const std::vector<ChannelPaletteEntry>& cp) {
        std::vector<PaletteEntry> pe(cp.size());
        for (size_t i = 0; i < cp.size(); i++)
            pe[i] = { cp[i].value, 0.0f, 0.0f, cp[i].error };
        return pe;
    };

    auto channelToLabFlat = [&](const std::vector<float>& ch) {
        std::vector<LabPixelFlat> flat(totalPixels);
        for (int i = 0; i < totalPixels; i++)
            flat[i] = { ch[i], 0.0f, 0.0f };
        return flat;
    };

    auto makeGradients = [&](
        const std::vector<int>& idxMatrix,
        const std::vector<LabPixelFlat>& flat) -> GradientData
    {
        GradientData g = encodeGradients(idxMatrix, flat, width, height,
                                          GradientPrecision::BITS_2, 1.0f, 32);
        for (auto& d : g.queue) d.shape = 1;
        g.valid = true;
        return g;
    };

    auto reconstructChannel = [&](
        const std::vector<int>& idxMatrix,
        const std::vector<PaletteEntry>& palette,
        const GradientData& gradients) -> std::vector<LabF>
    {
        std::vector<LabF> recon(totalPixels);
        for (int i = 0; i < totalPixels; i++)
            recon[i] = { palette[idxMatrix[i]].L, 0.0f, 0.0f };
        applyGradients(recon, idxMatrix, palette, gradients, width, height);
        return recon;
    };





    // ══════════════════════════════════════════════════════════════════════════
    // L CHANNEL
    // ══════════════════════════════════════════════════════════════════════════
    std::cout << "\n--- L Channel ---\n";

    std::vector<ChannelPaletteEntry> paletteL;
    std::vector<int> indexMatrixL;
    quantizeChannel(channelL, epsilonL, paletteL, indexMatrixL);
    std::vector<PaletteEntry> palEntL = channelPalToPaletteEntry(paletteL);

    // Subsample + reconstruct to get decoder-side index matrix
    std::vector<int> reconIdxL = indexMatrixL;
    if (subsample) {
        int subWL, subHL;
        auto subL  = subsampleChannelMatrix_odd(indexMatrixL, width, height, subWL, subHL);
        reconIdxL  = upsampleChannelMatrix_odd (subL, subWL, subHL, width, height, palEntL);
    }

    // Encode gradients on reconstructed matrix
    auto flatL      = channelToLabFlat(channelL);
    auto gradientsL = makeGradients(reconIdxL, flatL);

    // Reconstruct L for residual computation
    auto reconL = reconstructChannel(reconIdxL, palEntL, gradientsL);

    // Compute L residuals
    std::vector<float> residualL(totalPixels);
    for (int i = 0; i < totalPixels; i++)
        residualL[i] = channelL[i] - reconL[i].L;

    std::cout << "L residual range: ["
              << *std::min_element(residualL.begin(), residualL.end()) << ", "
              << *std::max_element(residualL.begin(), residualL.end()) << "]\n";

    // ── Quantize L residuals ──────────────────────────────────────────────────
    std::vector<ChannelPaletteEntry> resPaletteL_raw;
    std::vector<int> resIdxL;

    float epsilonResL = 3.0f;

    quantizeChannel(residualL, epsilonResL, resPaletteL_raw, resIdxL);
    std::vector<PaletteEntry> resPaletteL = channelPalToPaletteEntry(resPaletteL_raw);

    std::cout << "L residual clusters: " << resPaletteL.size() << "\n";
    std::cout << "L residual palette range: ["
              << resPaletteL.front().L << ", "
              << resPaletteL.back().L  << "]\n";







    // ══════════════════════════════════════════════════════════════════════════
    // A CHANNEL
    // ══════════════════════════════════════════════════════════════════════════
    std::cout << "\n--- A Channel ---\n";

    std::vector<ChannelPaletteEntry> paletteA;
    std::vector<int> indexMatrixA;
    quantizeChannel(channelA, epsilonA, paletteA, indexMatrixA);
    std::vector<PaletteEntry> palEntA = channelPalToPaletteEntry(paletteA);

    // Subsample + reconstruct to get decoder-side index matrix
    std::vector<int> reconIdxA = indexMatrixA;
    if (subsample) {
        int subWA, subHA;
        auto subA  = subsampleChannelMatrix_odd(indexMatrixA, width, height, subWA, subHA);
        reconIdxA  = upsampleChannelMatrix_odd (subA, subWA, subHA, width, height, palEntA);
    }

    // Encode gradients on reconstructed matrix
    auto flatA      = channelToLabFlat(channelA);
    auto gradientsA = makeGradients(reconIdxA, flatA);

    // Reconstruct A for residual computation
    auto reconA = reconstructChannel(reconIdxA, palEntA, gradientsA);

    // Compute A residuals
    std::vector<float> residualA(totalPixels);
    for (int i = 0; i < totalPixels; i++)
        residualA[i] = channelA[i] - reconA[i].a;

    std::cout << "A residual range: ["
              << *std::min_element(residualA.begin(), residualA.end()) << ", "
              << *std::max_element(residualA.begin(), residualA.end()) << "]\n";

    // ── Quantize A residuals ──────────────────────────────────────────────────
    std::vector<ChannelPaletteEntry> resPaletteA_raw;
    std::vector<int> resIdxA;

    float epsilonResA = 3.0f;

    quantizeChannel(residualA, epsilonResA, resPaletteA_raw, resIdxA);
    std::vector<PaletteEntry> resPaletteA = channelPalToPaletteEntry(resPaletteA_raw);

    std::cout << "A residual clusters: " << resPaletteL.size() << "\n";
    std::cout << "A residual palette range: ["
              << resPaletteA.front().a << ", "
              << resPaletteA.back().a << "]\n";





    



    // ══════════════════════════════════════════════════════════════════════════
    // B CHANNEL
    // ══════════════════════════════════════════════════════════════════════════
    std::cout << "\n--- B Channel ---\n";

    std::vector<ChannelPaletteEntry> paletteB;
    std::vector<int> indexMatrixB;
    quantizeChannel(channelB, epsilonB, paletteB, indexMatrixB);
    std::vector<PaletteEntry> palEntB = channelPalToPaletteEntry(paletteB);

    // Subsample + reconstruct to get decoder-side index matrix
    std::vector<int> reconIdxB = indexMatrixB;
    if (subsample) {
        int subWB, subHB;
        auto subB  = subsampleChannelMatrix_odd(indexMatrixB, width, height, subWB, subHB);
        reconIdxB  = upsampleChannelMatrix_odd (subB, subWB, subHB, width, height, palEntB);
    }

    // Encode gradients on reconstructed matrix
    auto flatB      = channelToLabFlat(channelB);
    auto gradientsB = makeGradients(reconIdxL, flatB);

    // Reconstruct B for residual computation
    auto reconB = reconstructChannel(reconIdxB, palEntB, gradientsB);

    // Compute B residuals
    std::vector<float> residualB(totalPixels);
    for (int i = 0; i < totalPixels; i++)
        residualB[i] = channelB[i] - reconB[i].b;

    std::cout << "B residual range: ["
              << *std::min_element(residualB.begin(), residualB.end()) << ", "
              << *std::max_element(residualB.begin(), residualB.end()) << "]\n";

    // ── Quantize B residuals ──────────────────────────────────────────────────
    std::vector<ChannelPaletteEntry> resPaletteB_raw;
    std::vector<int> resIdxB;

    float epsilonResB = 3.0f;

    quantizeChannel(residualB, epsilonResB, resPaletteB_raw, resIdxB);
    std::vector<PaletteEntry> resPaletteB = channelPalToPaletteEntry(resPaletteB_raw);

    std::cout << "B residual clusters: " << resPaletteL.size() << "\n";
    std::cout << "B residual palette range: ["
              << resPaletteL.front().b << ", "
              << resPaletteL.back().b  << "]\n";







    // ══════════════════════════════════════════════════════════════════════════
    // COMBINE + VISUALIZE
    // ══════════════════════════════════════════════════════════════════════════
    std::vector<LabF> fullRecon(totalPixels);
    for (int i = 0; i < totalPixels; i++)
        fullRecon[i] = { reconL[i].L, reconA[i].L, reconB[i].L };

    auto saveLabImage = [&](const std::vector<LabF>& img, const std::string& fname) {
        std::vector<unsigned char> buf(totalPixels * 3);
        for (int i = 0; i < totalPixels; i++) {
            ImageConverter::convertPixelLabToRGB(
                img[i].L, img[i].a, img[i].b,
                buf[i*3], buf[i*3+1], buf[i*3+2]);
        }
        stbi_write_png(fname.c_str(), width, height, 3, buf.data(), width * 3);
        std::cout << "Saved: " << fname << "\n";
    };

    std::vector<LabF> original(totalPixels);
    for (int i = 0; i < totalPixels; i++)
        original[i] = { channelL[i], channelA[i], channelB[i] };

    saveLabImage(original,  "out_original.png");
    saveLabImage(fullRecon, "out_per_channel_reconstruction.png");

    // ══════════════════════════════════════════════════════════════════════════
    // SAVE
    // ══════════════════════════════════════════════════════════════════════════
    saveSIF_perChannel("output_per_channel.sif",
                   width, height,
                   palEntL, indexMatrixL, gradientsL,
                   palEntA, indexMatrixA, gradientsA,
                   palEntB, indexMatrixB, gradientsB,
                   resPaletteL, resIdxL,
                   resPaletteA, resIdxA,
                   resPaletteB, resIdxB,
                   subsample);  // subsample

    std::cout << "\n--- Per-Channel Summary ---\n";
    std::cout << "L palette: " << paletteL.size()
              << "  A palette: " << paletteA.size()
              << "  B palette: " << paletteB.size() << "\n";

    stbi_image_free(imgData);
    return 0;
}