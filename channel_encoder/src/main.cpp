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



// ── Subsample: keep only even-indexed rows and columns ────────────────────
// Output dimensions: subW = width/2, subH = height/2
std::vector<int> subsampleChannelMatrix(
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
                    
                        bool subsample)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    // ── Optionally subsample all three channel matrices ────────────────────
    int subW = width, subH = height;
    std::vector<int> idxL, idxA, idxB;

    if (subsample) {
        idxL = subsampleChannelMatrix(indexMatrixL, width, height, subW, subH);
        idxA = subsampleChannelMatrix(indexMatrixA, width, height, subW, subH);
        idxB = subsampleChannelMatrix(indexMatrixB, width, height, subW, subH);
        std::cout << "Subsampled: " << width << "x" << height 
                  << " -> " << subW << "x" << subH << "\n";
    } else {
        idxL = indexMatrixL;
        idxA = indexMatrixA;
        idxB = indexMatrixB;
    }

    struct RLESymbol { int value; int runLength; };

    // ── encodeSection: now stores only 1 float16 per palette entry ────────────
    auto encodeSection = [&](
        const std::vector<PaletteEntry>& pal,
        const std::vector<int>& idxMatrix,
        std::vector<RLESymbol>& rleStream,
        std::map<int, std::pair<uint32_t,int>>& huffTable,
        int& maxRun, int& bitsPerIndex)
    {
        uint16_t palSize = (uint16_t)pal.size();
        file.write((char*)&palSize, 2);
        for (const auto& p : pal) {
            uint16_t val = floatToHalf(p.L);
            file.write((char*)&val, 2);
        }

        bitsPerIndex = 1;
        while ((1 << bitsPerIndex) < palSize) bitsPerIndex++;

        // ── Delta prediction before RLE ───────────────────────────────────────
        // Instead of encoding raw indices, encode the difference between
        // each index and the previous one. This creates longer runs of 0s
        // in smooth regions, improving RLE compression significantly.
        // We use palette value differences to find the nearest palette entry
        // for each delta — but for simplicity, just use index deltas directly.
        // The delta is stored as: delta + palSize (to make it positive)
        // so range is [0, 2*palSize-1]
        std::vector<int> deltaMatrix(idxMatrix.size());
        deltaMatrix[0] = idxMatrix[0];  // first value stored as-is
        for (size_t i = 1; i < idxMatrix.size(); i++) {
            deltaMatrix[i] = idxMatrix[i] - idxMatrix[i-1] + (int)palSize;
        }
        int deltaRange = 2 * (int)palSize;  // range of delta values

        // ── RLE on delta stream ───────────────────────────────────────────────
        if (!deltaMatrix.empty()) {
            int cur = deltaMatrix[0], run = 1;
            for (size_t i = 1; i < deltaMatrix.size(); i++) {
                if (deltaMatrix[i] == cur) { run++; }
                else { rleStream.push_back({cur, run}); cur = deltaMatrix[i]; run = 1; }
            }
            rleStream.push_back({cur, run});
        }

        maxRun = 1;
        for (auto& s : rleStream) maxRun = std::max(maxRun, s.runLength);

        auto pairKey = [&](int value, int run) {
            return value * (maxRun + 1) + (run - 1);
        };

        // ── Huffman entropy coding ────────────────────────────────────────────
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

        // Write delta range so decoder can reconstruct
        uint16_t deltaRangeU = (uint16_t)deltaRange;
        file.write((char*)&deltaRangeU, 2);

        file.write((char*)&bitsPerIndex, 1);
        file.write((char*)&maxRun,       4);

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

    auto encodeChannelLayer = [&](
        uint8_t magic,
        const std::vector<PaletteEntry>& pal,
        const std::vector<int>& idxMatrix,
        const GradientData& grad,
        ChannelStats& stats)
    {
        file.write((char*)&magic, 1);
        encodeSection(pal, idxMatrix, stats.rle, stats.huff, stats.maxRun, stats.bitsPerIndex);
        encodeGradientSection(grad);
    };

    // ── 1. Header ─────────────────────────────────────────────────────────────
    uint32_t w = width, h = height;
    if (subsample){
        uint32_t w = width, h = height;
        uint32_t sw = subW, sh = subH;
        file.write((char*)&w,  4);
        file.write((char*)&h,  4);
        file.write((char*)&sw, 4);  // ← store subsampled dims so decoder knows
        file.write((char*)&sh, 4);
        uint8_t subsampleFlag = subsample ? 1 : 0;
        file.write((char*)&subsampleFlag, 1);
    }else{
        file.write((char*)&w, 4);
        file.write((char*)&h, 4);
    }
    

    // ── 2. Three channel layers ───────────────────────────────────────────────
    // Magic bytes: 0xC1=L channel, 0xC2=A channel, 0xC3=B channel
    ChannelStats statsL, statsA, statsB;
    if(subsample){
        encodeChannelLayer(0xC1, paletteL, idxL, gradientsL, statsL);
        encodeChannelLayer(0xC2, paletteA, idxA, gradientsA, statsA);
        encodeChannelLayer(0xC3, paletteB, idxB, gradientsB, statsB);
    }else{
        encodeChannelLayer(0xC1, paletteL, indexMatrixL, gradientsL, statsL);
        encodeChannelLayer(0xC2, paletteA, indexMatrixA, gradientsA, statsA);
        encodeChannelLayer(0xC3, paletteB, indexMatrixB, gradientsB, statsB);
    }

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

    // Palette bytes: 1D so only 2 bytes per entry (float16 scalar)
    auto palBytes1D = [](const std::vector<PaletteEntry>& pal) -> size_t {
        return 2 + pal.size() * 2;  // palSize(2) + 1×float16(2) per entry
    };

    auto huffTblBytes = [](const std::map<int,std::pair<uint32_t,int>>& huff) -> size_t {
        return 2 + huff.size() * (4+1+4);
    };

    auto channelLayerBytes = [&](const ChannelStats& s,
                                  const std::vector<PaletteEntry>& pal,
                                  const GradientData& grad) -> size_t {
        return 1                           // magic
             + palBytes1D(pal)             // palette (1D)
             + 1 + 4                       // bitsPerIndex + maxRun
             + huffTblBytes(s.huff)        // huffman table
             + 4 + 4                       // rleCount + byteCount
             + (rleBits(s.rle, s.huff, s.maxRun) + 7) / 8
             + gradBytes(grad);
    };

    size_t lTotal = channelLayerBytes(statsL, paletteL, gradientsL);
    size_t aTotal = channelLayerBytes(statsA, paletteA, gradientsA);
    size_t bTotal = channelLayerBytes(statsB, paletteB, gradientsB);

    size_t lBits = rleBits(statsL.rle, statsL.huff, statsL.maxRun);
    size_t aBits = rleBits(statsA.rle, statsA.huff, statsA.maxRun);
    size_t bBits = rleBits(statsB.rle, statsB.huff, statsB.maxRun);

    float bpp = (float)(fileSize * 8) / totalPixels;

    auto pct      = [&](size_t b) { return (float)b * 100.0f / (float)fileSize; };
    auto bitsPerPx = [&](size_t b) { return (float)(b * 8) / (float)totalPixels; };

    auto row = [&](const std::string& name, size_t bytes) {
        std::cout << "| " << std::left  << std::setw(19) << name
                  << "| "  << std::right << std::setw(7)  << bytes
                  << " | "              << std::setw(5)   << std::fixed
                                        << std::setprecision(3) << bitsPerPx(bytes)
                  << "  | "             << std::setw(7)   << std::setprecision(2)
                                        << pct(bytes) << "%  |\n";
    };

    std::cout << "\n|------------------------------------------------------|\n";
    std::cout << "|          SIF Per-Channel File Breakdown             |\n";
    std::cout << "|------------------------------------------------------|\n";
    std::cout << "| Section            | Bytes   |  bpp   |   % file  |\n";
    std::cout << "|------------------------------------------------------|\n";
    row("Header",             4 + 4);
    row("L channel",          lTotal);
    row("  - palette",        palBytes1D(paletteL));
    row("  - RLE stream",     (lBits + 7) / 8);
    row("  - gradients",      gradBytes(gradientsL));
    row("A channel",          aTotal);
    row("  - palette",        palBytes1D(paletteA));
    row("  - RLE stream",     (aBits + 7) / 8);
    row("  - gradients",      gradBytes(gradientsA));
    row("B channel",          bTotal);
    row("  - palette",        palBytes1D(paletteB));
    row("  - RLE stream",     (bBits + 7) / 8);
    row("  - gradients",      gradBytes(gradientsB));
    std::cout << "|------------------------------------------------------|\n";
    row("TOTAL",              fileSize);
    std::cout << "|------------------------------------------------------|\n";

    int precBits = (int)gradientsL.precision;
    std::cout << "\n--- Per-element bit costs ---\n";
    std::cout << "Palette entry (1D): " << 16         << " bits (float16 scalar)\n";
    std::cout << "Huffman table entry:" << (4+1+4)*8  << " bits\n";
    std::cout << "Gradient descriptor:" << precBits   << " bits\n";
    std::cout << "Change point:       " << (2+2+4)*8  << " bits\n";
    std::cout << "Avg L RLE symbol:   " << std::fixed << std::setprecision(2)
              << (lBits / (float)statsL.rle.size()) << " bits\n";
    std::cout << "Avg A RLE symbol:   "
              << (aBits / (float)statsA.rle.size()) << " bits\n";
    std::cout << "Avg B RLE symbol:   "
              << (bBits / (float)statsB.rle.size()) << " bits\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Image:           " << width << "x" << height
              << " (" << totalPixels << " pixels)\n";
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

    float epsilonL = std::stof(argv[1]);
    float epsilonA = std::stof(argv[2]);
    float epsilonB = std::stof(argv[3]);
    int   maxStepsL = std::stoi(argv[4]);
    int   maxStepsA = std::stoi(argv[5]);
    int   maxStepsB = std::stoi(argv[6]);

    std::cout << "Per-channel Encoder: epsilonL=" << epsilonL
              << " epsilonA=" << epsilonA << " epsilonB=" << epsilonB << "\n";

    //const char* filename = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\encoder\\data\\images\\Lenna.png";
    const char* filename = "C:\\Users\\rical\\OneDrive\\Desktop\\Wallpaper\\Napoli.png";

    int width, height, channels;
    unsigned char* imgData = stbi_load(filename, &width, &height, &channels, 0);
    if (!imgData) { std::cerr << "Failed to load image.\n"; return 1; }

    int totalPixels = width * height;

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

    // ── Helper: quantize a single channel ────────────────────────────────────
    // Returns palette (1D, just float values) and index matrix
    struct ChannelPaletteEntry { float value; float error; };

    auto quantizeChannel = [&](
        const std::vector<float>& channel,
        float epsilon,
        int maxSteps,
        std::vector<ChannelPaletteEntry>& palette,
        std::vector<int>& indexMatrix)
    {
        // Build 1D voxel map — use a std::map<int, ...> directly
        // Each voxel key is floor(value / epsilon)
        struct VoxelAccum { float sum = 0; int count = 0; float minV = 1e9, maxV = -1e9; };
        std::map<int, VoxelAccum> voxels;

        for (int i = 0; i < totalPixels; i++) {
            int key = static_cast<int>(std::floor(channel[i] / epsilon));
            auto& v = voxels[key];
            v.sum += channel[i];
            v.count++;
            v.minV = std::min(v.minV, channel[i]);
            v.maxV = std::max(v.maxV, channel[i]);
        }

        // Build palette from voxels
        // With maxSteps=0 each voxel is its own cluster
        // For now just use voxel centroids directly (equivalent to maxSteps=0)
        std::map<int, int> voxelToIdx;
        int idx = 0;
        for (auto& [key, accum] : voxels) {
            float centroid = accum.sum / accum.count;
            float error    = std::max(centroid - accum.minV, accum.maxV - centroid);
            palette.push_back({ centroid, error });
            voxelToIdx[key] = idx++;
        }

        // Build index matrix
        indexMatrix.resize(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            int key = static_cast<int>(std::floor(channel[i] / epsilon));
            indexMatrix[i] = voxelToIdx[key];
        }

        std::cout << "  Voxels/clusters: " << palette.size() << "\n";
    };

    // ── Quantize each channel ─────────────────────────────────────────────────
    std::vector<ChannelPaletteEntry> paletteL, paletteA, paletteB;
    std::vector<int> indexMatrixL, indexMatrixA, indexMatrixB;

    std::cout << "Quantizing L channel:\n";
    quantizeChannel(channelL, epsilonL, maxStepsL, paletteL, indexMatrixL);
    std::cout << "Quantizing A channel:\n";
    quantizeChannel(channelA, epsilonA, maxStepsA, paletteA, indexMatrixA);
    std::cout << "Quantizing B channel:\n";
    quantizeChannel(channelB, epsilonB, maxStepsB, paletteB, indexMatrixB);

    // ── Build flat Lab arrays for gradient encoding ───────────────────────────
    // For each channel, we treat it as a 1D "image" for gradient encoding
    // We need LabPixelFlat but only one channel matters per gradient
    // Use L=channel value, a=0, b=0 for simplicity
    auto channelToLabFlat = [&](const std::vector<float>& ch) {
        std::vector<LabPixelFlat> flat(totalPixels);
        for (int i = 0; i < totalPixels; i++)
            flat[i] = { ch[i], 0.0f, 0.0f };
        return flat;
    };

    // Build PaletteEntry vectors for applyGradients compatibility
    auto channelPalToPaletteEntry = [](const std::vector<ChannelPaletteEntry>& cp) {
        std::vector<PaletteEntry> pe(cp.size());
        for (size_t i = 0; i < cp.size(); i++)
            pe[i] = { cp[i].value, 0.0f, 0.0f, cp[i].error };
        return pe;
    };

    std::vector<PaletteEntry> palEntL = channelPalToPaletteEntry(paletteL);
    std::vector<PaletteEntry> palEntA = channelPalToPaletteEntry(paletteA);
    std::vector<PaletteEntry> palEntB = channelPalToPaletteEntry(paletteB);

    // ── Encode gradients for each channel ─────────────────────────────────────
    std::cout << "\nEncoding gradients:\n";

    auto flatL = channelToLabFlat(channelL);
    auto flatA = channelToLabFlat(channelA);
    auto flatB = channelToLabFlat(channelB);

    GradientData gradientsL = encodeGradients(indexMatrixL, flatL, width, height,
                                               GradientPrecision::BITS_2, 1.0f, 32);
    for (auto& d : gradientsL.queue) d.shape = 1;
    gradientsL.valid = true;

    GradientData gradientsA = encodeGradients(indexMatrixA, flatA, width, height,
                                               GradientPrecision::BITS_2, 1.0f, 32);
    for (auto& d : gradientsA.queue) d.shape = 1;
    gradientsA.valid = true;

    GradientData gradientsB = encodeGradients(indexMatrixB, flatB, width, height,
                                               GradientPrecision::BITS_2, 1.0f, 32);
    for (auto& d : gradientsB.queue) d.shape = 1;
    gradientsB.valid = true;

    // ── Reconstruct and apply gradients ───────────────────────────────────────
    // For gradient application, build LabF images per channel
    std::vector<LabF> reconL(totalPixels), reconA(totalPixels), reconB(totalPixels);

    for (int i = 0; i < totalPixels; i++) {
        reconL[i] = { palEntL[indexMatrixL[i]].L, 0.0f, 0.0f };
        reconA[i] = { palEntA[indexMatrixA[i]].L, 0.0f, 0.0f };
        reconB[i] = { palEntB[indexMatrixB[i]].L, 0.0f, 0.0f };
    }

    applyGradients(reconL, indexMatrixL, palEntL, gradientsL, width, height);
    applyGradients(reconA, indexMatrixA, palEntA, gradientsA, width, height);
    applyGradients(reconB, indexMatrixB, palEntB, gradientsB, width, height);

    // ── Combine channels and save ─────────────────────────────────────────────
    std::vector<LabF> fullRecon(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
        fullRecon[i] = { reconL[i].L, reconA[i].L, reconB[i].L };
    }

    // ── Visualization ─────────────────────────────────────────────────────────
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


    saveSIF_perChannel("output_per_channel.sif",
                   width, height,
                   palEntL, indexMatrixL, gradientsL,
                   palEntA, indexMatrixA, gradientsA,
                   palEntB, indexMatrixB, gradientsB,
                   true);



    // Original for comparison
    std::vector<LabF> original(totalPixels);
    for (int i = 0; i < totalPixels; i++)
        original[i] = { channelL[i], channelA[i], channelB[i] };

    saveLabImage(original,   "out_original.png");
    saveLabImage(fullRecon,  "out_per_channel_reconstruction.png");

    // ── Stats ─────────────────────────────────────────────────────────────────
    std::cout << "\n--- Per-Channel Compression Summary ---\n";
    std::cout << "L palette size: " << paletteL.size() << " entries\n";
    std::cout << "A palette size: " << paletteA.size() << " entries\n";
    std::cout << "B palette size: " << paletteB.size() << " entries\n";
    std::cout << "Total unique values: " << paletteL.size() + paletteA.size() + paletteB.size() << "\n";
    std::cout << "Original pixels: " << totalPixels << "\n";

    stbi_image_free(imgData);
    return 0;
}