#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

#include <string>
#include <vector>
#include "VoxelGrid_mod.hpp" // For PaletteEntry/LabPixel types
#include "GradientEncoder_mod.hpp"
#include "Compression.hpp"
#include <PaletteEntry.hpp>
#include "GradientTypes.hpp"


/*

#include <PaletteEntry.hpp>

void saveSIF_claude_reduce(const std::string& path,
                    int width, int height,           // subsampled
                    int origWidth, int origHeight,   // full resolution
                    const std::vector<PaletteEntry>& palette,
                    const std::vector<int>& indexMatrix,
                    const GradientData& gradients,
                    const ResidualData& residual)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    // ── 1. Header & Palette ──────────────────────────────────────────────────
    uint32_t w     = width,          h     = height;
    uint32_t origW = origWidth,  origH = origHeight;  // full res before subsampling
    uint16_t palSize = (uint16_t)palette.size();

    file.write((char*)&w,       4);
    file.write((char*)&h,       4);
    file.write((char*)&origW,   4);   // ← must come before palSize
    file.write((char*)&origH,   4);   // ← must come before palSize
    file.write((char*)&palSize, 2);

    for (const auto& p : palette) {
        int8_t L = (int8_t)std::round(p.L);
        int8_t a = (int8_t)std::round(p.a);
        int8_t b = (int8_t)std::round(p.b);
        int8_t e = (int8_t)std::floor(p.error);
        file.write((char*)&L, 1); file.write((char*)&a, 1);
        file.write((char*)&b, 1); file.write((char*)&e, 1);
    }

    // ── 2. Bit-width reduction ───────────────────────────────────────────────
    int bitsPerIndex = 1;
    while ((1 << bitsPerIndex) < palSize) bitsPerIndex++;

    // ── 3. RLE on index stream ───────────────────────────────────────────────
    struct RLESymbol { int value; int runLength; };
    std::vector<RLESymbol> rleStream;

    if (!indexMatrix.empty()) {
        int cur = indexMatrix[0], run = 1;
        for (size_t i = 1; i < indexMatrix.size(); i++) {
            if (indexMatrix[i] == cur) { run++; }
            else { rleStream.push_back({cur, run}); cur = indexMatrix[i]; run = 1; }
        }
        rleStream.push_back({cur, run});
    }

    // ── 4. Huffman over RLE pairs ────────────────────────────────────────────
    int maxRun = 1;
    for (auto& s : rleStream) maxRun = std::max(maxRun, s.runLength);

    auto pairKey = [&](int value, int run) {
        return value * (maxRun + 1) + (run - 1);
    };

    std::map<int, int> freq;
    for (auto& s : rleStream) freq[pairKey(s.value, s.runLength)]++;

    std::map<int, std::pair<uint32_t, int>> huffmanTable;
    if (freq.size() == 1) {
        huffmanTable[freq.begin()->first] = {0, 1};
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
        buildCodes(pq.top(), "", huffmanTable);
    }

    // ── 5. Metadata ──────────────────────────────────────────────────────────
    file.write((char*)&bitsPerIndex, 1);
    file.write((char*)&maxRun,       4);

    // ── 6. Huffman table ─────────────────────────────────────────────────────
    uint16_t tableEntries = (uint16_t)huffmanTable.size();
    file.write((char*)&tableEntries, 2);
    for (auto const& [key, code] : huffmanTable) {
        uint8_t len = (uint8_t)code.second;
        file.write((char*)&key,        4);
        file.write((char*)&len,        1);
        file.write((char*)&code.first, 4);
    }

    // ── 7. RLE + Huffman bit stream ──────────────────────────────────────────
    uint32_t rleCount = (uint32_t)rleStream.size();
    file.write((char*)&rleCount, 4);

    BitWriter bw(file);
    for (auto& s : rleStream) {
        int key = pairKey(s.value, s.runLength);
        auto& [code, len] = huffmanTable[key];
        bw.write(code, len);
    }
    bw.flush();

    // ── 8. Gradient section ──────────────────────────────────────────────────
    uint8_t precisionByte = (uint8_t)gradients.precision;
    file.write((char*)&precisionByte, 1);

    uint32_t queueSize = (uint32_t)gradients.queue.size();
    file.write((char*)&queueSize, 4);

    BitWriter bwGrad(file);
    int precBits = (int)gradients.precision;
    for (uint8_t desc : gradients.queue)
        bwGrad.write(desc, precBits);
    bwGrad.flush();

    uint32_t cpCount = (uint32_t)gradients.changePoints.size();
    file.write((char*)&cpCount, 4);
    for (const auto& cp : gradients.changePoints) {
        file.write((char*)&cp.x,        2);
        file.write((char*)&cp.y,        2);
        file.write((char*)&cp.queueIdx, 4);
    }

    // ── 9. Residual DCT section ───────────────────────────────────────────────
    // Magic byte so decoder knows residual data is present
    uint8_t residualMagic = 0xDC;  // 'R'esidual 'C'oefficients marker
    file.write((char*)&residualMagic, 1);
    writeResidual(file, residual);

    file.close();


    // ── Detailed Statistics ───────────────────────────────────────────────────
    size_t fileSize = std::filesystem::file_size(path);

    // ── Section sizes in bytes ────────────────────────────────────────────────
    size_t headerBytes    = 4 + 4 + 2;                          // w, h, palSize
    size_t paletteBytes   = palette.size() * 4;                 // L,a,b,e × 4 entries
    size_t metaBytes      = 1 + 4;                              // bitsPerIndex + maxRun
    size_t huffTableBytes = 2 + huffmanTable.size() * (4+1+4);  // count + entries
    size_t rleBytes       = 4;                                  // rleCount uint32
    
    // Huffman bit stream: we know the exact bit count
    size_t rleStreamBits  = 0;
    for (auto& s : rleStream) {
        int key = pairKey(s.value, s.runLength);
        rleStreamBits += huffmanTable.at(key).second;  // code length in bits
    }
    size_t rleStreamBytes = (rleStreamBits + 7) / 8;  // round up to bytes

    // Gradient section
    size_t gradPrecBytes  = 1;                                  // precision byte
    size_t gradQueueBytes = 4 + ((gradients.queue.size() * precBits + 7) / 8);
    size_t gradCPBytes    = 4 + gradients.changePoints.size() * (2+2+4);
    size_t gradTotalBytes = gradPrecBytes + gradQueueBytes + gradCPBytes;

    // Residual section
    size_t residualMagicBytes  = 1;
    size_t residualConfigBytes = 1 + 1 + 4;                    // blockSize, keepCoeffs, quantStep
    size_t residualCountBytes  = 4;                            // coefficient count
    size_t residualCoeffBytes  = residual.coefficients.size(); // 1 byte each (int8)
    size_t residualTotalBytes  = residualMagicBytes + residualConfigBytes
                               + residualCountBytes + residualCoeffBytes;

    float bpp = (float)(fileSize * 8) / (width * height);
    int   totalPixels = width * height;

    auto pct = [&](size_t bytes) {
        return (float)bytes * 100.0f / (float)fileSize;
    };
    auto bitsPerPx = [&](size_t bytes) {
        return (float)(bytes * 8) / (float)totalPixels;
    };

    std::cout << "\n|------------------------------------------------------|\n";
    std::cout << "|              SIF File Breakdown                     |\n";
    std::cout << "|------------------------------------------------------|\n";

    std::cout << "| Section            | Bytes   |  bpp   |   % file  |\n";
    std::cout << "|------------------------------------------------------|\n";

    auto row = [&](const std::string& name, size_t bytes) {
        std::cout << "| " << std::left  << std::setw(19) << name
                  << "| "  << std::right << std::setw(7)  << bytes
                  << " | "              << std::setw(5)   << std::fixed
                                        << std::setprecision(3) << bitsPerPx(bytes)
                  << "  | "             << std::setw(7)   << std::setprecision(2)
                                        << pct(bytes) << "%  |\n";
    };

    row("Header",           headerBytes);
    row("Palette",          paletteBytes);
    row("  - per color",    4);           // just informational
    row("Huffman metadata", metaBytes);
    row("Huffman table",    huffTableBytes);
    row("RLE header",       rleBytes);
    row("RLE bitstream",    rleStreamBytes);
    row("Gradient section", gradTotalBytes);
    row("  - precision",    gradPrecBytes);
    row("  - queue",        gradQueueBytes);
    row("  - change pts",   gradCPBytes);
    row("Residual section", residualTotalBytes);
    row("  - magic byte",   residualMagicBytes);
    row("  - config",       residualConfigBytes);
    row("  - coeff count",  residualCountBytes);
    row("  - coefficients", residualCoeffBytes);

    std::cout << "|------------------------------------------------------|\n";
    row("TOTAL",            fileSize);
    std::cout << "|------------------------------------------------------|\n";

    std::cout << "\n--- Per-element bit costs ---\n";
    std::cout << "Palette entry:       " << 4*8             << " bits (L:8 a:8 b:8 e:8)\n";
    std::cout << "Huffman table entry: " << (4+1+4)*8       << " bits (key:32 len:8 code:32)\n";
    std::cout << "Gradient descriptor: " << precBits        << " bits\n";
    std::cout << "Change point:        " << (2+2+4)*8       << " bits (x:16 y:16 idx:32)\n";
    std::cout << "Residual coeff:      " << 8               << " bits (int8)\n";
    std::cout << "Avg RLE symbol:      " << std::fixed << std::setprecision(2)
              << (rleStreamBits / (float)rleStream.size())  << " bits (Huffman coded)\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Image:           " << width << "x" << height
              << " (" << totalPixels << " pixels)\n";
    std::cout << "Original RGB:    " << (totalPixels * 3) / 1024  << " KB\n";
    std::cout << "SIF File:        " << fileSize / 1024           << " KB\n";
    std::cout << "Bits Per Pixel:  " << bpp                       << " bpp\n";
    std::cout << "Compression:     " << (float)(totalPixels*24) / (fileSize*8) << ":1\n";
    std::cout << "Location: "        << std::filesystem::absolute(path)        << "\n";

}






void saveSIF_claude(const std::string& path,
                    int width, int height,          
                    const std::vector<PaletteEntry>& palette,
                    const std::vector<int>& indexMatrix,
                    const GradientData& gradients,
                    const ResidualData& residual)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    // ── 1. Header & Palette ──────────────────────────────────────────────────
    uint32_t w     = width,          h     = height;
    uint16_t palSize = (uint16_t)palette.size();

    file.write((char*)&w,       4);
    file.write((char*)&h,       4);
    file.write((char*)&palSize, 2);

    for (const auto& p : palette) {
        int8_t L = (int8_t)std::round(p.L);
        int8_t a = (int8_t)std::round(p.a);
        int8_t b = (int8_t)std::round(p.b);
        int8_t e = (int8_t)std::floor(p.error);
        file.write((char*)&L, 1); file.write((char*)&a, 1);
        file.write((char*)&b, 1); file.write((char*)&e, 1);
    }

    // ── 2. Bit-width reduction ───────────────────────────────────────────────
    int bitsPerIndex = 1;
    while ((1 << bitsPerIndex) < palSize) bitsPerIndex++;

    // ── 3. RLE on index stream ───────────────────────────────────────────────
    struct RLESymbol { int value; int runLength; };
    std::vector<RLESymbol> rleStream;

    if (!indexMatrix.empty()) {
        int cur = indexMatrix[0], run = 1;
        for (size_t i = 1; i < indexMatrix.size(); i++) {
            if (indexMatrix[i] == cur) { run++; }
            else { rleStream.push_back({cur, run}); cur = indexMatrix[i]; run = 1; }
        }
        rleStream.push_back({cur, run});
    }

    // ── 4. Huffman over RLE pairs ────────────────────────────────────────────
    int maxRun = 1;
    for (auto& s : rleStream) maxRun = std::max(maxRun, s.runLength);

    auto pairKey = [&](int value, int run) {
        return value * (maxRun + 1) + (run - 1);
    };

    std::map<int, int> freq;
    for (auto& s : rleStream) freq[pairKey(s.value, s.runLength)]++;

    std::map<int, std::pair<uint32_t, int>> huffmanTable;
    if (freq.size() == 1) {
        huffmanTable[freq.begin()->first] = {0, 1};
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
        buildCodes(pq.top(), "", huffmanTable);
    }

    // ── 5. Metadata ──────────────────────────────────────────────────────────
    file.write((char*)&bitsPerIndex, 1);
    file.write((char*)&maxRun,       4);

    // ── 6. Huffman table ─────────────────────────────────────────────────────
    uint16_t tableEntries = (uint16_t)huffmanTable.size();
    file.write((char*)&tableEntries, 2);
    for (auto const& [key, code] : huffmanTable) {
        uint8_t len = (uint8_t)code.second;
        file.write((char*)&key,        4);
        file.write((char*)&len,        1);
        file.write((char*)&code.first, 4);
    }

    // ── 7. RLE + Huffman bit stream ──────────────────────────────────────────
    uint32_t rleCount = (uint32_t)rleStream.size();
    file.write((char*)&rleCount, 4);

    BitWriter bw(file);
    for (auto& s : rleStream) {
        int key = pairKey(s.value, s.runLength);
        auto& [code, len] = huffmanTable[key];
        bw.write(code, len);
    }
    bw.flush();

    // ── 8. Gradient section ──────────────────────────────────────────────────
    uint8_t precisionByte = (uint8_t)gradients.precision;
    file.write((char*)&precisionByte, 1);

    uint32_t queueSize = (uint32_t)gradients.queue.size();
    file.write((char*)&queueSize, 4);

    BitWriter bwGrad(file);
    int precBits = (int)gradients.precision;
    for (uint8_t desc : gradients.queue)
        bwGrad.write(desc, precBits);
    bwGrad.flush();

    uint32_t cpCount = (uint32_t)gradients.changePoints.size();
    file.write((char*)&cpCount, 4);
    for (const auto& cp : gradients.changePoints) {
        file.write((char*)&cp.x,        2);
        file.write((char*)&cp.y,        2);
        file.write((char*)&cp.queueIdx, 4);
    }

    // ── 9. Residual DCT section ───────────────────────────────────────────────
    // Magic byte so decoder knows residual data is present
    uint8_t residualMagic = 0xDC;  // 'R'esidual 'C'oefficients marker
    file.write((char*)&residualMagic, 1);
    writeResidual(file, residual);

    file.close();


    // ── Detailed Statistics ───────────────────────────────────────────────────
    size_t fileSize = std::filesystem::file_size(path);

    // ── Section sizes in bytes ────────────────────────────────────────────────
    size_t headerBytes    = 4 + 4 + 2;                          // w, h, palSize
    size_t paletteBytes   = palette.size() * 4;                 // L,a,b,e × 4 entries
    size_t metaBytes      = 1 + 4;                              // bitsPerIndex + maxRun
    size_t huffTableBytes = 2 + huffmanTable.size() * (4+1+4);  // count + entries
    size_t rleBytes       = 4;                                  // rleCount uint32
    
    // Huffman bit stream: we know the exact bit count
    size_t rleStreamBits  = 0;
    for (auto& s : rleStream) {
        int key = pairKey(s.value, s.runLength);
        rleStreamBits += huffmanTable.at(key).second;  // code length in bits
    }
    size_t rleStreamBytes = (rleStreamBits + 7) / 8;  // round up to bytes

    // Gradient section
    size_t gradPrecBytes  = 1;                                  // precision byte
    size_t gradQueueBytes = 4 + ((gradients.queue.size() * precBits + 7) / 8);
    size_t gradCPBytes    = 4 + gradients.changePoints.size() * (2+2+4);
    size_t gradTotalBytes = gradPrecBytes + gradQueueBytes + gradCPBytes;

    // Residual section
    size_t residualMagicBytes  = 1;
    size_t residualConfigBytes = 1 + 1 + 4;                    // blockSize, keepCoeffs, quantStep
    size_t residualCountBytes  = 4;                            // coefficient count
    size_t residualCoeffBytes  = residual.coefficients.size(); // 1 byte each (int8)
    size_t residualTotalBytes  = residualMagicBytes + residualConfigBytes
                               + residualCountBytes + residualCoeffBytes;

    float bpp = (float)(fileSize * 8) / (width * height);
    int   totalPixels = width * height;

    auto pct = [&](size_t bytes) {
        return (float)bytes * 100.0f / (float)fileSize;
    };
    auto bitsPerPx = [&](size_t bytes) {
        return (float)(bytes * 8) / (float)totalPixels;
    };

    std::cout << "\n|------------------------------------------------------|\n";
    std::cout << "|              SIF File Breakdown                     |\n";
    std::cout << "|------------------------------------------------------|\n";

    std::cout << "| Section            | Bytes   |  bpp   |   % file  |\n";
    std::cout << "|------------------------------------------------------|\n";

    auto row = [&](const std::string& name, size_t bytes) {
        std::cout << "| " << std::left  << std::setw(19) << name
                  << "| "  << std::right << std::setw(7)  << bytes
                  << " | "              << std::setw(5)   << std::fixed
                                        << std::setprecision(3) << bitsPerPx(bytes)
                  << "  | "             << std::setw(7)   << std::setprecision(2)
                                        << pct(bytes) << "%  |\n";
    };

    row("Header",           headerBytes);
    row("Palette",          paletteBytes);
    row("  - per color",    4);           // just informational
    row("Huffman metadata", metaBytes);
    row("Huffman table",    huffTableBytes);
    row("RLE header",       rleBytes);
    row("RLE bitstream",    rleStreamBytes);
    row("Gradient section", gradTotalBytes);
    row("  - precision",    gradPrecBytes);
    row("  - queue",        gradQueueBytes);
    row("  - change pts",   gradCPBytes);
    row("Residual section", residualTotalBytes);
    row("  - magic byte",   residualMagicBytes);
    row("  - config",       residualConfigBytes);
    row("  - coeff count",  residualCountBytes);
    row("  - coefficients", residualCoeffBytes);

    std::cout << "|------------------------------------------------------|\n";
    row("TOTAL",            fileSize);
    std::cout << "|------------------------------------------------------|\n";

    std::cout << "\n--- Per-element bit costs ---\n";
    std::cout << "Palette entry:       " << 4*8             << " bits (L:8 a:8 b:8 e:8)\n";
    std::cout << "Huffman table entry: " << (4+1+4)*8       << " bits (key:32 len:8 code:32)\n";
    std::cout << "Gradient descriptor: " << precBits        << " bits\n";
    std::cout << "Change point:        " << (2+2+4)*8       << " bits (x:16 y:16 idx:32)\n";
    std::cout << "Residual coeff:      " << 8               << " bits (int8)\n";
    std::cout << "Avg RLE symbol:      " << std::fixed << std::setprecision(2)
              << (rleStreamBits / (float)rleStream.size())  << " bits (Huffman coded)\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Image:           " << width << "x" << height
              << " (" << totalPixels << " pixels)\n";
    std::cout << "Original RGB:    " << (totalPixels * 3) / 1024  << " KB\n";
    std::cout << "SIF File:        " << fileSize / 1024           << " KB\n";
    std::cout << "Bits Per Pixel:  " << bpp                       << " bpp\n";
    std::cout << "Compression:     " << (float)(totalPixels*24) / (fileSize*8) << ":1\n";
    std::cout << "Location: "        << std::filesystem::absolute(path)        << "\n";

}


*/


// ── float16 encode/decode helpers ─────────────────────────────────────────
inline uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign     = (x >> 16) & 0x8000;
    int32_t  exponent = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exponent <= 0)  return sign;               // underflow → zero
    if (exponent >= 31) return sign | 0x7C00;      // overflow → inf
    return sign | (exponent << 10) | (mantissa >> 13);
}

inline float halfToFloat(uint16_t h) {
    uint32_t sign     = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF);

    if (exponent == 0)  return 0.0f;               // zero/denormal → zero
    if (exponent == 31) return std::numeric_limits<float>::infinity();

    exponent = exponent - 15 + 127;
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, 4);
    return f;
}



void saveSIF_v2(const std::string& path,
                int width, int height,
                const std::vector<PaletteEntry>& palette,
                const std::vector<int>& indexMatrix,
                const GradientData& gradients,
                const std::vector<PaletteEntry>& residualPalette,
                const std::vector<int>& residualIndexMatrix,
                const GradientData& residualGradients,
                const std::vector<PaletteEntry>& residualPalette2,
                const std::vector<int>& residualIndexMatrix2,
                const GradientData& residualGradients2,
                const std::vector<PaletteEntry>& residualPalette3,
                const std::vector<int>& residualIndexMatrix3,
                const GradientData& residualGradients3)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    struct RLESymbol { int value; int runLength; };

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
            uint16_t L = floatToHalf(p.L);
            uint16_t a = floatToHalf(p.a);
            uint16_t b = floatToHalf(p.b);
            file.write((char*)&L, 2);
            file.write((char*)&a, 2);
            file.write((char*)&b, 2);
        }

        bitsPerIndex = 1;
        while ((1 << bitsPerIndex) < palSize) bitsPerIndex++;

        if (!idxMatrix.empty()) {
            int cur = idxMatrix[0], run = 1;
            for (size_t i = 1; i < idxMatrix.size(); i++) {
                if (idxMatrix[i] == cur) { run++; }
                else { rleStream.push_back({cur, run}); cur = idxMatrix[i]; run = 1; }
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

    // ── Helper: encode a full residual layer ──────────────────────────────────
    struct ResidualStats {
        std::vector<RLESymbol> rle;
        std::map<int, std::pair<uint32_t,int>> huff;
        int maxRun = 1, bitsPerIndex = 1;
    };

    auto encodeResidualLayer = [&](
        uint8_t magic,
        const std::vector<PaletteEntry>& pal,
        const std::vector<int>& idxMatrix,
        const GradientData& grad,
        ResidualStats& stats)
    {
        file.write((char*)&magic, 1);
        encodeSection(pal, idxMatrix, stats.rle, stats.huff, stats.maxRun, stats.bitsPerIndex);
        encodeGradientSection(grad);
    };

    // ── 1. Header ─────────────────────────────────────────────────────────────
    uint32_t w = width, h = height;
    file.write((char*)&w, 4);
    file.write((char*)&h, 4);

    // ── 2. Main palette + index matrix + gradients ────────────────────────────
    std::vector<RLESymbol> mainRLE;
    std::map<int, std::pair<uint32_t,int>> mainHuff;
    int mainMaxRun = 1, mainBitsPerIndex = 1;
    encodeSection(palette, indexMatrix, mainRLE, mainHuff, mainMaxRun, mainBitsPerIndex);
    encodeGradientSection(gradients);

    // ── 3. Residual layers ────────────────────────────────────────────────────
    ResidualStats res1Stats, res2Stats, res3Stats;
    encodeResidualLayer(0xFE, residualPalette,  residualIndexMatrix,  residualGradients,  res1Stats);
    encodeResidualLayer(0xFD, residualPalette2, residualIndexMatrix2, residualGradients2, res2Stats);
    encodeResidualLayer(0xFC, residualPalette3, residualIndexMatrix3, residualGradients3, res3Stats);

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

    size_t mainBits = rleBits(mainRLE,      mainHuff,      mainMaxRun);
    size_t r1Bits   = rleBits(res1Stats.rle, res1Stats.huff, res1Stats.maxRun);
    size_t r2Bits   = rleBits(res2Stats.rle, res2Stats.huff, res2Stats.maxRun);
    size_t r3Bits   = rleBits(res3Stats.rle, res3Stats.huff, res3Stats.maxRun);

    auto palBytes = [](const std::vector<PaletteEntry>& pal) -> size_t {
        return 2 + pal.size() * 6;  // palSize(2) + 3×float16(2) per entry
    };

    auto huffTblBytes = [](const std::map<int,std::pair<uint32_t,int>>& huff) -> size_t {
        return 2 + huff.size() * (4+1+4);
    };

    auto resLayerBytes = [&](const ResidualStats& s,
                              const std::vector<PaletteEntry>& pal,
                              const GradientData& grad) -> size_t {
        return 1                          // magic
             + palBytes(pal)              // palette
             + 1 + 4                      // bitsPerIndex + maxRun
             + huffTblBytes(s.huff)       // huffman table
             + 4 + 4                      // rleCount + byteCount
             + (rleBits(s.rle, s.huff, s.maxRun) + 7) / 8  // bitstream
             + gradBytes(grad);           // gradients
    };

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

    size_t mainSection = 4 + 4                           // header
                       + palBytes(palette)               // main palette
                       + 1 + 4                           // bitsPerIndex + maxRun
                       + huffTblBytes(mainHuff)          // huffman table
                       + 4 + 4                           // rleCount + byteCount
                       + (mainBits + 7) / 8              // bitstream
                       + gradBytes(gradients);           // gradients

    size_t r1Total = resLayerBytes(res1Stats, residualPalette,  residualGradients);
    size_t r2Total = resLayerBytes(res2Stats, residualPalette2, residualGradients2);
    size_t r3Total = resLayerBytes(res3Stats, residualPalette3, residualGradients3);

    std::cout << "\n|------------------------------------------------------|\n";
    std::cout << "|              SIF File Breakdown                     |\n";
    std::cout << "|------------------------------------------------------|\n";
    std::cout << "| Section            | Bytes   |  bpp   |   % file  |\n";
    std::cout << "|------------------------------------------------------|\n";
    row("Main section",      mainSection);
    row("  - palette",       palBytes(palette));
    row("  - RLE stream",    (mainBits + 7) / 8);
    row("  - gradients",     gradBytes(gradients));
    row("Residual 1",        r1Total);
    row("  - palette",       palBytes(residualPalette));
    row("  - RLE stream",    (r1Bits + 7) / 8);
    row("  - gradients",     gradBytes(residualGradients));
    row("Residual 2",        r2Total);
    row("  - palette",       palBytes(residualPalette2));
    row("  - RLE stream",    (r2Bits + 7) / 8);
    row("  - gradients",     gradBytes(residualGradients2));
    row("Residual 3",        r3Total);
    row("  - palette",       palBytes(residualPalette3));
    row("  - RLE stream",    (r3Bits + 7) / 8);
    row("  - gradients",     gradBytes(residualGradients3));
    std::cout << "|------------------------------------------------------|\n";
    row("TOTAL",             fileSize);
    std::cout << "|------------------------------------------------------|\n";

    int precBits = (int)gradients.precision;
    std::cout << "\n--- Per-element bit costs ---\n";
    std::cout << "Palette entry:       " << 3*16      << " bits (L:16 a:16 b:16 float16)\n";
    std::cout << "Huffman table entry: " << (4+1+4)*8 << " bits\n";
    std::cout << "Gradient descriptor: " << precBits  << " bits\n";
    std::cout << "Change point:        " << (2+2+4)*8 << " bits\n";
    std::cout << "Avg main RLE symbol: " << std::fixed << std::setprecision(2)
              << (mainBits / (float)mainRLE.size())        << " bits\n";
    std::cout << "Avg res1 RLE symbol: " << (r1Bits / (float)res1Stats.rle.size()) << " bits\n";
    std::cout << "Avg res2 RLE symbol: " << (r2Bits / (float)res2Stats.rle.size()) << " bits\n";
    std::cout << "Avg res3 RLE symbol: " << (r3Bits / (float)res3Stats.rle.size()) << " bits\n";

    std::cout << "\n--- Summary ---\n";
    std::cout << "Image:           " << width << "x" << height
              << " (" << totalPixels << " pixels)\n";
    std::cout << "Original RGB:    " << (totalPixels * 3) / 1024 << " KB\n";
    std::cout << "SIF File:        " << fileSize / 1024          << " KB\n";
    std::cout << "Bits Per Pixel:  " << bpp                      << " bpp\n";
    std::cout << "Compression:     " << (float)(totalPixels*24) / (fileSize*8) << ":1\n";
    std::cout << "Location: "        << std::filesystem::absolute(path)        << "\n";
}

#endif
