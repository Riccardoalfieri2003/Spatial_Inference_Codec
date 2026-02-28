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
                const GradientData& residualGradients2)
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
            int8_t L = (int8_t)std::round(p.L);
            int8_t a = (int8_t)std::round(p.a);
            int8_t b = (int8_t)std::round(p.b);
            file.write((char*)&L, 1); file.write((char*)&a, 1);
            file.write((char*)&b, 1);
            //int8_t e = (int8_t)std::floor(p.error);
            //file.write((char*)&L, 1); file.write((char*)&a, 1);
            //file.write((char*)&b, 1); file.write((char*)&e, 1);
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

        // ── Pre-compute bit count and write byte count before bitstream ───────
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

        // ── Pre-compute and write byte count before bitstream ─────────────────
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

    // ── 1. Header ─────────────────────────────────────────────────────────────
    uint32_t w = width, h = height;
    file.write((char*)&w, 4);
    file.write((char*)&h, 4);

    // ── 2. Main palette + index matrix ────────────────────────────────────────
    std::vector<RLESymbol> mainRLE;
    std::map<int, std::pair<uint32_t,int>> mainHuff;
    int mainMaxRun = 1, mainBitsPerIndex = 1;
    encodeSection(palette, indexMatrix, mainRLE, mainHuff, mainMaxRun, mainBitsPerIndex);

    // ── 3. Main gradients ─────────────────────────────────────────────────────
    encodeGradientSection(gradients);

    // ── 4. Residual 1 ─────────────────────────────────────────────────────────
    uint8_t magic1 = 0xFE;
    file.write((char*)&magic1, 1);
    std::vector<RLESymbol> resRLE;
    std::map<int, std::pair<uint32_t,int>> resHuff;
    int resMaxRun = 1, resBitsPerIndex = 1;
    encodeSection(residualPalette, residualIndexMatrix, resRLE, resHuff, resMaxRun, resBitsPerIndex);
    encodeGradientSection(residualGradients);

    // ── 5. Residual 2 ─────────────────────────────────────────────────────────
    uint8_t magic2 = 0xFD;
    file.write((char*)&magic2, 1);
    std::vector<RLESymbol> resRLE2;
    std::map<int, std::pair<uint32_t,int>> resHuff2;
    int resMaxRun2 = 1, resBitsPerIndex2 = 1;
    encodeSection(residualPalette2, residualIndexMatrix2, resRLE2, resHuff2, resMaxRun2, resBitsPerIndex2);
    encodeGradientSection(residualGradients2);

    file.close();

    // ── Statistics ────────────────────────────────────────────────────────────
    size_t fileSize = std::filesystem::file_size(path);
    int totalPixels = width * height;

    auto gradSectionBytes = [&](const GradientData& grad) -> size_t {
        int pb = (int)grad.precision;
        return 1 + 4 + ((grad.queue.size() * pb + 7) / 8) +
               4 + grad.changePoints.size() * (2+2+4);
    };

    auto pairKeyFn = [](int value, int run, int maxRun) {
        return value * (maxRun + 1) + (run - 1);
    };

    size_t mainRleBits = 0;
    for (auto& s : mainRLE)
        mainRleBits += mainHuff.at(pairKeyFn(s.value, s.runLength, mainMaxRun)).second;

    size_t resRleBits = 0;
    for (auto& s : resRLE)
        resRleBits += resHuff.at(pairKeyFn(s.value, s.runLength, resMaxRun)).second;

    size_t resRleBits2 = 0;
    for (auto& s : resRLE2)
        resRleBits2 += resHuff2.at(pairKeyFn(s.value, s.runLength, resMaxRun2)).second;

    size_t headerBytes       = 4 + 4;
    size_t mainPalBytes      = 2 + palette.size() * 4;
    size_t mainMetaBytes     = 1 + 4;
    size_t mainHuffTblBytes  = 2 + mainHuff.size() * (4+1+4);
    size_t mainRleBytes      = (mainRleBits + 7) / 8;
    size_t mainGradBytes     = gradSectionBytes(gradients);

    size_t res1MagicBytes    = 1;
    size_t res1PalBytes      = 2 + residualPalette.size() * 4;
    size_t res1MetaBytes     = 1 + 4;
    size_t res1HuffTblBytes  = 2 + resHuff.size() * (4+1+4);
    size_t res1RleBytes      = (resRleBits + 7) / 8;
    size_t res1GradBytes     = gradSectionBytes(residualGradients);
    size_t res1TotalBytes    = res1MagicBytes + res1PalBytes + res1MetaBytes
                             + res1HuffTblBytes + 4 + res1RleBytes + res1GradBytes;

    size_t res2MagicBytes    = 1;
    size_t res2PalBytes      = 2 + residualPalette2.size() * 4;
    size_t res2MetaBytes     = 1 + 4;
    size_t res2HuffTblBytes  = 2 + resHuff2.size() * (4+1+4);
    size_t res2RleBytes      = (resRleBits2 + 7) / 8;
    size_t res2GradBytes     = gradSectionBytes(residualGradients2);
    size_t res2TotalBytes    = res2MagicBytes + res2PalBytes + res2MetaBytes
                             + res2HuffTblBytes + 4 + res2RleBytes + res2GradBytes;

    float bpp = (float)(fileSize * 8) / totalPixels;

    auto pct = [&](size_t bytes) {
        return (float)bytes * 100.0f / (float)fileSize;
    };
    auto bitsPerPx = [&](size_t bytes) {
        return (float)(bytes * 8) / (float)totalPixels;
    };

    auto row = [&](const std::string& name, size_t bytes) {
        std::cout << "| " << std::left  << std::setw(19) << name
                  << "| "  << std::right << std::setw(7)  << bytes
                  << " | "              << std::setw(5)   << std::fixed
                                        << std::setprecision(3) << bitsPerPx(bytes)
                  << "  | "             << std::setw(7)   << std::setprecision(2)
                                        << pct(bytes) << "%  |\n";
    };

    std::cout << "\n|------------------------------------------------------|\n";
    std::cout << "|              SIF File Breakdown                     |\n";
    std::cout << "|------------------------------------------------------|\n";
    std::cout << "| Section            | Bytes   |  bpp   |   % file  |\n";
    std::cout << "|------------------------------------------------------|\n";

    row("Header",             headerBytes);
    row("Main palette",       mainPalBytes);
    row("Main Huff meta",     mainMetaBytes);
    row("Main Huff table",    mainHuffTblBytes);
    row("Main RLE stream",    mainRleBytes);
    row("Main gradients",     mainGradBytes);
    row("Residual 1",         res1TotalBytes);
    row("  - palette",        res1PalBytes);
    row("  - Huff meta",      res1MetaBytes);
    row("  - Huff table",     res1HuffTblBytes);
    row("  - RLE stream",     res1RleBytes);
    row("  - gradients",      res1GradBytes);
    row("Residual 2",         res2TotalBytes);
    row("  - palette",        res2PalBytes);
    row("  - Huff meta",      res2MetaBytes);
    row("  - Huff table",     res2HuffTblBytes);
    row("  - RLE stream",     res2RleBytes);
    row("  - gradients",      res2GradBytes);
    std::cout << "|------------------------------------------------------|\n";
    row("TOTAL",              fileSize);
    std::cout << "|------------------------------------------------------|\n";

    int precBits = (int)gradients.precision;
    std::cout << "\n--- Per-element bit costs ---\n";
    std::cout << "Palette entry:       " << 4*8       << " bits (L:8 a:8 b:8 e:8)\n";
    std::cout << "Huffman table entry: " << (4+1+4)*8 << " bits\n";
    std::cout << "Gradient descriptor: " << precBits  << " bits\n";
    std::cout << "Change point:        " << (2+2+4)*8 << " bits\n";
    std::cout << "Avg main RLE symbol: " << std::fixed << std::setprecision(2)
              << (mainRleBits / (float)mainRLE.size()) << " bits\n";
    std::cout << "Avg res1 RLE symbol: " << std::fixed << std::setprecision(2)
              << (resRleBits  / (float)resRLE.size())  << " bits\n";
    std::cout << "Avg res2 RLE symbol: " << std::fixed << std::setprecision(2)
              << (resRleBits2 / (float)resRLE2.size()) << " bits\n";

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
