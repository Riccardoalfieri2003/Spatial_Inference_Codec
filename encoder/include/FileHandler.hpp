#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

#include <string>
#include <vector>
#include "VoxelGrid.hpp" // For PaletteEntry/LabPixel types
#include "GradientEncoder.hpp"

#include <map>
#include <queue>
#include <bitset>


#include <fstream>   // For std::ofstream and std::ifstream
#include <iostream>  // For std::cout
#include <vector>    // For std::vector
#include <string>    // For std::string
#include <filesystem> // For creating directories (C++17 or later)
#include <iomanip>



#include <fstream>
#include <cstdint>

class BitWriter {
    std::ofstream& out;
    uint8_t buffer;   // Temporary storage for bits
    int bitCount;     // How many bits are currently in the buffer

public:
    BitWriter(std::ofstream& f) : out(f), buffer(0), bitCount(0) {}

    // value: the data to write
    // numBits: how many bits of that data to actually use
    void write(uint32_t value, int numBits) {
        for (int i = numBits - 1; i >= 0; --i) {
            // Extract the i-th bit from the value
            bool bit = (value >> i) & 1;
            
            // Push it into our 8-bit buffer
            buffer = (buffer << 1) | bit;
            bitCount++;

            // If we have a full byte (8 bits), write it to the file
            if (bitCount == 8) {
                out.put(static_cast<char>(buffer));
                buffer = 0;
                bitCount = 0;
            }
        }
    }

    // Very important: writes the last remaining bits
    void flush() {
        if (bitCount > 0) {
            buffer <<= (8 - bitCount); // Move bits to the start of the byte
            out.put(static_cast<char>(buffer));
            buffer = 0;
            bitCount = 0;
        }
    }
};



// Internal structure for Huffman Tree
struct HuffmanNode {
    int value;
    unsigned freq;
    HuffmanNode *left, *right;
    HuffmanNode(int v, unsigned f) : value(v), freq(f), left(nullptr), right(nullptr) {}
};




#include <queue>
#include <map>

// Node for the Huffman Tree
struct Node {
    int id;
    int freq;
    Node *left, *right;
    Node(int i, int f) : id(i), freq(f), left(nullptr), right(nullptr) {}
};

// Comparison for the priority queue
struct Compare {
    bool operator()(Node* l, Node* r) { return l->freq > r->freq; }
};

// Recursive function to build the code table
void buildCodes(Node* root, std::string str, std::map<int, std::pair<uint32_t, int>>& huffmanTable) {
    if (!root) return;
    if (!root->left && !root->right) {
        // Convert bit-string "101" to an integer 5 and length 3
        huffmanTable[root->id] = { (uint32_t)std::bitset<32>(str).to_ulong(), (int)str.length() };
    }
    buildCodes(root->left, str + "0", huffmanTable);
    buildCodes(root->right, str + "1", huffmanTable);
}



/*


void saveSIF_claude(const std::string& path, int width, int height,
             const std::vector<PaletteEntry>& palette,
             const std::vector<int>& indexMatrix,
             const GradientData& gradients)            // ← new parameter
{

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return;
    }

    // ── 1. Header & Palette ──────────────────────────────────────────────────
    uint32_t w = width, h = height;
    uint16_t palSize = (uint16_t)palette.size();
    file.write((char*)&w, 4);
    file.write((char*)&h, 4);
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
    // Minimum bits to represent any palette index: ceil(log2(palSize))
    int bitsPerIndex = 1;
    while ((1 << bitsPerIndex) < palSize) bitsPerIndex++;

    // ── 3. RLE on index stream ───────────────────────────────────────────────
    struct RLESymbol { int value; int runLength; };
    std::vector<RLESymbol> rleStream;

    if (!indexMatrix.empty()) {
        int cur = indexMatrix[0], run = 1;
        for (size_t i = 1; i < indexMatrix.size(); i++) {
            if (indexMatrix[i] == cur) {
                run++;
            } else {
                rleStream.push_back({cur, run});
                cur = indexMatrix[i];
                run = 1;
            }
        }
        rleStream.push_back({cur, run});
    }

    // ── 4. Build Huffman over (value, runLength) PAIRS as single symbols ─────
    // Encode each pair into one integer key: key = value * MAX_RUN + (runLength-1)
    // This way Huffman sees frequent pairs (e.g. "color 3, run 1") as one symbol,
    // and rare long runs naturally get longer codes — no fixed overhead per symbol.
    int maxRun = 1;
    for (auto& s : rleStream) maxRun = std::max(maxRun, s.runLength);

    auto pairKey = [&](int value, int run) {
        return value * (maxRun + 1) + (run - 1);
    };

    std::map<int, int> freq;
    for (auto& s : rleStream) freq[pairKey(s.value, s.runLength)]++;

    // Build tree
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

    // ── 5. Write metadata ────────────────────────────────────────────────────
    file.write((char*)&bitsPerIndex, 1);  // decoder needs this to reconstruct pairs
    file.write((char*)&maxRun,       4);  // decoder needs this to decode pair keys

    // ── 6. Write Huffman table ───────────────────────────────────────────────
    // Each entry: the integer key (4 bytes), code length (1 byte), code bits (4 bytes)
    uint16_t tableEntries = (uint16_t)huffmanTable.size();
    file.write((char*)&tableEntries, 2);
    for (auto const& [key, code] : huffmanTable) {
        uint8_t len = (uint8_t)code.second;
        file.write((char*)&key,        4);
        file.write((char*)&len,        1);
        file.write((char*)&code.first, 4);
    }

    // ── 7. Write RLE count + encoded bit stream ──────────────────────────────
    uint32_t rleCount = (uint32_t)rleStream.size();
    file.write((char*)&rleCount, 4);

    BitWriter bw(file);
    for (auto& s : rleStream) {
        int key = pairKey(s.value, s.runLength);
        auto& [code, len] = huffmanTable[key];
        bw.write(code, len);
    }

    // ... all your existing code stays exactly the same up to bw.flush() ...

    bw.flush();

    // ── 9. Gradient section ──────────────────────────────────────────────────

    // 9a. Precision flag (1 byte) — tells decoder how many bits per descriptor
    uint8_t precisionByte = (uint8_t)gradients.precision;
    file.write((char*)&precisionByte, 1);

    // 9b. Gradient queue
    //     We pack multiple descriptors into bytes using BitWriter
    uint32_t queueSize = (uint32_t)gradients.queue.size();
    file.write((char*)&queueSize, 4);

    BitWriter bwGrad(file);
    int precBits = (int)gradients.precision;
    for (uint8_t desc : gradients.queue)
        bwGrad.write(desc, precBits);
    bwGrad.flush();

    // 9c. Change points
    uint32_t cpCount = (uint32_t)gradients.changePoints.size();
    file.write((char*)&cpCount, 4);
    for (const auto& cp : gradients.changePoints) {
        file.write((char*)&cp.x,        2);  // uint16_t x
        file.write((char*)&cp.y,        2);  // uint16_t y
        file.write((char*)&cp.queueIdx, 4);  // uint32_t index into queue
    }

    file.close();

    // ── 10. Updated statistics ────────────────────────────────────────────────
    size_t fileSize = std::filesystem::file_size(path);
    float  bpp      = (float)(fileSize * 8) / (width * height);
    std::cout << "\n--- Final File ---\n";
    std::cout << "SIF File Size:   " << fileSize/1024 << " KB\n";
    std::cout << "Bits Per Pixel:  " << bpp << " bpp\n";
    std::cout << "Compression:     " << (float)(width*height*24)/(fileSize*8) << ":1\n";
    std::cout << "Location: " << std::filesystem::absolute(path) << "\n";
}

*/

#include "ResidualEncoder.hpp"
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



#endif