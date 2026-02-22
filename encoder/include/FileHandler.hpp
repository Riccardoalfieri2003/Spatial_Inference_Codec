#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

#include <string>
#include <vector>
#include "VoxelGrid.hpp" // For PaletteEntry/LabPixel types

struct PaletteEntry { float L, a, b, error; };

#include <map>
#include <queue>
#include <bitset>


#include <fstream>   // For std::ofstream and std::ifstream
#include <iostream>  // For std::cout
#include <vector>    // For std::vector
#include <string>    // For std::string
#include <filesystem> // For creating directories (C++17 or later)




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

/*
void saveSIF(const std::string& path, int width, int height, 
             const std::vector<PaletteEntry>& palette, 
             const std::vector<int>& indexMatrix) {
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create file at " << path << std::endl;
        return;
    }

    // --- Part 1: Header & Palette ---
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
        file.write((char*)&L, 1);
        file.write((char*)&a, 1);
        file.write((char*)&b, 1);
        file.write((char*)&e, 1);
    }

    // --- Part 2: Delta Stream ---
    // Delta range is [-(palSize-1), +(palSize-1)]
    std::vector<int> deltaStream;
    int lastIdx = 0;
    for (int currentIdx : indexMatrix) {
        deltaStream.push_back(currentIdx - lastIdx);
        lastIdx = currentIdx; // Update for next pixel
    }

    // --- Part 3: Frequency & Bits ---
    // To keep this robust and error-free, we calculate the range needed
    // for all possible deltas.
    int maxPossibleDelta = palSize; 
    int bitsNeeded = (int)std::ceil(std::log2(maxPossibleDelta * 2)); 

    // --- Part 4: Writing Table Size ---
    // (Kept for your paper's structure, even if using fixed-width deltas for now)
    uint32_t tableSize = 0; // Set to 0 if not using a full tree yet to save space
    file.write((char*)&tableSize, 4);

    // --- Part 5: Bit-Packing ---
    BitWriter bw(file);
    for (int delta : deltaStream) {
        // We offset the delta so it's always positive for the BitWriter
        // Range: [-(palSize-1), (palSize-1)] + palSize = [1, 2*palSize-1]
        uint32_t encodedValue = static_cast<uint32_t>(delta + maxPossibleDelta);
        bw.write(encodedValue, bitsNeeded);
    }

    bw.flush();


    // --- Compression Metrics ---
    size_t headerBits = (4 + 4 + 2) * 8;
    size_t paletteBits = palette.size() * 4 * 8;
    size_t matrixBits = indexMatrix.size() * bitsNeeded;
    size_t totalBits = headerBits + paletteBits + (tableSize * 8 * 8) + matrixBits;

    float bpp = (float)totalBits / (width * height);
    float ratio = 24.0f / bpp; // Comparing to standard 24-bit RGB

    std::cout << "\n--- Bit Usage Report ---" << std::endl;
    std::cout << "Bits Per Index:  " << bitsNeeded << " bits" << std::endl;
    std::cout << "Total File Size: " << std::filesystem::file_size(path) << " bytes" << std::endl;
    std::cout << "Bits Per Pixel:  " << bpp << " bpp" << std::endl;
    std::cout << "Compression:     " << ratio << ":1" << std::endl;


    file.close();

    // Show exactly where it is!
    std::cout << "\n[SUCCESS] File saved!" << std::endl;
    std::cout << "Location: " << std::filesystem::absolute(path) << std::endl;
    std::cout << "Size: " << std::filesystem::file_size(path) << " bytes" << std::endl;
}
*/



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

void saveSIF(const std::string& path, int width, int height, 
             const std::vector<PaletteEntry>& palette, 
             const std::vector<int>& indexMatrix) {
    
    std::ofstream file(path, std::ios::binary);

    // 1. Header & Palette (Same as before)
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

    // 2. Build Huffman Tree
    std::map<int, int> freq;
    for (int x : indexMatrix) freq[x]++;

    std::priority_queue<Node*, std::vector<Node*>, Compare> pq;
    for (auto const& [id, f] : freq) pq.push(new Node(id, f));

    while (pq.size() != 1) {
        Node *left = pq.top(); pq.pop();
        Node *right = pq.top(); pq.pop();
        Node *top = new Node(-1, left->freq + right->freq);
        top->left = left; top->right = right;
        pq.push(top);
    }

    // 3. Generate Codes
    std::map<int, std::pair<uint32_t, int>> huffmanTable;
    buildCodes(pq.top(), "", huffmanTable);

    // 4. Save the Huffman Table to the file (so we can decode later)
    uint16_t tableEntries = (uint16_t)huffmanTable.size();
    file.write((char*)&tableEntries, 2);
    for (auto const& [id, code] : huffmanTable) {
        file.write((char*)&id, 4);           // The index
        uint8_t len = (uint8_t)code.second;
        file.write((char*)&len, 1);         // Length of bit-code
        file.write((char*)&code.first, 4);  // The bit-code itself
    }

    // 5. Write Huffman-Encoded Matrix
    BitWriter bw(file);
    for (int index : indexMatrix) {
        auto codeInfo = huffmanTable[index];
        bw.write(codeInfo.first, codeInfo.second);
    }
    bw.flush();
    file.close();

    // --- Statistics ---
    size_t fileSize = std::filesystem::file_size(path);
    std::cout << "\n--- Huffman Compression Complete ---" << std::endl;
    std::cout << "Original RGB:    " << (width * height * 3) / 1024 << " KB" << std::endl;
    std::cout << "SIF File Size:   " << fileSize / 1024 << " KB" << std::endl;
    std::cout << "Efficiency:      " << (float)(width * height * 3) / fileSize << ":1 ratio" << std::endl;

}




void saveSIF_claude(const std::string& path, int width, int height,
             const std::vector<PaletteEntry>& palette,
             const std::vector<int>& indexMatrix) {

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
    bw.flush();
    file.close();

    // ── 8. Statistics ────────────────────────────────────────────────────────
    size_t fileSize = std::filesystem::file_size(path);
    float  bpp      = (float)(fileSize * 8) / (width * height);
    std::cout << "\n--- RLE + Huffman Compression Complete ---\n";
    std::cout << "Palette Size:    " << palSize              << " colors\n";
    std::cout << "Bits Per Index:  " << bitsPerIndex         << " bits\n";
    std::cout << "RLE Symbols:     " << rleStream.size()     << " (from " << indexMatrix.size() << " pixels)\n";
    std::cout << "Unique Symbols:  " << huffmanTable.size()  << " Huffman symbols\n";
    std::cout << "Original RGB:    " << (width*height*3)/1024 << " KB\n";
    std::cout << "SIF File Size:   " << fileSize/1024         << " KB\n";
    std::cout << "Bits Per Pixel:  " << bpp                   << " bpp\n";
    std::cout << "Compression:     " << (float)(width*height*24)/(fileSize*8) << ":1\n";
    std::cout << "Location: " << std::filesystem::absolute(path) << "\n";
}



#endif