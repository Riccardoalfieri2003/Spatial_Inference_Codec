#ifndef SIF_DECODER_HPP
#define SIF_DECODER_HPP

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <filesystem>

// ── Palette entry (mirrors the encoder's PaletteEntry) ──────────────────────
struct PaletteEntry {
    float L, a, b, error;
};

// ── Decoded SIF data ─────────────────────────────────────────────────────────
struct SIFData {
    int width, height;
    std::vector<PaletteEntry> palette;
    std::vector<int> indexMatrix;   // one palette index per pixel
    bool valid = false;
};


// ── BitReader: mirrors the encoder's BitWriter ───────────────────────────────
class BitReader {
    std::ifstream& in;
    uint8_t buffer   = 0;
    int     bitCount = 0;   // how many bits are still valid in buffer

public:
    BitReader(std::ifstream& f) : in(f) {}

    // Read `numBits` bits and return them as a uint32_t (MSB first, same as BitWriter)
    uint32_t read(int numBits) {
        uint32_t result = 0;
        for (int i = 0; i < numBits; i++) {
            if (bitCount == 0) {
                char c;
                if (!in.get(c)) {
                    std::cerr << "BitReader: unexpected end of file\n";
                    return result;
                }
                buffer   = static_cast<uint8_t>(c);
                bitCount = 8;
            }
            // Pull the MSB out of the buffer
            result   = (result << 1) | ((buffer >> 7) & 1);
            buffer <<= 1;
            bitCount--;
        }
        return result;
    }
};


// ── Huffman tree node (for decoding) ────────────────────────────────────────
struct DecodeNode {
    int  key   = -1;          // -1 → internal node, >= 0 → leaf (pair key)
    DecodeNode* left  = nullptr;
    DecodeNode* right = nullptr;
};

// Insert one code into the decode tree
static void insertCode(DecodeNode* root, uint32_t code, int len, int key) {
    DecodeNode* cur = root;
    for (int i = len - 1; i >= 0; --i) {
        bool bit = (code >> i) & 1;
        if (!bit) {
            if (!cur->left)  cur->left  = new DecodeNode();
            cur = cur->left;
        } else {
            if (!cur->right) cur->right = new DecodeNode();
            cur = cur->right;
        }
    }
    cur->key = key;
}

// Decode one symbol by walking the tree bit by bit
static int decodeNext(DecodeNode* root, BitReader& br) {
    DecodeNode* cur = root;
    while (cur->left || cur->right) {   // while not a leaf
        uint32_t bit = br.read(1);
        cur = bit ? cur->right : cur->left;
        if (!cur) {
            std::cerr << "decodeNext: invalid Huffman stream\n";
            return -1;
        }
    }
    return cur->key;
}

// Free the decode tree
static void freeTree(DecodeNode* node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


// ── Main decode function ─────────────────────────────────────────────────────
SIFData loadSIF(const std::string& path) {
    SIFData result;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return result;
    }

    // ── 1. Header ────────────────────────────────────────────────────────────
    uint32_t w = 0, h = 0;
    uint16_t palSize = 0;
    file.read((char*)&w,       4);
    file.read((char*)&h,       4);
    file.read((char*)&palSize, 2);
    result.width  = (int)w;
    result.height = (int)h;

    // ── 2. Palette ───────────────────────────────────────────────────────────
    result.palette.resize(palSize);
    for (auto& p : result.palette) {
        int8_t L, a, b, e;
        file.read((char*)&L, 1);
        file.read((char*)&a, 1);
        file.read((char*)&b, 1);
        file.read((char*)&e, 1);
        p.L     = (float)L;
        p.a     = (float)a;
        p.b     = (float)b;
        p.error = (float)e;
    }

    // ── 3. Metadata written by encoder ───────────────────────────────────────
    uint8_t  bitsPerIndex = 0;
    uint32_t maxRun       = 0;
    file.read((char*)&bitsPerIndex, 1);
    file.read((char*)&maxRun,       4);

    // ── 4. Huffman table ─────────────────────────────────────────────────────
    uint16_t tableEntries = 0;
    file.read((char*)&tableEntries, 2);

    // Build the decode tree from the stored table
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

    // ── 5. RLE count ─────────────────────────────────────────────────────────
    uint32_t rleCount = 0;
    file.read((char*)&rleCount, 4);

    // ── 6. Decode bit stream ─────────────────────────────────────────────────
    BitReader br(file);
    result.indexMatrix.reserve(result.width * result.height);

    for (uint32_t i = 0; i < rleCount; i++) {
        int pairKey = decodeNext(root, br);
        if (pairKey < 0) break;

        // Reverse of encoder's pairKey = value * (maxRun + 1) + (runLength - 1)
        int value     =  pairKey / (int)(maxRun + 1);
        int runLength = (pairKey % (int)(maxRun + 1)) + 1;

        for (int r = 0; r < runLength; r++)
            result.indexMatrix.push_back(value);
    }

    freeTree(root);
    file.close();

    // Sanity check
    int expectedPixels = result.width * result.height;
    if ((int)result.indexMatrix.size() != expectedPixels) {
        std::cerr << "Warning: expected " << expectedPixels
                  << " pixels but decoded " << result.indexMatrix.size() << "\n";
    } else {
        result.valid = true;
    }

    std::cout << "\n--- SIF Decode Complete ---\n";
    std::cout << "Resolution:   " << result.width << "x" << result.height << "\n";
    std::cout << "Palette size: " << palSize << " colors\n";
    std::cout << "Pixels:       " << result.indexMatrix.size() << "\n";

    return result;
}

#endif // SIF_DECODER_HPP
