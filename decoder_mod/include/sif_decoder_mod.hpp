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
#include <PaletteEntry.hpp>
#include "Indexmatrixsubsampling_dec_mod.hpp"
#include "GradientTypes.hpp"




// ── Full decoded SIF data ─────────────────────────────────────────────────────
struct SIFData {
    int width = 0, height = 0;
    std::vector<PaletteEntry> palette;
    std::vector<int>          indexMatrix;
    GradientData              gradients;
    std::vector<PaletteEntry> residualPalette;
    std::vector<int>          residualIndexMatrix;
    GradientData              residualGradients;
    bool valid = false;
};


// ═════════════════════════════════════════════════════════════════════════════
// BitReader
// ═════════════════════════════════════════════════════════════════════════════
class BitReader {
    std::ifstream& in;
    uint8_t buffer   = 0;
    int     bitCount = 0;
public:
    BitReader(std::ifstream& f) : in(f) {}

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
            result   = (result << 1) | ((buffer >> 7) & 1);
            buffer <<= 1;
            bitCount--;
        }
        return result;
    }
};


// ═════════════════════════════════════════════════════════════════════════════
// Huffman decode tree
// ═════════════════════════════════════════════════════════════════════════════
struct DecodeNode {
    int key = -1;
    DecodeNode* left  = nullptr;
    DecodeNode* right = nullptr;
};

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

static int decodeNext(DecodeNode* root, BitReader& br) {
    DecodeNode* cur = root;
    while (cur->left || cur->right) {
        uint32_t bit = br.read(1);
        cur = bit ? cur->right : cur->left;
        if (!cur) {
            std::cerr << "decodeNext: invalid Huffman stream\n";
            return -1;
        }
    }
    return cur->key;
}

static void freeTree(DecodeNode* node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


// ═════════════════════════════════════════════════════════════════════════════
// Main decode function
// ═════════════════════════════════════════════════════════════════════════════
SIFData loadSIF(const std::string& path) {
    SIFData result;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << path << "\n";
        return result;
    }

    // ── Helper: decode a palette + RLE/Huffman index matrix ──────────────────
    auto decodeSection = [&](
        std::vector<PaletteEntry>& pal,
        std::vector<int>& idxMatrix,
        int totalPixels)
    {
        // Palette
        uint16_t palSize = 0;
        file.read((char*)&palSize, 2);
        pal.resize(palSize);
        for (auto& p : pal) {
            int8_t L, a, b, e;
            file.read((char*)&L, 1); file.read((char*)&a, 1);
            file.read((char*)&b, 1); file.read((char*)&e, 1);
            p.L = (float)L; p.a = (float)a;
            p.b = (float)b; p.error = (float)e;
        }

        // Metadata
        uint8_t  bitsPerIndex = 0;
        uint32_t maxRun       = 0;
        file.read((char*)&bitsPerIndex, 1);
        file.read((char*)&maxRun,       4);

        // Huffman table
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

        // RLE + Huffman bitstream
        uint32_t rleCount = 0;
        file.read((char*)&rleCount, 4);

        BitReader br(file);
        idxMatrix.reserve(totalPixels);
        for (uint32_t i = 0; i < rleCount; i++) {
            int pairKey = decodeNext(root, br);
            if (pairKey < 0) break;
            int value     =  pairKey / (int)(maxRun + 1);
            int runLength = (pairKey % (int)(maxRun + 1)) + 1;
            for (int r = 0; r < runLength; r++)
                idxMatrix.push_back(value);
        }

        freeTree(root);
    };

    // ── Helper: decode a gradient section ────────────────────────────────────
    auto decodeGradientSection = [&](GradientData& gradients) {
        uint8_t precByte = 0;
        file.read((char*)&precByte, 1);
        gradients.precision = (GradientPrecision)precByte;
        int precBits = (int)gradients.precision;

        uint32_t queueSize = 0;
        file.read((char*)&queueSize, 4);

        BitReader brGrad(file);
        gradients.queue.reserve(queueSize);
        for (uint32_t i = 0; i < queueSize; i++) {
            uint8_t packed = (uint8_t)brGrad.read(precBits);
            gradients.queue.push_back(
                GradientDescriptor::unpack(packed, gradients.precision));
        }

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

    // ── 1. Header ────────────────────────────────────────────────────────────
    uint32_t w = 0, h = 0;
    file.read((char*)&w, 4);
    file.read((char*)&h, 4);
    result.width  = (int)w;
    result.height = (int)h;
    int totalPixels = result.width * result.height;

    std::cout << "Header: " << w << "x" << h << "\n";

    // ── 2. Main palette + index matrix ───────────────────────────────────────
    decodeSection(result.palette, result.indexMatrix, totalPixels);

    std::cout << "Main palette: " << result.palette.size() << " colors\n";
    std::cout << "Main index matrix: " << result.indexMatrix.size() << " pixels\n";

    // ── 3. Gradient section ───────────────────────────────────────────────────
    decodeGradientSection(result.gradients);

    std::cout << "Gradients: " << result.gradients.queue.size() << " descriptors, "
              << result.gradients.changePoints.size() << " change points\n";

    // ── 4. Residual section ───────────────────────────────────────────────────
    uint8_t residualMagic = 0;
    file.read((char*)&residualMagic, 1);

    if (!file.fail() && residualMagic == 0xFE) {
        decodeSection(result.residualPalette, result.residualIndexMatrix, totalPixels);

        std::cout << "Residual palette: " << result.residualPalette.size() << " colors\n";
        std::cout << "Residual index matrix: " << result.residualIndexMatrix.size() << " pixels\n";

        // ── 5. Residual gradient section ─────────────────────────────────────
        decodeGradientSection(result.residualGradients);

        std::cout << "Residual gradients: " << result.residualGradients.queue.size()
                  << " descriptors, " << result.residualGradients.changePoints.size()
                  << " change points\n";

    } else {
        std::cout << "No residual data in file.\n";
    }

    // ── Sanity check ─────────────────────────────────────────────────────────
    if ((int)result.indexMatrix.size() == totalPixels)
        result.valid = true;
    else
        std::cerr << "Warning: expected " << totalPixels
                  << " pixels but decoded " << result.indexMatrix.size() << "\n";

    file.close();

    std::cout << "\n--- SIF Decode Complete ---\n";
    std::cout << "Resolution:      " << result.width << "x" << result.height << "\n";
    std::cout << "Main palette:    " << result.palette.size() << " colors\n";
    std::cout << "Residual palette:" << result.residualPalette.size() << " colors\n";
    std::cout << "Pixels:          " << result.indexMatrix.size() << "\n";

    return result;
}


#endif // SIF_DECODER_HPP