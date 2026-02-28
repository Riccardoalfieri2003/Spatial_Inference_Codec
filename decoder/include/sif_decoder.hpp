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
#include "ResidualEncoder.hpp"
#include <PaletteEntry.hpp>
#include "Indexmatrixsubsampling_dec.hpp"



// ── Gradient precision (mirrors GradientEncoder.hpp) ─────────────────────────
enum class GradientPrecision : uint8_t {
    BITS_2 = 2,
    BITS_4 = 4,
    BITS_6 = 6
};

// ── Gradient descriptor ───────────────────────────────────────────────────────
struct GradientDescriptor {
    uint8_t shape;      // 0=sharp, 1=linear, 2=ease-in/out, 3=S-curve
    uint8_t direction;  // 0=horizontal, 1=vertical, 2=diag-left, 3=diag-right
    uint8_t width;      // 0=1px, 1=3px, 2=6px, 3=12px

    static GradientDescriptor unpack(uint8_t bits, GradientPrecision prec) {
        GradientDescriptor d{0, 0, 0};
        switch (prec) {
            case GradientPrecision::BITS_2:
                d.shape     = bits & 0x03;
                break;
            case GradientPrecision::BITS_4:
                d.shape     = (bits >> 2) & 0x03;
                d.direction = bits & 0x03;
                break;
            case GradientPrecision::BITS_6:
            default:
                d.shape     = (bits >> 4) & 0x03;
                d.direction = (bits >> 2) & 0x03;
                d.width     = bits & 0x03;
                break;
        }
        return d;
    }
};

// ── Change point ──────────────────────────────────────────────────────────────
struct ChangePoint {
    uint16_t x, y;
    uint32_t queueIdx;  // index into the gradient queue where new descriptor lives
};

// ── Gradient data read from file ──────────────────────────────────────────────
struct GradientData {
    GradientPrecision              precision;
    std::vector<GradientDescriptor> queue;        // decoded descriptors in order
    std::vector<ChangePoint>        changePoints;
    bool valid = false;
};



// ── Full decoded SIF data ─────────────────────────────────────────────────────
struct SIFData {
    int width, height;         // subsampled dims (used during loading)
    int origWidth, origHeight; // ← NEW: full resolution dims
    std::vector<PaletteEntry>  palette;
    std::vector<int>           indexMatrix;
    GradientData               gradients;
    ResidualData               residual;
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

    void sync() {
        // Discard any partially-read byte remaining in buffer
        buffer   = 0;
        bitCount = 0;
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

    bool reduceMatrix = true;

    // ── 1. Header ────────────────────────────────────────────────────────────
    uint32_t w = 0, h = 0;
    uint32_t origW = 0, origH = 0;
    uint16_t palSize = 0;

    file.read((char*)&w, 4);
    file.read((char*)&h, 4);

    if (reduceMatrix) {
        file.read((char*)&origW,   4);
        file.read((char*)&origH,   4);
    }

    file.read((char*)&palSize, 2);

    result.width      = (int)w;
    result.height     = (int)h;
    result.origWidth  = reduceMatrix ? (int)origW : (int)w;
    result.origHeight = reduceMatrix ? (int)origH : (int)h;

    std::cout << "DEBUG header: subW=" << w << " subH=" << h
              << " origW=" << result.origWidth << " origH=" << result.origHeight
              << " palSize=" << palSize << "\n";

    // ── 2. Palette ───────────────────────────────────────────────────────────
    result.palette.resize(palSize);
    for (auto& p : result.palette) {
        int8_t L, a, b, e;
        file.read((char*)&L, 1);
        file.read((char*)&a, 1);
        file.read((char*)&b, 1);
        file.read((char*)&e, 1);
        p.L = (float)L; p.a = (float)a;
        p.b = (float)b; p.error = (float)e;
    }

    // ── 3. Metadata ───────────────────────────────────────────────────────────
    uint8_t  bitsPerIndex = 0;
    uint32_t maxRun       = 0;
    file.read((char*)&bitsPerIndex, 1);
    file.read((char*)&maxRun,       4);

    // ── 4. Huffman table ──────────────────────────────────────────────────────
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

    // ── 5. RLE count + bit stream ─────────────────────────────────────────────
    uint32_t rleCount = 0;
    file.read((char*)&rleCount, 4);

    BitReader br(file);
    result.indexMatrix.reserve(result.width * result.height);

    for (uint32_t i = 0; i < rleCount; i++) {
        int pairKey = decodeNext(root, br);
        if (pairKey < 0) break;

        int value     =  pairKey / (int)(maxRun + 1);
        int runLength = (pairKey % (int)(maxRun + 1)) + 1;

        for (int r = 0; r < runLength; r++)
            result.indexMatrix.push_back(value);
    }

    freeTree(root);

    // ── 6. Upsample if matrix was subsampled ──────────────────────────────────
    if (reduceMatrix) {
        result.indexMatrix = upsampleIndexMatrix(
            result.indexMatrix,
            result.width,     result.height,
            result.origWidth, result.origHeight,
            result.palette);
        result.width  = result.origWidth;
        result.height = result.origHeight;
    }

    // Sanity check
    int expectedPixels = result.width * result.height;
    if ((int)result.indexMatrix.size() != expectedPixels) {
        std::cerr << "Warning: expected " << expectedPixels
                  << " pixels but decoded " << result.indexMatrix.size() << "\n";
    } else {
        result.valid = true;
    }

    // ── 6. Gradient section (optional — file may not have it) ─────────────────
    // We check if there are bytes remaining before trying to read gradients.
    // This keeps the decoder backward-compatible with files that have no gradient data.
    {
        // Peek: try to read the precision byte
        uint8_t precByte = 0;
        file.read((char*)&precByte, 1);

        if (!file.fail() &&
            (precByte == 2 || precByte == 4 || precByte == 6)) {

            result.gradients.precision = (GradientPrecision)precByte;
            int precBits = (int)result.gradients.precision;

            // 6a. Queue
            uint32_t queueSize = 0;
            file.read((char*)&queueSize, 4);

            BitReader brGrad(file);
            result.gradients.queue.reserve(queueSize);
            for (uint32_t i = 0; i < queueSize; i++) {
                uint8_t packed = (uint8_t)brGrad.read(precBits);
                result.gradients.queue.push_back(
                    GradientDescriptor::unpack(packed, result.gradients.precision));
            }

            // 6b. Change points
            uint32_t cpCount = 0;
            file.read((char*)&cpCount, 4);
            result.gradients.changePoints.reserve(cpCount);
            for (uint32_t i = 0; i < cpCount; i++) {
                ChangePoint cp;
                file.read((char*)&cp.x,        2);
                file.read((char*)&cp.y,        2);
                file.read((char*)&cp.queueIdx, 4);
                result.gradients.changePoints.push_back(cp);
            }

            result.gradients.valid = true;

            std::cout << "Gradient data found:\n";
            std::cout << "  Precision:     " << precBits << " bits\n";
            std::cout << "  Queue entries: " << result.gradients.queue.size() << "\n";
            std::cout << "  Change points: " << result.gradients.changePoints.size() << "\n";
        } else {
            std::cout << "No gradient data in file (older format).\n";
        }
    }

    // ── 7. Residual section (optional) ───────────────────────────────────────
    uint8_t residualMagic = 0;
    file.read((char*)&residualMagic, 1);

    if (!file.fail() && residualMagic == 0xDC) {
        result.residual = readResidual(file, result.width, result.height);

        if (result.residual.valid) {
            std::cout << "Residual data found:\n";
            std::cout << "  Block size:    " << result.residual.config.blockSize << "x"
                                                << result.residual.config.blockSize << "\n";
            std::cout << "  Coeffs kept:   " << result.residual.config.keepCoeffs << "\n";
            std::cout << "  Quant step:    " << result.residual.config.quantStep << "\n";
            std::cout << "  Total coeffs:  " << result.residual.coefficients.size() << "\n";
        }
    } else {
        std::cout << "No residual data in file.\n";
    }


    file.close();

    file.close();

    std::cout << "\n--- SIF Decode Complete ---\n";
    std::cout << "Resolution:   " << result.width << "x" << result.height << "\n";
    std::cout << "Palette size: " << palSize << " colors\n";
    std::cout << "Pixels:       " << result.indexMatrix.size() << "\n";

    return result;
}

#endif // SIF_DECODER_HPP