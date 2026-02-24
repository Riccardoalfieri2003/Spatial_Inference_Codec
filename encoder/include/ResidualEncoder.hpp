#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═════════════════════════════════════════════════════════════════════════════
// DCT Residual Encoder
//
// Pipeline:
//   1. Compute residual = original Lab - quantized Lab (per channel)
//   2. Split residual into NxN blocks
//   3. Apply 2D DCT to each block
//   4. Keep only the top-left KxK coefficients (low frequency)
//   5. Quantize those coefficients to int8
//   6. Serialize to file
//
// Parameters you can tune:
//   blockSize  : size of DCT block (8 = standard JPEG, 16 = smoother)
//   keepCoeffs : how many low-freq coefficients to keep per block (1-blockSize²)
//                fewer = smaller file, less gradient recovery
//                more  = larger file, better gradient recovery
//   quantStep  : quantization step for DCT coefficients
//                smaller = more precision, larger = more compression
// ═════════════════════════════════════════════════════════════════════════════

struct ResidualConfig {
    int   blockSize  = 8;    // DCT block size (8 or 16 recommended)
    int   keepCoeffs = 6;    // how many low-freq DCT coefficients to keep per block
                             // e.g. 6 keeps DC + 5 lowest AC terms (zig-zag order)
    float quantStep  = 2.0f; // quantization step for coefficients
                             // 1.0 = fine, 4.0 = coarse (smaller file)
};

struct ResidualData {
    ResidualConfig config;
    int width, height;
    // Stored coefficients: one int8 per kept coefficient per block per channel (L,a,b)
    // Layout: [block0_L_coeff0, block0_L_coeff1, ..., block0_a_coeff0, ...]
    std::vector<int8_t> coefficients;
    bool valid = false;
};


// ── Zig-zag scan order for an NxN block ──────────────────────────────────────
// Returns pairs (row, col) in zig-zag order starting from DC (0,0)
static std::vector<std::pair<int,int>> zigzagOrder(int N) {
    std::vector<std::pair<int,int>> order;
    order.reserve(N * N);
    int r = 0, c = 0;
    bool goingUp = true;
    while ((int)order.size() < N * N) {
        order.push_back({r, c});
        if (goingUp) {
            if (c == N-1)      { r++; goingUp = false; }
            else if (r == 0)   { c++; goingUp = false; }
            else               { r--; c++; }
        } else {
            if (r == N-1)      { c++; goingUp = true; }
            else if (c == 0)   { r++; goingUp = true; }
            else               { r++; c--; }
        }
    }
    return order;
}


// ── 2D DCT-II on a square block (in-place) ───────────────────────────────────
// Standard formula used in JPEG.
// Input/output: block[row][col], size N x N
static void dct2d(std::vector<float>& block, int N) {
    std::vector<float> temp(N * N);

    // Row-wise 1D DCT
    for (int r = 0; r < N; r++) {
        for (int u = 0; u < N; u++) {
            float sum = 0.0f;
            for (int x = 0; x < N; x++)
                sum += block[r * N + x] * std::cos((2*x + 1) * u * M_PI / (2*N));
            float cu = (u == 0) ? std::sqrt(1.0f/N) : std::sqrt(2.0f/N);
            temp[r * N + u] = cu * sum;
        }
    }

    // Column-wise 1D DCT
    for (int v = 0; v < N; v++) {
        for (int u = 0; u < N; u++) {
            float sum = 0.0f;
            for (int y = 0; y < N; y++)
                sum += temp[y * N + u] * std::cos((2*y + 1) * v * M_PI / (2*N));
            float cv = (v == 0) ? std::sqrt(1.0f/N) : std::sqrt(2.0f/N);
            block[v * N + u] = cv * sum;
        }
    }
}


// ── 2D inverse DCT on a square block (in-place) ──────────────────────────────
static void idct2d(std::vector<float>& block, int N) {
    std::vector<float> temp(N * N);

    // Column-wise inverse
    for (int y = 0; y < N; y++) {
        for (int u = 0; u < N; u++) {
            float sum = 0.0f;
            for (int v = 0; v < N; v++) {
                float cv = (v == 0) ? std::sqrt(1.0f/N) : std::sqrt(2.0f/N);
                sum += cv * block[v * N + u] * std::cos((2*y + 1) * v * M_PI / (2*N));
            }
            temp[y * N + u] = sum;
        }
    }

    // Row-wise inverse
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float sum = 0.0f;
            for (int u = 0; u < N; u++) {
                float cu = (u == 0) ? std::sqrt(1.0f/N) : std::sqrt(2.0f/N);
                sum += cu * temp[y * N + u] * std::cos((2*x + 1) * u * M_PI / (2*N));
            }
            block[y * N + x] = sum;
        }
    }
}


// ── Extract one channel from a flat Lab image ─────────────────────────────────
struct LabPixelFlat;  // forward declaration — defined in GradientEncoder.hpp

// Channel: 0=L, 1=a, 2=b
static std::vector<float> extractChannel(
    const std::vector<float>& labFlat,  // interleaved L,a,b flat array
    int width, int height, int channel)
{
    std::vector<float> ch(width * height);
    for (int i = 0; i < width * height; i++)
        ch[i] = labFlat[i * 3 + channel];
    return ch;
}


// ═════════════════════════════════════════════════════════════════════════════
// Main encoder function
// ═════════════════════════════════════════════════════════════════════════════

// originalLab: flat array of {L, a, b} for every pixel (row-major)
// quantizedLab: same layout, but using palette centroid values
// Both are interleaved: [L0, a0, b0, L1, a1, b1, ...]
ResidualData encodeResidual(
    const std::vector<float>& originalLab,
    const std::vector<float>& quantizedLab,
    int width, int height,
    const ResidualConfig& config = ResidualConfig{})
{
    ResidualData result;
    result.config = config;
    result.width  = width;
    result.height = height;

    int N    = config.blockSize;
    int keep = std::min(config.keepCoeffs, N * N);

    auto zigzag = zigzagOrder(N);

    // Number of blocks in each dimension (pad to block boundary)
    int blocksX = (width  + N - 1) / N;
    int blocksY = (height + N - 1) / N;

    // Process all 3 channels
    for (int ch = 0; ch < 3; ch++) {

        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {

                // ── Extract residual block ────────────────────────────────────
                std::vector<float> block(N * N, 0.0f);
                for (int r = 0; r < N; r++) {
                    for (int c = 0; c < N; c++) {
                        int px = bx * N + c;
                        int py = by * N + r;

                        // Clamp to image bounds (padding with 0)
                        if (px >= width || py >= height) continue;

                        int idx = py * width + px;
                        float residual = originalLab[idx * 3 + ch]
                                       - quantizedLab[idx * 3 + ch];
                        block[r * N + c] = residual;
                    }
                }

                // ── Apply 2D DCT ──────────────────────────────────────────────
                dct2d(block, N);

                // ── Keep only the first `keep` coefficients in zig-zag order ─
                for (int k = 0; k < keep; k++) {
                    auto [r, c] = zigzag[k];
                    float coeff = block[r * N + c];

                    // Quantize to int8
                    int quantized = (int)std::round(coeff / config.quantStep);
                    quantized = std::clamp(quantized, -127, 127);
                    result.coefficients.push_back((int8_t)quantized);
                }
            }
        }
    }

    result.valid = true;

    // ── Statistics ────────────────────────────────────────────────────────────
    size_t bytes = result.coefficients.size() * sizeof(int8_t)
                 + sizeof(ResidualConfig);
    std::cout << "\n--- Residual DCT Encoding ---\n";
    std::cout << "Block size:      " << N << "x" << N << "\n";
    std::cout << "Coeffs kept:     " << keep << " / " << N*N << " per block\n";
    std::cout << "Quant step:      " << config.quantStep << "\n";
    std::cout << "Total blocks:    " << blocksX * blocksY << " per channel\n";
    std::cout << "Residual size:   " << bytes / 1024 << " KB\n";

    return result;
}


// ═════════════════════════════════════════════════════════════════════════════
// Decoder side: reconstruct and apply residual
// ═════════════════════════════════════════════════════════════════════════════

// Takes the quantized Lab image and adds back the decoded residual.
// labImage: interleaved flat Lab array [L0,a0,b0, L1,a1,b1, ...]  — modified in place
void applyResidual(
    std::vector<float>& labImage,
    const ResidualData& residual,
    int width, int height,
    float strength = 0.25f)  // 0.0 = no effect, 1.0 = full correction, 0.25 = gentle guide
{
    if (!residual.valid || residual.coefficients.empty()) {
        std::cout << "No residual data to apply.\n";
        return;
    }

    int N    = residual.config.blockSize;
    int keep = std::min(residual.config.keepCoeffs, N * N);
    auto zigzag = zigzagOrder(N);

    int blocksX = (width  + N - 1) / N;
    int blocksY = (height + N - 1) / N;

    int coeffIdx = 0;

    for (int ch = 0; ch < 3; ch++) {
        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {

                // Reconstruct the residual block via IDCT
                std::vector<float> block(N * N, 0.0f);
                for (int k = 0; k < keep && coeffIdx < (int)residual.coefficients.size(); k++) {
                    auto [r, c] = zigzag[k];
                    block[r * N + c] = residual.coefficients[coeffIdx++]
                                     * residual.config.quantStep;
                }
                idct2d(block, N);

                // Compute the magnitude of change in this block
                // If the residual is small, it means flat region → apply very little
                // If the residual is large, it means gradient area → apply more
                float maxMag = 0.0f;
                for (float v : block) maxMag = std::max(maxMag, std::abs(v));

                // Normalize strength by block magnitude so flat regions are untouched
                // and gradient regions get gently guided
                float blockStrength = strength * std::min(maxMag / 10.0f, 1.0f);

                // Apply scaled residual back to image
                for (int r = 0; r < N; r++) {
                    for (int c = 0; c < N; c++) {
                        int px = bx * N + c;
                        int py = by * N + r;
                        if (px >= width || py >= height) continue;

                        int idx = py * width + px;
                        labImage[idx * 3 + ch] += block[r * N + c] * blockStrength;
                    }
                }
            }
        }
    }

    std::cout << "Residual guidance applied (strength=" << strength << ").\n";
}

// ═════════════════════════════════════════════════════════════════════════════
// Serialization helpers (called from saveSIF and loadSIF)
// ═════════════════════════════════════════════════════════════════════════════

inline void writeResidual(std::ofstream& file, const ResidualData& res) {
    // Config
    uint8_t  bs   = (uint8_t)res.config.blockSize;
    uint8_t  keep = (uint8_t)res.config.keepCoeffs;
    float    qs   = res.config.quantStep;
    file.write((char*)&bs,   1);
    file.write((char*)&keep, 1);
    file.write((char*)&qs,   4);

    // Coefficients
    uint32_t count = (uint32_t)res.coefficients.size();
    file.write((char*)&count, 4);
    file.write((char*)res.coefficients.data(), count);
}

inline ResidualData readResidual(std::ifstream& file, int width, int height) {
    ResidualData res;
    res.width  = width;
    res.height = height;

    uint8_t bs, keep;
    float   qs;
    file.read((char*)&bs,   1);
    file.read((char*)&keep, 1);
    file.read((char*)&qs,   4);

    if (file.fail()) return res;

    res.config.blockSize  = (int)bs;
    res.config.keepCoeffs = (int)keep;
    res.config.quantStep  = qs;

    uint32_t count = 0;
    file.read((char*)&count, 4);
    res.coefficients.resize(count);
    file.read((char*)res.coefficients.data(), count);

    res.valid = !file.fail();
    return res;
}