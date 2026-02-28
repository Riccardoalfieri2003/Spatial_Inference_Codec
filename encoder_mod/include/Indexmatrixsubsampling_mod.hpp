#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>

// ═════════════════════════════════════════════════════════════════════════════
// Index Matrix Subsampling
//
// Encoder: remove every even row and column → ~4x fewer entries to RLE/Huffman
// Decoder: reconstruct missing rows and columns by palette-aware interpolation
//
// Subsampled layout: keeps pixels at ODD coordinates (1,1), (1,3), (3,1)...
// This means:
//   original (x,y) → subsampled if x is ODD and y is ODD
//   subsampled width  = ceil(width  / 2)
//   subsampled height = ceil(height / 2)
// ═════════════════════════════════════════════════════════════════════════════


// ── Encoder: subsample the index matrix ──────────────────────────────────────
// Keeps only pixels at odd (x, y) coordinates.
// Returns the subsampled matrix and stores the subsampled dimensions.
std::vector<int> subsampleIndexMatrix(
    const std::vector<int>& indexMatrix,
    int width, int height,
    int& outWidth, int& outHeight)
{
    // Subsampled dimensions: we keep pixels at x=1,3,5,... and y=1,3,5,...
    // If width=512 → outWidth=256, if width=511 → outWidth=256
    outWidth  = width  / 2;
    outHeight = height / 2;

    std::vector<int> sub(outWidth * outHeight);

    for (int sy = 0; sy < outHeight; sy++) {
        for (int sx = 0; sx < outWidth; sx++) {
            // Map subsampled coord back to original odd coord
            int ox = sx * 2 + 1;
            int oy = sy * 2 + 1;

            // Clamp just in case
            ox = std::min(ox, width  - 1);
            oy = std::min(oy, height - 1);

            sub[sy * outWidth + sx] = indexMatrix[oy * width + ox];
        }
    }

    std::cout << "Index matrix subsampled: "
              << width << "x" << height << " → "
              << outWidth << "x" << outHeight
              << " (" << sub.size() << " entries, was " << indexMatrix.size() << ")\n";

    return sub;
}