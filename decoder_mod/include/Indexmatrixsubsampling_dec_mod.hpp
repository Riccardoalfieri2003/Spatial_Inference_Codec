#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include "sif_decoder_mod.hpp"

// Find the palette index whose Lab color is closest to the given Lab values.
// To keep this fast, we only search among a small candidate set rather than
// the whole palette.
static int findClosestInPalette(
    float L, float a, float b,
    const std::vector<PaletteEntry>& palette,
    const std::vector<int>& candidates)   // only search these indices
{
    int   bestIdx  = candidates[0];
    float bestDist = 1e9f;

    for (int idx : candidates) {
        const PaletteEntry& p = palette[idx];
        float dL = p.L - L;
        float da = p.a - a;
        float db = p.b - b;
        float d  = dL*dL + da*da + db*db;  // squared distance, no sqrt needed
        if (d < bestDist) {
            bestDist = d;
            bestIdx  = idx;
        }
    }
    return bestIdx;
}

// Average two palette colors in Lab space and find the closest palette entry
// Only searches among the two source indices and any palette entries whose
// Lab color falls between them (L within range) — keeps the search fast.
static int interpolatePaletteIndex(
    int idxA, int idxB,
    const std::vector<PaletteEntry>& palette)
{
    if (idxA == idxB) return idxA;

    const PaletteEntry& pA = palette[idxA];
    const PaletteEntry& pB = palette[idxB];

    // Target: midpoint in Lab space
    float midL = (pA.L + pB.L) * 0.5f;
    float midA = (pA.a + pB.a) * 0.5f;
    float midB = (pA.b + pB.b) * 0.5f;

    // Build candidate set: the two source entries + any entries whose L
    // falls between the two (perceptually "in between" colors)
    float loL = std::min(pA.L, pB.L);
    float hiL = std::max(pA.L, pB.L);
    float loA = std::min(pA.a, pB.a);
    float hiA = std::max(pA.a, pB.a);
    float loB = std::min(pA.b, pB.b);
    float hiB = std::max(pA.b, pB.b);

    // Add some margin so we don't miss near-boundary colors
    float margin = 5.0f;

    std::vector<int> candidates;
    candidates.push_back(idxA);
    candidates.push_back(idxB);

    for (int i = 0; i < (int)palette.size(); i++) {
        if (i == idxA || i == idxB) continue;
        const PaletteEntry& p = palette[i];
        if (p.L >= loL - margin && p.L <= hiL + margin &&
            p.a >= loA - margin && p.a <= hiA + margin &&
            p.b >= loB - margin && p.b <= hiB + margin)
            candidates.push_back(i);
    }

    return findClosestInPalette(midL, midA, midB, palette, candidates);
}


// ── Decoder: reconstruct the full index matrix from the subsampled one ────────
//
// Two passes, matching the order described:
//   Pass 1 — Fill missing COLUMNS (even x) by averaging left and right neighbors
//   Pass 2 — Fill missing ROWS    (even y) by averaging top and bottom neighbors
//
// After pass 1 the full-width, half-height matrix is available.
// After pass 2 the full-width, full-height matrix is ready.
//
std::vector<int> upsampleIndexMatrix(
    const std::vector<int>&          subMatrix,
    int subWidth, int subHeight,
    int fullWidth, int fullHeight,
    const std::vector<PaletteEntry>& palette)
{

    // ← Add this guard
    if (subWidth <= 0 || subHeight <= 0 || fullWidth <= 0 || fullHeight <= 0) {
        std::cerr << "upsampleIndexMatrix: invalid dimensions "
                  << subWidth << "x" << subHeight << " → "
                  << fullWidth << "x" << fullHeight << "\n";
        return subMatrix;
    }
    if ((int)subMatrix.size() != subWidth * subHeight) {
        std::cerr << "upsampleIndexMatrix: subMatrix size mismatch, expected "
                  << subWidth * subHeight << " got " << subMatrix.size() << "\n";
        return subMatrix;
    }

    // ── Step 1: expand subsampled matrix into a full-width, half-height grid ──
    // The subsampled pixels sit at odd x,y in the final grid.
    // First we build the half-height grid (all rows are "odd" rows).

    // halfGrid has dimensions fullWidth × subHeight
    // It will contain correct values at odd x, interpolated values at even x.
    std::vector<int> halfGrid(fullWidth * subHeight, 0);

    // Plant the known values at odd x positions
    for (int sy = 0; sy < subHeight; sy++) {
        for (int sx = 0; sx < subWidth; sx++) {
            int fx = sx * 2 + 1;  // odd x in full grid
            if (fx < fullWidth)
                halfGrid[sy * fullWidth + fx] = subMatrix[sy * subWidth + sx];
        }
    }

    // Pass 1: fill even x columns by interpolating between left and right odd neighbors
    for (int sy = 0; sy < subHeight; sy++) {
        for (int fx = 0; fx < fullWidth; fx++) {
            if (fx % 2 == 1) continue;  // already filled (odd x = known pixel)

            // Left neighbor (odd x to the left)
            int lx = fx - 1;
            // Right neighbor (odd x to the right)
            int rx = fx + 1;

            int idxA, idxB;

            if (lx < 0 && rx < fullWidth) {
                // At left border: copy from right
                idxA = idxB = halfGrid[sy * fullWidth + rx];
            } else if (rx >= fullWidth && lx >= 0) {
                // At right border: copy from left
                idxA = idxB = halfGrid[sy * fullWidth + lx];
            } else if (lx >= 0 && rx < fullWidth) {
                idxA = halfGrid[sy * fullWidth + lx];
                idxB = halfGrid[sy * fullWidth + rx];
            } else {
                continue;
            }

            halfGrid[sy * fullWidth + fx] = interpolatePaletteIndex(idxA, idxB, palette);
        }
    }

    // ── Step 2: expand halfGrid into the full-height grid ────────────────────
    // halfGrid rows correspond to ODD y positions in the final grid.
    // fullGrid has dimensions fullWidth × fullHeight.
    std::vector<int> fullGrid(fullWidth * fullHeight, 0);

    // Plant known rows at odd y positions
    for (int sy = 0; sy < subHeight; sy++) {
        int fy = sy * 2 + 1;  // odd y in full grid
        if (fy < fullHeight) {
            for (int fx = 0; fx < fullWidth; fx++)
                fullGrid[fy * fullWidth + fx] = halfGrid[sy * fullWidth + fx];
        }
    }

    // Pass 2: fill even y rows by interpolating between top and bottom odd neighbors
    for (int fy = 0; fy < fullHeight; fy++) {
        if (fy % 2 == 1) continue;  // already filled

        int ty = fy - 1;  // top odd row
        int by = fy + 1;  // bottom odd row

        for (int fx = 0; fx < fullWidth; fx++) {
            int idxA, idxB;

            if (ty < 0 && by < fullHeight) {
                idxA = idxB = fullGrid[by * fullWidth + fx];
            } else if (by >= fullHeight && ty >= 0) {
                idxA = idxB = fullGrid[ty * fullWidth + fx];
            } else if (ty >= 0 && by < fullHeight) {
                idxA = fullGrid[ty * fullWidth + fx];
                idxB = fullGrid[by * fullWidth + fx];
            } else {
                continue;
            }

            fullGrid[fy * fullWidth + fx] = interpolatePaletteIndex(idxA, idxB, palette);
        }
    }

    std::cout << "Index matrix upsampled: "
              << subWidth << "x" << subHeight << " → "
              << fullWidth << "x" << fullHeight << "\n";

    return fullGrid;
}