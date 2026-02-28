#pragma once

#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "PaletteEntry.hpp"
#include "GradientTypes.hpp"



// ═════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═════════════════════════════════════════════════════════════════════════════

// Classify the transition shape by looking at Lab values sampled perpendicular
// to the boundary at several points along a segment.
// labA/labB: centroid Lab of the two adjacent clusters
// pixelSamples: Lab values of pixels along the shared edge (already collected by caller)
static uint8_t classifyShape(float labAL, float labBL,
                              const std::vector<float>& transitionSamples) {
    if (transitionSamples.empty()) return 1; // default: linear

    // Normalize samples to [0,1] between the two endpoint values
    float lo = std::min(labAL, labBL);
    float hi = std::max(labAL, labBL);
    float range = hi - lo;
    if (range < 1e-4f) return 0; // no perceptible difference → sharp

    int n = (int)transitionSamples.size();
    std::vector<float> norm(n);
    for (int i = 0; i < n; i++)
        norm[i] = std::clamp((transitionSamples[i] - lo) / range, 0.0f, 1.0f);

    // Compare against ideal curves
    float errLinear = 0, errEase = 0, errS = 0;
    for (int i = 0; i < n; i++) {
        float t = (float)i / std::max(n - 1, 1);

        float ideal_linear = t;
        float ideal_ease   = t < 0.5f ? 2*t*t : 1 - std::pow(-2*t+2,2)/2; // ease-in-out
        float ideal_s      = t*t*(3 - 2*t);  // smoothstep

        errLinear += std::abs(norm[i] - ideal_linear);
        errEase   += std::abs(norm[i] - ideal_ease);
        errS      += std::abs(norm[i] - ideal_s);
    }

    // If all errors are high → sharp boundary
    float minErr = std::min({errLinear, errEase, errS});
    float threshold = 0.3f * n;
    if (minErr > threshold) return 0; // sharp

    if (minErr == errLinear) return 1; // linear
    if (minErr == errEase)   return 2; // ease-in/out
    return 3;                          // S-curve
}

// Classify gradient direction by looking at which axis has more variation
// along the boundary pixels
static uint8_t classifyDirection(const std::vector<std::pair<int,int>>& boundaryPixels) {
    if (boundaryPixels.size() < 2) return 0;

    int minX = boundaryPixels[0].first,  maxX = boundaryPixels[0].first;
    int minY = boundaryPixels[0].second, maxY = boundaryPixels[0].second;
    for (auto& [x, y] : boundaryPixels) {
        minX = std::min(minX, x); maxX = std::max(maxX, x);
        minY = std::min(minY, y); maxY = std::max(maxY, y);
    }

    int spanX = maxX - minX;
    int spanY = maxY - minY;

    if (spanX > spanY * 2) return 0;  // mostly horizontal
    if (spanY > spanX * 2) return 1;  // mostly vertical
    // Diagonal: check slope direction
    if (boundaryPixels.size() >= 2) {
        auto& first = boundaryPixels.front();
        auto& last  = boundaryPixels.back();
        bool sameSign = ((last.first - first.first) * (last.second - first.second)) >= 0;
        return sameSign ? 2 : 3; // diag-left or diag-right
    }
    return 0;
}

// Classify transition width by counting pixels that are "in between" the two colors
static uint8_t classifyWidth(const std::vector<float>& transitionSamples, float labAL, float labBL) {
    float lo = std::min(labAL, labBL) + 0.1f;
    float hi = std::max(labAL, labBL) - 0.1f;
    int inBetween = 0;
    for (float v : transitionSamples)
        if (v > lo && v < hi) inBetween++;

    // Map pixel count to width class
    if (inBetween <= 1)  return 0; // ~1px
    if (inBetween <= 3)  return 1; // ~3px
    if (inBetween <= 6)  return 2; // ~6px
    return 3;                      // ~12px+
}

// Difference between two descriptors — used to detect change points
static float descriptorDifference(const GradientDescriptor& a, const GradientDescriptor& b) {
    float diff = 0;
    if (a.shape     != b.shape)     diff += 1.0f;
    if (a.direction != b.direction) diff += 0.5f;
    if (a.width     != b.width)     diff += 0.3f;
    return diff;
}


// ═════════════════════════════════════════════════════════════════════════════
// Main encoder function
// ═════════════════════════════════════════════════════════════════════════════


GradientData encodeGradients(
    const std::vector<int>&          indexMatrix,
    const std::vector<LabPixelFlat>& imgLab,
    int width, int height,
    GradientPrecision precision  = GradientPrecision::BITS_4,
    float changeThreshold        = 0.5f,  // lowered: catches more subtle changes
    int   segmentSize            = 8)     // halved: more frequent descriptor updates
{
    GradientData result;
    result.precision = precision;

    std::map<std::pair<int,int>, GradientDescriptor> activeBoundary;

    struct SegmentAccumulator {
        std::vector<std::pair<int,int>> pixels;
        std::vector<float>              samples;
        int   clusterA = -1, clusterB = -1;
        float centroidLA = 0, centroidLB = 0;
    };

    int numClusters = *std::max_element(indexMatrix.begin(), indexMatrix.end()) + 1;
    std::vector<float> clusterL(numClusters, 0);
    std::vector<int>   clusterCount(numClusters, 0);
    for (int i = 0; i < (int)indexMatrix.size(); i++) {
        int c = indexMatrix[i];
        clusterL[c] += imgLab[i].L;
        clusterCount[c]++;
    }
    for (int c = 0; c < numClusters; c++)
        if (clusterCount[c] > 0) clusterL[c] /= clusterCount[c];

    std::map<std::pair<int,int>, SegmentAccumulator> accumulators;

    auto flushSegment = [&](std::pair<int,int> pairKey, SegmentAccumulator& acc) {
        if (acc.pixels.empty()) return;

        GradientDescriptor desc;
        desc.shape     = classifyShape(acc.centroidLA, acc.centroidLB, acc.samples);
        desc.direction = classifyDirection(acc.pixels);
        desc.width     = classifyWidth(acc.samples, acc.centroidLA, acc.centroidLB);

        // ── Sensitivity improvement 1: widen the transition zone ─────────────
        // Bump width up by one level so gradients spread further from the edge.
        // This makes the effect visible even in narrow boundary regions.
        if (desc.width < 3) desc.width++;

        auto it = activeBoundary.find(pairKey);
        if (it == activeBoundary.end()) {
            activeBoundary[pairKey] = desc;
            result.queue.push_back(desc);
        } else {
            // ── Sensitivity improvement 2: finer difference measure ───────────
            // Original used integer field comparisons (0 or 1 per field).
            // Now we also consider how many fields changed, not just whether
            // the total crosses the threshold.
            float diff = descriptorDifference(it->second, desc);

            // Additionally force a new descriptor if direction changed —
            // a direction flip always means a meaningfully different gradient
            bool directionFlipped = (it->second.direction != desc.direction);

            if (diff >= changeThreshold || directionFlipped) {
                ChangePoint cp;
                cp.x        = (uint16_t)acc.pixels.front().first;
                cp.y        = (uint16_t)acc.pixels.front().second;
                cp.queueIdx = (uint32_t)result.queue.size();
                result.changePoints.push_back(cp);
                result.queue.push_back(desc);
                it->second = desc;
            }
        }

        acc.pixels.clear();
        acc.samples.clear();
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cA = indexMatrix[y * width + x];

            auto processNeighbor = [&](int nx, int ny) {
                int cB = indexMatrix[ny * width + nx];
                if (cA == cB) return;

                auto key = std::make_pair(std::min(cA, cB), std::max(cA, cB));
                auto& acc = accumulators[key];
                acc.clusterA   = cA; acc.clusterB = cB;
                acc.centroidLA = clusterL[cA];
                acc.centroidLB = clusterL[cB];
                acc.pixels.push_back({x, y});
                acc.samples.push_back(imgLab[y * width + x].L);
                acc.samples.push_back(imgLab[ny * width + nx].L);

                // ── Sensitivity improvement 3: also sample diagonal neighbors ─
                // Original only sampled the two pixels on each side of the edge.
                // Adding diagonal samples gives classifyShape more data points
                // to fit the curve against, making shape classification more accurate.
                for (int ddx = -1; ddx <= 1; ddx++) {
                    int sx = x + ddx, sy = ny;
                    if (sx >= 0 && sx < width &&
                        indexMatrix[sy * width + sx] == cB)
                        acc.samples.push_back(imgLab[sy * width + sx].L);
                }

                if ((int)acc.pixels.size() >= segmentSize)
                    flushSegment(key, acc);
            };

            if (x + 1 < width)  processNeighbor(x + 1, y);
            if (y + 1 < height) processNeighbor(x, y + 1);
        }
    }

    for (auto& [key, acc] : accumulators)
        flushSegment(key, acc);

    std::cout << "\n--- Gradient Encoding ---\n";
    std::cout << "Precision:      " << (int)precision << " bits per descriptor\n";
    std::cout << "Segment size:   " << segmentSize << " pixels\n";
    std::cout << "Change thresh:  " << changeThreshold << "\n";
    std::cout << "Queue size:     " << result.queue.size() << " descriptors\n";
    std::cout << "Change points:  " << result.changePoints.size() << "\n";
    std::cout << "Gradient data:  "
              << result.queue.size() * (int)precision / 8 +
                 result.changePoints.size() * (2+2+4)
              << " bytes (approx)\n";

    return result;
}