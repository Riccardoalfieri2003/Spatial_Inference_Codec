#pragma once

#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

// ── Bit precision mode ───────────────────────────────────────────────────────
// 2 bits: shape only (4 options)          → simplest images
// 4 bits: shape + direction (4+4)         → moderate complexity
// 6 bits: shape + direction + width (4+4+4) → full detail
enum class GradientPrecision : uint8_t {
    BITS_2 = 2,
    BITS_4 = 4,
    BITS_6 = 6
};

// ── Gradient descriptor ───────────────────────────────────────────────────────
// Each field is always computed, but only the relevant bits are serialized
// depending on the chosen GradientPrecision.
struct GradientDescriptor {
    uint8_t shape;      // 2 bits: 0=sharp, 1=linear, 2=ease-in/out, 3=S-curve
    uint8_t direction;  // 2 bits: 0=horizontal, 1=vertical, 2=diag-left, 3=diag-right
    uint8_t width;      // 2 bits: 0=1px, 1=3px, 2=6px, 3=12px

    // Pack into N bits depending on precision
    uint8_t pack(GradientPrecision prec) const {
        switch (prec) {
            case GradientPrecision::BITS_2:
                return shape & 0x03;                                      // 2 bits
            case GradientPrecision::BITS_4:
                return ((shape & 0x03) << 2) | (direction & 0x03);       // 4 bits
            case GradientPrecision::BITS_6:
            default:
                return ((shape & 0x03) << 4) | ((direction & 0x03) << 2) | (width & 0x03); // 6 bits
        }
    }

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

// ── Change point: where a boundary segment changes gradient character ─────────
struct ChangePoint {
    uint16_t x, y;      // image coordinate where the new gradient starts
    uint32_t queueIdx;  // index into the gradient queue for the new descriptor
};

// ── Full gradient data produced by the encoder ───────────────────────────────
struct GradientData {
    GradientPrecision      precision;
    std::vector<uint8_t>   queue;        // packed gradient descriptors, in boundary-encounter order
    std::vector<ChangePoint> changePoints;
};


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

// imgLab: flat Lab pixel array, same layout as indexMatrix (row-major)
// Each entry is {L, a, b} for pixel at (x, y) = index [y*width + x]
struct LabPixelFlat { float L, a, b; };

GradientData encodeGradients(
    const std::vector<int>&          indexMatrix,
    const std::vector<LabPixelFlat>& imgLab,       // Lab value of every original pixel
    int width, int height,
    GradientPrecision precision = GradientPrecision::BITS_6,
    float changeThreshold = 1.0f)   // how different two descriptors must be to emit a change point
{
    GradientData result;
    result.precision = precision;

    // Track which boundary pairs we have already registered a descriptor for,
    // and what their current active descriptor is.
    // Key: {minClusterIdx, maxClusterIdx}  (always min first for consistency)
    std::map<std::pair<int,int>, GradientDescriptor> activeBoundary;

    // For change-point detection: collect boundary pixels per pair as we scan
    // We scan top-to-bottom, left-to-right. For each pixel we check its right
    // and bottom neighbor. If different cluster → boundary event.

    // We process boundaries segment by segment. A "segment" here is a run of
    // consecutive boundary pixels (in scan order) belonging to the same pair.
    // When the pair changes, or when we re-encounter a pair but the descriptor
    // has changed significantly, we emit a change point.

    struct SegmentAccumulator {
        std::vector<std::pair<int,int>> pixels;   // (x,y) of boundary pixels
        std::vector<float>              samples;  // L values perpendicular to edge
        int clusterA = -1, clusterB = -1;
        float centroidLA = 0, centroidLB = 0;
    };

    // We need palette centroids for shape analysis.
    // Extract them from imgLab by averaging per cluster.
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

    // ── Scan ─────────────────────────────────────────────────────────────────
    // We accumulate boundary pixels for each pair encountered in scan order.
    // Every SEGMENT_SIZE pixels along the same boundary we compute a descriptor
    // and check for change points.
    const int SEGMENT_SIZE = 16; // tune this: larger = less data, less precision

    // boundary pixel accumulator per pair
    std::map<std::pair<int,int>, SegmentAccumulator> accumulators;

    auto flushSegment = [&](std::pair<int,int> pairKey, SegmentAccumulator& acc) {
        if (acc.pixels.empty()) return;

        GradientDescriptor desc;
        desc.shape     = classifyShape(acc.centroidLA, acc.centroidLB, acc.samples);
        desc.direction = classifyDirection(acc.pixels);
        desc.width     = classifyWidth(acc.samples, acc.centroidLA, acc.centroidLB);

        auto it = activeBoundary.find(pairKey);
        if (it == activeBoundary.end()) {
            // First time we see this boundary → push to queue
            activeBoundary[pairKey] = desc;
            result.queue.push_back(desc.pack(precision));
        } else {
            // Already seen — check if gradient changed significantly
            float diff = descriptorDifference(it->second, desc);
            if (diff >= changeThreshold) {
                // Emit a change point at the first pixel of this segment
                ChangePoint cp;
                cp.x        = (uint16_t)acc.pixels.front().first;
                cp.y        = (uint16_t)acc.pixels.front().second;
                cp.queueIdx = (uint32_t)result.queue.size();
                result.changePoints.push_back(cp);
                result.queue.push_back(desc.pack(precision));
                it->second = desc; // update active descriptor
            }
            // else: same gradient, no new data needed
        }

        acc.pixels.clear();
        acc.samples.clear();
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cA = indexMatrix[y * width + x];

            // Check right neighbor
            if (x + 1 < width) {
                int cB = indexMatrix[y * width + (x + 1)];
                if (cA != cB) {
                    auto key = std::make_pair(std::min(cA, cB), std::max(cA, cB));
                    auto& acc = accumulators[key];
                    acc.clusterA   = cA; acc.clusterB = cB;
                    acc.centroidLA = clusterL[cA];
                    acc.centroidLB = clusterL[cB];
                    acc.pixels.push_back({x, y});
                    // Sample L values of both pixels as transition reference
                    acc.samples.push_back(imgLab[y * width + x].L);
                    acc.samples.push_back(imgLab[y * width + (x+1)].L);

                    if ((int)acc.pixels.size() >= SEGMENT_SIZE)
                        flushSegment(key, acc);
                }
            }

            // Check bottom neighbor
            if (y + 1 < height) {
                int cB = indexMatrix[(y + 1) * width + x];
                if (cA != cB) {
                    auto key = std::make_pair(std::min(cA, cB), std::max(cA, cB));
                    auto& acc = accumulators[key];
                    acc.clusterA   = cA; acc.clusterB = cB;
                    acc.centroidLA = clusterL[cA];
                    acc.centroidLB = clusterL[cB];
                    acc.pixels.push_back({x, y});
                    acc.samples.push_back(imgLab[y * width + x].L);
                    acc.samples.push_back(imgLab[(y+1) * width + x].L);

                    if ((int)acc.pixels.size() >= SEGMENT_SIZE)
                        flushSegment(key, acc);
                }
            }
        }
    }

    // Flush any remaining partial segments
    for (auto& [key, acc] : accumulators)
        flushSegment(key, acc);

    std::cout << "\n--- Gradient Encoding ---\n";
    std::cout << "Precision:      " << (int)precision << " bits per descriptor\n";
    std::cout << "Queue size:     " << result.queue.size() << " descriptors\n";
    std::cout << "Change points:  " << result.changePoints.size() << "\n";
    std::cout << "Gradient data:  "
              << result.queue.size() * (int)precision / 8 +
                 result.changePoints.size() * (2+2+4)
              << " bytes (approx)\n";

    return result;
}