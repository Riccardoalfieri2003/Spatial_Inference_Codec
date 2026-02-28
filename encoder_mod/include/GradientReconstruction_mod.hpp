#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <PaletteEntry.hpp>
#include "GradientTypes.hpp"






// ═════════════════════════════════════════════════════════════════════════════
// Easing functions  (t is always in [0, 1])
// ═════════════════════════════════════════════════════════════════════════════
static float applyShape(float t, uint8_t shape) {
    switch (shape) {
        case 0: return t < 0.5f ? 0.0f : 1.0f;           // sharp (step)
        case 1: return t;                                  // linear
        case 2: return t < 0.5f                            // ease-in/out
                    ? 2.0f * t * t
                    : 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
        case 3: return t * t * (3.0f - 2.0f * t);        // S-curve (smoothstep)
        default: return t;
    }
}

// Width in pixels for each width code
static int widthPixels(uint8_t w) {
    switch (w) {
        case 0: return 1;
        case 1: return 3;
        case 2: return 6;
        case 3: return 12;
        default: return 2;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Gradient application
//
// Strategy:
//   1. Build a float Lab image from the index matrix (flat quantized colors).
//   2. Find boundary pixels between different clusters.
//   3. For each boundary pixel, look up the active gradient descriptor using
//      the queue + change points, then blend colors across the transition zone.
//   4. Convert the final Lab image to RGB and write PNG.
// ═════════════════════════════════════════════════════════════════════════════

// Build a map from (x,y) → active queue index using the change point list.
// The map only contains pixels where a change point fires; all other boundary
// pixels use the first descriptor for that cluster pair.
static std::map<std::pair<int,int>, uint32_t>
buildChangePointMap(const std::vector<ChangePoint>& changePoints) {
    std::map<std::pair<int,int>, uint32_t> cpMap;
    for (const auto& cp : changePoints)
        cpMap[{cp.x, cp.y}] = cp.queueIdx;
    return cpMap;
}

// Returns the active descriptor index for a given boundary pixel.
// We look up whether this pixel (or any recent pixel on the same boundary)
// has triggered a change point.
static uint32_t activeQueueIdx(
    int x, int y,
    const std::map<std::pair<int,int>, uint32_t>& cpMap,
    const std::map<std::pair<int,int>, uint32_t>& boundaryFirstIdx,
    std::pair<int,int> clusterPair)
{
    // Check if this exact pixel is a change point
    auto cpIt = cpMap.find({x, y});
    if (cpIt != cpMap.end()) return cpIt->second;

    // Otherwise use the first-encounter index for this cluster pair
    auto bIt = boundaryFirstIdx.find(clusterPair);
    if (bIt != boundaryFirstIdx.end()) return bIt->second;

    return 0; // fallback
}


void applyGradients(
    std::vector<LabF>&       labImage,
    const std::vector<int>&  indexMatrix,
    const std::vector<PaletteEntry>& palette,
    const GradientData&      gradients,
    int width, int height)
{
    if (!gradients.valid || gradients.queue.empty()) {
        std::cout << "No gradient data to apply.\n";
        return;
    }

    auto cpMap = buildChangePointMap(gradients.changePoints);

    std::map<std::pair<int,int>, uint32_t> boundaryFirstIdx;
    uint32_t nextQueueIdx = 0;
    std::map<std::pair<int,int>, bool> seen;

    for (int y = 0; y < height && nextQueueIdx < gradients.queue.size(); y++) {
        for (int x = 0; x < width && nextQueueIdx < gradients.queue.size(); x++) {
            int cA = indexMatrix[y * width + x];

            if (x + 1 < width) {
                int cB = indexMatrix[y * width + (x + 1)];
                if (cA != cB) {
                    auto key = std::make_pair(std::min(cA, cB), std::max(cA, cB));
                    if (!seen.count(key)) {
                        seen[key] = true;
                        boundaryFirstIdx[key] = nextQueueIdx++;
                    }
                }
            }

            if (y + 1 < height) {
                int cB = indexMatrix[(y + 1) * width + x];
                if (cA != cB) {
                    auto key = std::make_pair(std::min(cA, cB), std::max(cA, cB));
                    if (!seen.count(key)) {
                        seen[key] = true;
                        boundaryFirstIdx[key] = nextQueueIdx++;
                    }
                }
            }
        }
    }

    std::vector<LabF> output = labImage;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cA = indexMatrix[y * width + x];
            const PaletteEntry& pA = palette[cA];

            int neighbors[4][2] = {{x+1,y},{x-1,y},{x,y+1},{x,y-1}};
            for (auto& nb : neighbors) {
                int nx = nb[0], ny = nb[1];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                int cB = indexMatrix[ny * width + nx];
                if (cA == cB) continue;

                auto key = std::make_pair(std::min(cA, cB), std::max(cA, cB));
                uint32_t qIdx = activeQueueIdx(x, y, cpMap, boundaryFirstIdx, key);

                if (qIdx >= gradients.queue.size()) continue;
                const GradientDescriptor& desc = gradients.queue[qIdx];

                int halfW = widthPixels(desc.width);
                const PaletteEntry& pB = palette[cB];

                // Blend along the line from current pixel toward neighbor
                // No stored direction needed — derived from actual neighbor position
                for (int d = -halfW; d <= halfW; d++) {
                    int bx, by;
                    if (halfW == 0) {
                        bx = x; by = y;
                    } else {
                        bx = x + (int)std::round((float)d * (nx - x) / halfW);
                        by = y + (int)std::round((float)d * (ny - y) / halfW);
                    }

                    if (bx < 0 || bx >= width || by < 0 || by >= height) continue;
                    if (indexMatrix[by * width + bx] != cA &&
                        indexMatrix[by * width + bx] != cB) continue;

                    float t = (float)(d + halfW) / (float)(2 * halfW + 1);
                    t = applyShape(t, desc.shape);

                    LabF& px = output[by * width + bx];
                    px.L = pA.L + t * (pB.L - pA.L);
                    px.a = pA.a + t * (pB.a - pA.a);
                    px.b = pA.b + t * (pB.b - pA.b);
                }
            }
        }
    }

    labImage = output;
    std::cout << "Gradients applied.\n";
}