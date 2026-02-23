#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include "sif_decoder.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// ── Lab → XYZ → RGB conversion ───────────────────────────────────────────────
static const float REF_X = 95.047f;
static const float REF_Y = 100.000f;
static const float REF_Z = 108.883f;

static float labInvF(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta) return t * t * t;
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

static float linearToSRGB(float val) {
    val = std::clamp(val, 0.0f, 1.0f);
    if (val <= 0.0031308f) return 12.92f * val;
    return 1.055f * std::pow(val, 1.0f / 2.4f) - 0.055f;
}

struct RGB { uint8_t r, g, b; };
struct LabF { float L, a, b; };  // floating point Lab pixel

static RGB labToRGB(float L, float a, float b) {
    float fy = (L + 16.0f) / 116.0f;
    float fx = (a / 500.0f) + fy;
    float fz = fy - (b / 200.0f);

    float X = REF_X * labInvF(fx);
    float Y = REF_Y * labInvF(fy);
    float Z = REF_Z * labInvF(fz);

    X /= 100.0f; Y /= 100.0f; Z /= 100.0f;

    float r_lin =  3.2406f * X - 1.5372f * Y - 0.4986f * Z;
    float g_lin = -0.9689f * X + 1.8758f * Y + 0.0415f * Z;
    float b_lin =  0.0557f * X - 0.2040f * Y + 1.0570f * Z;

    return RGB {
        (uint8_t)std::round(linearToSRGB(r_lin) * 255.0f),
        (uint8_t)std::round(linearToSRGB(g_lin) * 255.0f),
        (uint8_t)std::round(linearToSRGB(b_lin) * 255.0f)
    };
}


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
        default: return 3;
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
    std::vector<LabF>&       labImage,   // in/out: flat Lab pixel buffer
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

    // Track which queue index was first assigned to each cluster pair
    std::map<std::pair<int,int>, uint32_t> boundaryFirstIdx;
    uint32_t nextQueueIdx = 0;

    // We scan in the same order as the encoder (top→bottom, left→right)
    // to reproduce the same queue assignment.
    std::map<std::pair<int,int>, bool> seen;

    for (int y = 0; y < height && nextQueueIdx < gradients.queue.size(); y++) {
        for (int x = 0; x < width && nextQueueIdx < gradients.queue.size(); x++) {
            int cA = indexMatrix[y * width + x];

            // Check right neighbor
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

            // Check bottom neighbor
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

    // ── Apply gradient blending at each boundary pixel ────────────────────────
    // For each pixel, check all 4 neighbors. If the neighbor belongs to a
    // different cluster, blend the two cluster colors across the transition zone
    // defined by the descriptor's width and shape.

    // Work on a copy so reads and writes don't interfere
    std::vector<LabF> output = labImage;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cA = indexMatrix[y * width + x];
            const PaletteEntry& pA = palette[cA];

            // Collect all neighboring clusters and their descriptors
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

                int   halfW = widthPixels(desc.width);
                const PaletteEntry& pB = palette[cB];

                // For each pixel in the transition zone, compute blend factor t
                // based on distance from the boundary along the gradient direction.
                for (int d = -halfW; d <= halfW; d++) {
                    int bx = x, by = y;

                    // Move along the direction perpendicular to the boundary edge
                    switch (desc.direction) {
                        case 0: bx = x + d; break;             // horizontal
                        case 1: by = y + d; break;             // vertical
                        case 2: bx = x + d; by = y + d; break; // diag-left
                        case 3: bx = x + d; by = y - d; break; // diag-right
                    }

                    if (bx < 0 || bx >= width || by < 0 || by >= height) continue;
                    if (indexMatrix[by * width + bx] != cA &&
                        indexMatrix[by * width + bx] != cB) continue;

                    // t=0 → color A, t=1 → color B
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


// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {

    std::string filePath = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif";
    if (argc > 1) filePath = argv[1];

    std::cout << "Loading: " << filePath << "\n";

    // ── Decode ───────────────────────────────────────────────────────────────
    SIFData data = loadSIF(filePath);
    if (!data.valid) {
        std::cerr << "Failed to decode SIF file.\n";
        return 1;
    }

    // ── Build flat Lab image from palette + index matrix ──────────────────────
    std::vector<LabF> labImage(data.width * data.height);
    for (int i = 0; i < data.width * data.height; i++) {
        const PaletteEntry& p = data.palette[data.indexMatrix[i]];
        labImage[i] = {p.L, p.a, p.b};
    }

    // ── Apply gradients ───────────────────────────────────────────────────────
    applyGradients(labImage, data.indexMatrix, data.palette,
                   data.gradients, data.width, data.height);

    // ── Convert Lab → RGB ─────────────────────────────────────────────────────
    std::vector<uint8_t> pixels;
    pixels.reserve(data.width * data.height * 3);

    for (int i = 0; i < data.width * data.height; i++) {
        RGB rgb = labToRGB(labImage[i].L, labImage[i].a, labImage[i].b);
        pixels.push_back(rgb.r);
        pixels.push_back(rgb.g);
        pixels.push_back(rgb.b);
    }

    // ── Save PNG ──────────────────────────────────────────────────────────────
    std::string outPath = filePath + "_reconstructed.png";
    int success = stbi_write_png(outPath.c_str(),
                                 data.width, data.height, 3,
                                 pixels.data(), data.width * 3);
    if (success)
        std::cout << "\n[SUCCESS] Saved to: " << outPath << "\n";
    else {
        std::cerr << "Failed to write PNG.\n";
        return 1;
    }

    return 0;
}