#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include "sif_decoder.hpp"
#include "ResidualEncoder.hpp"   // applyResidual()

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <random>


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










// ── Shared helper: clamp a Lab offset to stay within the error sphere ─────────
static LabF clampToSphere(float centerL, float centerA, float centerB,
                           float L, float a, float b, float radius) {
    float dL = L - centerL;
    float da = a - centerA;
    float db = b - centerB;
    float dist = std::sqrt(dL*dL + da*da + db*db);
    if (dist > radius) {
        float scale = radius / dist;
        dL *= scale; da *= scale; db *= scale;
    }
    return {centerL + dL, centerA + da, centerB + db};
}


// ── Version 1: Uniform sampling between gradient color and centroid ───────────
//
// Picks a random point on the LINE SEGMENT between the centroid and the
// gradient-derived color. Every point on that segment is guaranteed to be
// inside the sphere (since both endpoints are inside or on the surface).
// The result is then clamped as a safety net for floating point edge cases.
//
static LabF sampleUniform(const PaletteEntry& p,   // cluster centroid + error
                           const LabF& gradColor,   // color from gradient blending
                           std::mt19937& rng) {
    float radius = std::abs(p.error);

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float t = uni(rng);  // 0 = centroid, 1 = gradient color

    float L = p.L + t * (gradColor.L - p.L);
    float a = p.a + t * (gradColor.a - p.a);
    float b = p.b + t * (gradColor.b - p.b);

    // Safety clamp — should rarely trigger
    return clampToSphere(p.L, p.a, p.b, L, a, b, radius);
}


// ── Version 2: Gaussian sampling biased toward the gradient color ─────────────
//
// The distribution is centered on the MIDPOINT between centroid and gradient
// color, slightly shifted toward the gradient color (by the bias factor).
// Sigma is chosen so that 3*sigma = distance from center to midpoint,
// meaning the tails reach the centroid and the gradient boundary naturally.
// Any sample that lands beyond the centroid (wrong side of sphere) is
// reflected back, making the distribution one-sided toward the gradient color.
//
static LabF sampleGaussian(const PaletteEntry& p,  // cluster centroid + error
                            const LabF& gradColor,  // color from gradient blending
                            std::mt19937& rng,
                            float bias = 0.6f)      // 0.5 = exact midpoint, >0.5 = lean toward gradient
{
    float radius = std::abs(p.error);

    // Midpoint between centroid and gradient color, shifted by bias
    // bias=0.5 → true midpoint, bias=0.6 → 60% toward gradient color
    float midL = p.L + bias * (gradColor.L - p.L);
    float midA = p.a + bias * (gradColor.a - p.a);
    float midB = p.b + bias * (gradColor.b - p.b);

    // Sigma: distance from midpoint to centroid, divided by 3
    // so that 99.7% of samples stay between centroid and gradient side
    float segLen = std::sqrt(
        (midL - p.L) * (midL - p.L) +
        (midA - p.a) * (midA - p.a) +
        (midB - p.b) * (midB - p.b));
    float sigma = std::max(segLen / 3.0f, 1e-4f);

    std::normal_distribution<float> gauss(0.0f, sigma);

    float L, a, b;
    int maxTries = 8;
    do {
        float dL = gauss(rng);
        float da = gauss(rng);
        float db = gauss(rng);
        L = midL + dL;
        a = midA + da;
        b = midB + db;

        // Reflect samples that land on the wrong side of the centroid
        // "Wrong side" = further from gradient color than the centroid is
        float dotL = (L - p.L) * (gradColor.L - p.L);
        float dotA = (a - p.a) * (gradColor.a - p.a);
        float dotB = (b - p.b) * (gradColor.b - p.b);
        float dot  = dotL + dotA + dotB;

        if (dot >= 0.0f) break;  // on the correct side, accept

        // Reflect across the centroid
        L = p.L + (p.L - L);
        a = p.a + (p.a - a);
        b = p.b + (p.b - b);
        break;

    } while (--maxTries > 0);

    // Final clamp to sphere
    return clampToSphere(p.L, p.a, p.b, L, a, b, radius);
}





// ── Region-confined Gaussian blur ─────────────────────────────────────────────
//
// Standard box-blur approximation (3 passes), but a pixel only contributes
// to the average if it belongs to the SAME cluster as the center pixel.
// This keeps blur strictly within region boundaries.
//
static std::vector<uint8_t> blurWithinRegions(
    const std::vector<uint8_t>& pixels,
    const std::vector<int>&     indexMatrix,
    int width, int height,
    int radius = 1)             // 1 = subtle, 2 = moderate, 3 = strong
{
    std::vector<uint8_t> current = pixels;

    for (int pass = 0; pass < 3; pass++) {
        std::vector<uint8_t> temp(current.size());

        // ── Horizontal pass ───────────────────────────────────────────────────
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int centerCluster = indexMatrix[y * width + x];
                float rSum = 0, gSum = 0, bSum = 0;
                int count = 0;

                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;

                    // Only include pixels from the same region
                    if (indexMatrix[y * width + nx] != centerCluster) continue;

                    int base = (y * width + nx) * 3;
                    rSum += current[base + 0];
                    gSum += current[base + 1];
                    bSum += current[base + 2];
                    count++;
                }

                int base = (y * width + x) * 3;
                temp[base + 0] = (uint8_t)(rSum / count);
                temp[base + 1] = (uint8_t)(gSum / count);
                temp[base + 2] = (uint8_t)(bSum / count);
            }
        }

        // ── Vertical pass ─────────────────────────────────────────────────────
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int centerCluster = indexMatrix[y * width + x];
                float rSum = 0, gSum = 0, bSum = 0;
                int count = 0;

                for (int dy = -radius; dy <= radius; dy++) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;

                    // Only include pixels from the same region
                    if (indexMatrix[ny * width + x] != centerCluster) continue;

                    int base = (ny * width + x) * 3;
                    rSum += temp[base + 0];
                    gSum += temp[base + 1];
                    bSum += temp[base + 2];
                    count++;
                }

                int base = (y * width + x) * 3;
                current[base + 0] = (uint8_t)(rSum / count);
                current[base + 1] = (uint8_t)(gSum / count);
                current[base + 2] = (uint8_t)(bSum / count);
            }
        }
    }

    return current;
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



    // ── 4. Apply DCT residual ─────────────────────────────────────────────────
    // Convert LabF array to interleaved float array that applyResidual expects
    std::vector<float> labFlat(data.width * data.height * 3);
    for (int i = 0; i < data.width * data.height; i++) {
        labFlat[i*3+0] = labImage[i].L;
        labFlat[i*3+1] = labImage[i].a;
        labFlat[i*3+2] = labImage[i].b;
    }

    applyResidual(labFlat, data.residual, data.width, data.height);

    // Write back into LabF image after residual correction
    for (int i = 0; i < data.width * data.height; i++) {
        labImage[i] = {labFlat[i*3+0], labFlat[i*3+1], labFlat[i*3+2]};
    }



    // ── Convert Lab → RGB ─────────────────────────────────────────────────────
    std::vector<uint8_t> pixels;
    pixels.reserve(data.width * data.height * 3);

    bool addNoise = false;
    int noiseMode = 2;
    float gaussBias = 0.6f;
    uint32_t seed = 1234;
    if (argc > 2) seed = (uint32_t)std::stoul(argv[2]);
    std::mt19937 rng(seed);


    for (int i = 0; i < data.width * data.height; i++) {
        const PaletteEntry& p = data.palette[data.indexMatrix[i]];
        LabF gradColor = labImage[i];

        // Always assign finalColor — noise is optional, RGB conversion is not
        LabF finalColor = gradColor;

        if (addNoise) {
            switch (noiseMode) {
                case 1:  finalColor = sampleUniform(p, gradColor, rng); break;
                case 2:
                default: finalColor = sampleGaussian(p, gradColor, rng, gaussBias); break;
            }
        }

        RGB rgb = labToRGB(finalColor.L, finalColor.a, finalColor.b);
        pixels.push_back(rgb.r);
        pixels.push_back(rgb.g);
        pixels.push_back(rgb.b);
    }


    // ── Region-confined blur ──────────────────────────────────────────────────
    int blurRadius = 1;  // tune: 1 = subtle, 2 = noticeable, 3 = strong
    pixels = blurWithinRegions(pixels, data.indexMatrix, data.width, data.height, blurRadius);


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