#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include "sif_decoder_mod.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <random>
#include <ImageConverter_mod.hpp>
#include "PaletteEntry.hpp"


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


// ── Check if a pixel is within `borderDist` pixels of a cluster boundary ──────
static bool nearBoundary(int i, int width, int height,
                          const std::vector<int>& indexMatrix,
                          int borderDist = 3) {
    int x = i % width;
    int y = i / width;
    int c = indexMatrix[i];

    for (int dy = -borderDist; dy <= borderDist; dy++) {
        for (int dx = -borderDist; dx <= borderDist; dx++) {
            if (std::abs(dx) + std::abs(dy) > borderDist) continue; // diamond shape
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (indexMatrix[ny * width + nx] != c) return true;
        }
    }
    return false;
}


// ── Uniform sampler with border-aware bias ────────────────────────────────────
//
// Interior pixels: t sampled uniformly in [0, 1]
//                  → free to roam between centroid and gradient color
//
// Border pixels:   t sampled from a distribution heavily biased toward 1.0
//                  → stays close to the gradient color, which already
//                     encodes the correct boundary transition
//
// borderBias: how strongly to push toward gradient color at borders
//             0.7 = moderate, 0.9 = very close to gradient color
//
static LabF sampleUniform(const PaletteEntry& p,
                           const LabF& gradColor,
                           std::mt19937& rng,
                           bool isNearBorder,
                           float borderBias = 0.85f) {
    float radius = std::abs(p.error);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    float t;
    if (isNearBorder) {
        // Sample t in [borderBias, 1.0] — close to gradient color
        t = borderBias + uni(rng) * (1.0f - borderBias);
    } else {
        // Sample t freely in [0, 1]
        t = uni(rng);
    }

    float L = p.L + t * (gradColor.L - p.L);
    float a = p.a + t * (gradColor.a - p.a);
    float b = p.b + t * (gradColor.b - p.b);

    return clampToSphere(p.L, p.a, p.b, L, a, b, radius);
}


static LabF sampleUniform_pre(const PaletteEntry& p,   // cluster centroid + error
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












// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {

    std::string filePath = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif";
    if (argc > 1) filePath = argv[1];

    std::cout << "Loading: " << filePath << "\n";

    SIFData data = loadSIF(filePath);
    if (!data.valid) {
        std::cerr << "Failed to decode SIF file.\n";
        return 1;
    }

    /*
    std::cout << "gradients.valid=" << data.gradients.valid 
          << " queue.size=" << data.gradients.queue.size()
          << " changePoints.size=" << data.gradients.changePoints.size() << "\n";
    std::cout << "residualPalette.size=" << data.residualPalette.size() << "\n";
    std::cout << "residualIndexMatrix.size=" << data.residualIndexMatrix.size() << "\n";
    */


    for (int i = 0; i < 5; i++)
    std::cout << "DEC palette[" << i << "] L=" << data.palette[i].L 
              << " a=" << data.palette[i].a << " b=" << data.palette[i].b << "\n";
    std::cout << "DEC indexMatrix[0]=" << data.indexMatrix[0] << "\n";

    int totalPixels = data.width * data.height;

    // ── Step 1: Quantized + main gradients ────────────────────────────────────
    std::vector<LabF> labImage(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
        const PaletteEntry& p = data.palette[data.indexMatrix[i]];
        labImage[i] = {p.L, p.a, p.b};
    }

    // DEBUG: save raw quantized image before any gradient application
    {
        std::vector<unsigned char> buf(totalPixels * 3);
        for (int i = 0; i < totalPixels; i++) {
            ImageConverter::convertPixelLabToRGB(
                labImage[i].L, labImage[i].a, labImage[i].b,
                buf[i*3], buf[i*3+1], buf[i*3+2]);
        }
        stbi_write_png("debug_raw_quantized_NO_GRAD_decoder.png", data.width, data.height, 3, buf.data(), data.width * 3);
    }

    std::cout << "labImage[0] before grad: L=" << labImage[0].L 
          << " a=" << labImage[0].a << " b=" << labImage[0].b << "\n";
    std::cout << "palette[0]: L=" << data.palette[0].L 
            << " a=" << data.palette[0].a << " b=" << data.palette[0].b << "\n";
    std::cout << "indexMatrix[0]=" << data.indexMatrix[0] << "\n";

    applyGradients(labImage, data.indexMatrix, data.palette,
                   data.gradients, data.width, data.height);


    // First 5 pixels after applyGradients
    for (int i = 0; i < 5; i++)
        std::cout << "POST_GRAD pixel[" << i << "] L=" << labImage[i].L  // or reconstructed[i]
                << " a=" << labImage[i].a << " b=" << labImage[i].b << "\n";

    // Also print the change points and queue first entry
    std::cout << "gradient queue[0]: shape=" << (int)data.gradients.queue[0].shape  // or data.gradients
            << " dir=" << (int)data.gradients.queue[0].direction
            << " width=" << (int)data.gradients.queue[0].width << "\n";

    std::cout << "indexMatrix around pixel 0: ";
    for (int i = 0; i < 10; i++)
        std::cout << data.indexMatrix[i] << " ";  // or data.indexMatrix[i]
    std::cout << "\n";


    // Add this after applyGradients in BOTH encoder and decoder
    for (int i = 0; i < 5; i++) {
        std::cout << "pixel[" << i << "] L=" << labImage[i].L  // or reconstructed[i] in encoder
                << " a=" << labImage[i].a
                << " b=" << labImage[i].b << "\n";
    }

    // ── Step 2: Residual 1 + gradients ────────────────────────────────────────
    std::vector<LabF> residualImage(totalPixels);
    if (!data.residualPalette.empty()) {
        for (int i = 0; i < totalPixels; i++) {
            const PaletteEntry& p = data.residualPalette[data.residualIndexMatrix[i]];
            residualImage[i] = {p.L, p.a, p.b};
        }
        //applyGradients(residualImage, data.residualIndexMatrix, data.residualPalette, data.residualGradients, data.width, data.height);
    }

    // ── Step 3: Residual 2 + gradients ────────────────────────────────────────
    std::vector<LabF> residualImage2(totalPixels, {0.0f, 0.0f, 0.0f});
    if (!data.residualPalette2.empty()) {
        for (int i = 0; i < totalPixels; i++) {
            const PaletteEntry& p = data.residualPalette2[data.residualIndexMatrix2[i]];
            residualImage2[i] = {p.L, p.a, p.b};
        }
        //applyGradients(residualImage2, data.residualIndexMatrix2, data.residualPalette2, data.residualGradients2, data.width, data.height);
    }

    // ── Step 4: Full reconstruction ───────────────────────────────────────────
    std::vector<LabF> finalImage(totalPixels);
    for (int i = 0; i < totalPixels; i++) {
        finalImage[i] = {
            labImage[i].L + residualImage[i].L + residualImage2[i].L,
            labImage[i].a + residualImage[i].a + residualImage2[i].a,
            labImage[i].b + residualImage[i].b + residualImage2[i].b
        };
    }

    // ── Step 5: Debug visualization ───────────────────────────────────────────
    auto saveAndOpen = [&](const std::vector<LabF>& img, const std::string& name, float shiftL = 0.0f) {
        std::vector<unsigned char> buf(totalPixels * 3);
        for (int i = 0; i < totalPixels; i++) {
            ImageConverter::convertPixelLabToRGB(
                img[i].L + shiftL, img[i].a, img[i].b,
                buf[i*3], buf[i*3+1], buf[i*3+2]);
        }
        stbi_write_png(name.c_str(), data.width, data.height, 3, buf.data(), data.width * 3);
        //system(("start " + name).c_str());
    };

    saveAndOpen(labImage,       "debug_quantized_grad.png");
    saveAndOpen(residualImage,  "debug_residual1_grad.png", 50.0f);
    saveAndOpen(residualImage2, "debug_residual2_grad.png", 50.0f);
    saveAndOpen(finalImage,     "debug_full_reconstruction.png");

    /*
    // ── Step 6: Sphere-constrained reconstruction ─────────────────────────────
    std::vector<LabF> sampledImage(totalPixels);
    bool random = false;

    if (random) {
        std::mt19937 rng(42);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        for (int i = 0; i < totalPixels; i++) {
            LabF  Q  = labImage[i];
            LabF  QR = finalImage[i];
            float e1 = data.palette[data.indexMatrix[i]].error;
            float e2 = data.residualPalette2.empty()
                     ? data.residualPalette[data.residualIndexMatrix[i]].error
                     : data.residualPalette2[data.residualIndexMatrix2[i]].error;

            if (e2 < 1e-4f) { sampledImage[i] = QR; continue; }

            const int maxAttempts = 32;
            bool found = false;
            LabF candidate;
            for (int attempt = 0; attempt < maxAttempts; attempt++) {
                float dL = gauss(rng), da = gauss(rng), db = gauss(rng);
                float norm = std::sqrt(dL*dL + da*da + db*db);
                if (norm < 1e-6f) continue;
                candidate = { QR.L + e2*dL/norm, QR.a + e2*da/norm, QR.b + e2*db/norm };
                float dist = std::sqrt(
                    (candidate.L-Q.L)*(candidate.L-Q.L) +
                    (candidate.a-Q.a)*(candidate.a-Q.a) +
                    (candidate.b-Q.b)*(candidate.b-Q.b));
                if (dist <= e1) { found = true; break; }
            }
            sampledImage[i] = found ? candidate : QR;
        }
    } else {
        for (int i = 0; i < totalPixels; i++) {
            LabF  Q   = labImage[i];     // center of Sphere 1 (quantized + grad)
            LabF  QR  = finalImage[i];   // center of Sphere 3 (Q + res1 + res2)
            
            float e1  = data.palette[data.indexMatrix[i]].error;
            float e2  = data.residualPalette.empty()  ? 0.0f 
                    : data.residualPalette[data.residualIndexMatrix[i]].error;
            float e3  = data.residualPalette2.empty() ? 0.0f 
                    : data.residualPalette2[data.residualIndexMatrix2[i]].error;

            // ── Sphere 1: center=Q, radius=e1 ────────────────────────────────────
            // ── Sphere 2: center=Q+res1+grad,  radius=e2 ─────────────────────────
            // ── Sphere 3: center=Q+res1+res2+grad, radius=e3 ─────────────────────

            // Step A: find the closest point on surface of Sphere 2 toward Q
            // This gives us the best point on Sphere 2 that agrees with Sphere 1
            LabF center2 = {
                Q.L + residualImage[i].L,
                Q.a + residualImage[i].a,
                Q.b + residualImage[i].b
            };

            LabF p2;  // best point on surface of Sphere 2
            if (e2 < 1e-4f) {
                p2 = center2;
            } else {
                float dL = Q.L - center2.L;
                float da = Q.a - center2.a;
                float db = Q.b - center2.b;
                float norm = std::sqrt(dL*dL + da*da + db*db);
                if (norm < 1e-6f) {
                    p2 = center2;
                } else {
                    p2 = {
                        center2.L + e2 * dL / norm,
                        center2.a + e2 * da / norm,
                        center2.b + e2 * db / norm
                    };
                }
            }

            // Check p2 is inside Sphere 1 — if not, clamp to Sphere 1 surface
            {
                float dL = p2.L - Q.L;
                float da = p2.a - Q.a;
                float db = p2.b - Q.b;
                float dist = std::sqrt(dL*dL + da*da + db*db);
                if (dist > e1 && dist > 1e-6f) {
                    p2 = {
                        Q.L + e1 * dL / dist,
                        Q.a + e1 * da / dist,
                        Q.b + e1 * db / dist
                    };
                }
            }

            // Step B: find the closest point on surface of Sphere 3 toward p2
            // p2 is now our target — the best constrained estimate from Sphere 2
            if (e3 < 1e-4f) {
                sampledImage[i] = QR;
                continue;
            }

            float dL = p2.L - QR.L;
            float da = p2.a - QR.a;
            float db = p2.b - QR.b;
            float norm = std::sqrt(dL*dL + da*da + db*db);

            if (norm < 1e-6f) {
                sampledImage[i] = QR;
                continue;
            }

            LabF p3 = {
                QR.L + e3 * dL / norm,
                QR.a + e3 * da / norm,
                QR.b + e3 * db / norm
            };

            // Final check: if p3 falls outside Sphere 1, fall back to p2
            {
                float dL2 = p3.L - Q.L;
                float da2 = p3.a - Q.a;
                float db2 = p3.b - Q.b;
                float dist = std::sqrt(dL2*dL2 + da2*da2 + db2*db2);
                sampledImage[i] = (dist <= e1) ? p3 : p2;
            }
        }
    }

    // ── Step 7: Convert Lab → RGB and save ────────────────────────────────────
    std::vector<unsigned char> pixels(totalPixels * 3);
    for (int i = 0; i < totalPixels; i++) {
        ImageConverter::convertPixelLabToRGB(
            sampledImage[i].L, sampledImage[i].a, sampledImage[i].b,
            pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
    }
    */

    // ── Step 7: Convert Lab → RGB and save ───────────────────────────────────
    std::vector<unsigned char> pixels(totalPixels * 3);
    for (int i = 0; i < totalPixels; i++) {
        ImageConverter::convertPixelLabToRGB(
            finalImage[i].L, finalImage[i].a, finalImage[i].b,
            pixels[i*3], pixels[i*3+1], pixels[i*3+2]);
    }



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