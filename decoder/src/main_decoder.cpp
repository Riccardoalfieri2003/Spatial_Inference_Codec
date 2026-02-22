#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "sif_decoder.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// ── Lab → XYZ → RGB conversion ───────────────────────────────────────────────
static const float REF_X = 95.047f;
static const float REF_Y = 100.000f;
static const float REF_Z = 108.883f;

static float labInvF(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta)
        return t * t * t;
    else
        return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

static float linearToSRGB(float val) {
    val = std::clamp(val, 0.0f, 1.0f);
    if (val <= 0.0031308f)
        return 12.92f * val;
    else
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


// ── Gaussian sampler inside the Lab error sphere ──────────────────────────────
//
// For each pixel we want to pick a random Lab color inside a sphere of
// radius = p.error, centered on (p.L, p.a, p.b).
//
// Strategy:
//   1. Sample a 3D offset (dL, da, db) from a Gaussian with sigma = error/3.
//      Using sigma = error/3 means ~99.7% of samples fall within the sphere.
//   2. If the sampled point falls outside the sphere (rare tail), clamp it
//      back onto the surface so we never exceed the guaranteed error radius.
//   3. Apply the offset to the palette color.
//
static RGB sampleLabSphere(const PaletteEntry& p, std::mt19937& rng) {

    float radius = std::abs(p.error);

    // Degenerate case: no error budget, just return the exact palette color
    if (radius < 1e-4f)
        return labToRGB(p.L, p.a, p.b);

    // Gaussian with sigma = radius/3 so that tails naturally touch the border
    float sigma = radius / 3.0f;
    std::normal_distribution<float> gauss(0.0f, sigma);

    float dL = gauss(rng);
    float da = gauss(rng);
    float db = gauss(rng);

    // Clamp to sphere surface if we happened to land outside
    float dist = std::sqrt(dL*dL + da*da + db*db);
    if (dist > radius) {
        float scale = radius / dist;
        dL *= scale;
        da *= scale;
        db *= scale;
    }

    return labToRGB(p.L + dL, p.a + da, p.b + db);
}

static std::vector<uint8_t> gaussianBlur(std::vector<uint8_t> pixels, int width, int height, int radius) {

    for (int pass = 0; pass < 3; pass++) {
        std::vector<uint8_t> temp(pixels.size());

        // Horizontal pass
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rSum = 0, gSum = 0, bSum = 0, count = 0;
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    int base = (y * width + nx) * 3;
                    rSum += pixels[base + 0];
                    gSum += pixels[base + 1];
                    bSum += pixels[base + 2];
                    count++;
                }
                int base = (y * width + x) * 3;
                temp[base + 0] = (uint8_t)(rSum / count);
                temp[base + 1] = (uint8_t)(gSum / count);
                temp[base + 2] = (uint8_t)(bSum / count);
            }
        }

        // Vertical pass
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rSum = 0, gSum = 0, bSum = 0, count = 0;
                for (int dy = -radius; dy <= radius; dy++) {
                    int ny = std::clamp(y + dy, 0, height - 1);
                    int base = (ny * width + x) * 3;
                    rSum += temp[base + 0];
                    gSum += temp[base + 1];
                    bSum += temp[base + 2];
                    count++;
                }
                int base = (y * width + x) * 3;
                pixels[base + 0] = (uint8_t)(rSum / count);
                pixels[base + 1] = (uint8_t)(gSum / count);
                pixels[base + 2] = (uint8_t)(bSum / count);
            }
        }
    }

    return pixels;
}

int main(int argc, char* argv[]) {

    // ── File path ────────────────────────────────────────────────────────────
    std::string filePath = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif";
    if (argc > 1) filePath = argv[1];

    // ── Optional: custom seed from command line (./decoder file.sif 42) ──────
    uint32_t seed = 1234;  // default seed → always reproducible
    if (argc > 2) seed = (uint32_t)std::stoul(argv[2]);

    std::cout << "Loading: " << filePath << "\n";
    std::cout << "Random seed: " << seed << "\n";

    // ── Decode ───────────────────────────────────────────────────────────────
    SIFData data = loadSIF(filePath);
    if (!data.valid) {
        std::cerr << "Failed to decode file.\n";
        return 1;
    }

    // ── RNG — seeded for full reproducibility ────────────────────────────────
    std::mt19937 rng(seed);

    // ── Build RGB pixel buffer with Gaussian sphere sampling ─────────────────
    std::vector<uint8_t> pixels;
    pixels.reserve(data.width * data.height * 3);

    for (int y = 0; y < data.height; y++) {
        for (int x = 0; x < data.width; x++) {
            int idx = data.indexMatrix[y * data.width + x];
            const PaletteEntry& p = data.palette[idx];

            RGB rgb = sampleLabSphere(p, rng);
            pixels.push_back(rgb.r);
            pixels.push_back(rgb.g);
            pixels.push_back(rgb.b);
        }
    }

    // ── Optional blur for debugging ──────────────────────────────────────────
    //int blurRadius = 1;  // change this: 1 = subtle, 2 = medium, 4 = strong
    //pixels = gaussianBlur(pixels, data.width, data.height, blurRadius);


    // ── Save as PNG ──────────────────────────────────────────────────────────
    std::string outPath = filePath + "_reconstructed.png";

    int success = stbi_write_png(
        outPath.c_str(),
        data.width,
        data.height,
        3,
        pixels.data(),
        data.width * 3
    );

    if (success) {
        std::cout << "\n[SUCCESS] Reconstructed image saved to:\n";
        std::cout << "  " << outPath << "\n";
    } else {
        std::cerr << "Failed to write PNG.\n";
        return 1;
    }

    return 0;
}