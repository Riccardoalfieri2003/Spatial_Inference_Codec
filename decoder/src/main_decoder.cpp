#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sif_decoder.hpp"

// stb_image_write: single header library, no installation needed
// Download from: https://github.com/nothings/stb/blob/master/stb_image_write.h
// and place it in decoder/include/
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// ── Lab → XYZ → RGB conversion ───────────────────────────────────────────────
// Reference white point D65 (standard daylight illuminant)
static const float REF_X = 95.047f;
static const float REF_Y = 100.000f;
static const float REF_Z = 108.883f;

// Helper: inverse of the f() function used in Lab encoding
static float labInvF(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta)
        return t * t * t;
    else
        return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

// Helper: linear RGB → gamma-corrected sRGB (0.0 to 1.0)
static float linearToSRGB(float val) {
    val = std::clamp(val, 0.0f, 1.0f);
    if (val <= 0.0031308f)
        return 12.92f * val;
    else
        return 1.055f * std::pow(val, 1.0f / 2.4f) - 0.055f;
}

struct RGB { uint8_t r, g, b; };

// Full pipeline: Lab → XYZ → linear RGB → sRGB
static RGB labToRGB(float L, float a, float b) {

    // Step 1: Lab → XYZ
    float fy = (L + 16.0f) / 116.0f;
    float fx = (a / 500.0f) + fy;
    float fz = fy - (b / 200.0f);

    float X = REF_X * labInvF(fx);
    float Y = REF_Y * labInvF(fy);
    float Z = REF_Z * labInvF(fz);

    // Normalize to [0, 1]
    X /= 100.0f;
    Y /= 100.0f;
    Z /= 100.0f;

    // Step 2: XYZ → linear RGB (sRGB matrix, D65)
    float r_lin =  3.2406f * X - 1.5372f * Y - 0.4986f * Z;
    float g_lin = -0.9689f * X + 1.8758f * Y + 0.0415f * Z;
    float b_lin =  0.0557f * X - 0.2040f * Y + 1.0570f * Z;

    // Step 3: linear → gamma corrected sRGB, then scale to [0, 255]
    return RGB {
        (uint8_t)std::round(linearToSRGB(r_lin) * 255.0f),
        (uint8_t)std::round(linearToSRGB(g_lin) * 255.0f),
        (uint8_t)std::round(linearToSRGB(b_lin) * 255.0f)
    };
}


int main(int argc, char* argv[]) {

    // ── Get file path from command line or use a default ────────────────────
    std::string filePath = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif";
    if (argc > 1) filePath = argv[1];

    std::cout << "Loading: " << filePath << "\n";

    // ── Decode the SIF file ──────────────────────────────────────────────────
    SIFData data = loadSIF(filePath);

    if (!data.valid) {
        std::cerr << "Failed to decode file.\n";
        return 1;
    }

    // ── Build RGB pixel buffer ───────────────────────────────────────────────
    // stb_image_write expects a flat array of: R G B R G B R G B ...
    std::vector<uint8_t> pixels;
    pixels.reserve(data.width * data.height * 3);

    for (int y = 0; y < data.height; y++) {
        for (int x = 0; x < data.width; x++) {
            int idx = data.indexMatrix[y * data.width + x];
            const PaletteEntry& p = data.palette[idx];

            RGB rgb = labToRGB(p.L, p.a, p.b);
            pixels.push_back(rgb.r);
            pixels.push_back(rgb.g);
            pixels.push_back(rgb.b);
        }
    }

    // ── Save as PNG ──────────────────────────────────────────────────────────
    std::string outPath = filePath + "_reconstructed.png";

    int success = stbi_write_png(
        outPath.c_str(),
        data.width,
        data.height,
        3,              // 3 channels: RGB
        pixels.data(),
        data.width * 3  // row stride in bytes
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
