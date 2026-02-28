#include "ImageConverter_mod.hpp"
#include <cmath>

std::vector<LabPixel> ImageConverter::RGBtoLab(unsigned char* rgbData, int width, int height, int channels) {
    std::vector<LabPixel> labData;
    labData.reserve(width * height);

    for (int i = 0; i < width * height * channels; i += channels) {
        // 1. Normalize RGB to [0, 1]
        float r = rgbData[i] / 255.0f;
        float g = rgbData[i + 1] / 255.0f;
        float b = rgbData[i + 2] / 255.0f;

        // 2. Pivot to XYZ (Simplified sRGB conversion)
        auto pivot = [](float n) {
            return (n > 0.04045f) ? std::pow((n + 0.055f) / 1.055f, 2.4f) : (n / 12.92f);
        };
        r = pivot(r); g = pivot(g); b = pivot(b);

        float x = r * 0.4124f + g * 0.3576f + b * 0.1805f;
        float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        float z = r * 0.0193f + g * 0.1192f + b * 0.9505f;

        // 3. XYZ to Lab (D65 White Point)
        x /= 0.95047f; y /= 1.00000f; z /= 1.08883f;

        auto f = [](float t) {
            return (t > 0.008856f) ? std::pow(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
        };

        float L = 116.0f * f(y) - 16.0f;
        float a = 500.0f * (f(x) - f(y));
        float b_val = 200.0f * (f(y) - f(z));

        labData.push_back({L, a, b_val});
    }
    return labData;
}


LabPixel ImageConverter::convertPixelRGBtoLab(unsigned char r_raw, unsigned char g_raw, unsigned char b_raw) {
    // 1. Normalize
    float r = r_raw / 255.0f;
    float g = g_raw / 255.0f;
    float b = b_raw / 255.0f;

    // 2. RGB -> XYZ
    auto pivot = [](float n) {
        return (n > 0.04045f) ? std::pow((n + 0.055f) / 1.055f, 2.4f) : (n / 12.92f);
    };
    r = pivot(r); g = pivot(g); b = pivot(b);

    float x = (r * 0.4124f + g * 0.3576f + b * 0.1805f) / 0.95047f;
    float y = (r * 0.2126f + g * 0.7152f + b * 0.0722f) / 1.00000f;
    float z = (r * 0.0193f + g * 0.1192f + b * 0.9505f) / 1.08883f;

    // 3. XYZ -> Lab
    auto f = [](float t) {
        return (t > 0.008856f) ? std::pow(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
    };

    float fx = f(x);
    float fy = f(y);
    float fz = f(z);

    return {
        116.0f * fy - 16.0f,        // L
        500.0f * (fx - fy),         // a
        200.0f * (fy - fz)          // b
    };
}

void ImageConverter::convertPixelLabToRGB(float L, float a, float b, unsigned char& r, unsigned char& g, unsigned char& bl) {
    // Lab → XYZ
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b / 200.0f;

    auto f_inv = [](float t) {
        return t > 0.206897f ? t * t * t : (t - 16.0f / 116.0f) / 7.787f;
    };

    float X = f_inv(fx) * 0.95047f;
    float Y = f_inv(fy) * 1.00000f;
    float Z = f_inv(fz) * 1.08883f;

    // XYZ → linear RGB
    float rf =  3.2406f*X - 1.5372f*Y - 0.4986f*Z;
    float gf = -0.9689f*X + 1.8758f*Y + 0.0415f*Z;
    float bf =  0.0557f*X - 0.2040f*Y + 1.0570f*Z;

    // Gamma correction + clamp
    auto gammaAndClamp = [](float c) -> unsigned char {
        c = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f/2.4f) - 0.055f;
        c = std::max(0.0f, std::min(1.0f, c));
        return static_cast<unsigned char>(c * 255.0f + 0.5f);
    };

    r  = gammaAndClamp(rf);
    g  = gammaAndClamp(gf);
    bl = gammaAndClamp(bf);
}