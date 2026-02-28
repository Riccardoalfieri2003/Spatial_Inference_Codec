#pragma once
#ifndef IMAGE_CONVERTER_HPP
#define IMAGE_CONVERTER_HPP

#include <vector>
#include "PaletteEntry.hpp"

struct LabPixel {
    float L; // Lightness
    float a; // Green-Red component
    float b; // Blue-Yellow component
};

class ImageConverter {
public:
    static std::vector<LabPixel> RGBtoLab(unsigned char* rgbData, int width, int height, int channels);
    static LabPixel convertPixelRGBtoLab(unsigned char r_raw, unsigned char g_raw, unsigned char b_raw);
    static void convertPixelLabToRGB(float L, float a, float b, unsigned char& r, unsigned char& g, unsigned char& bl);
};

#endif