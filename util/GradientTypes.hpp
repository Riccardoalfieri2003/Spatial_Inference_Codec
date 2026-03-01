#pragma once
#include <vector>
#include <cstdint>

enum class GradientPrecision : uint8_t { BITS_2 = 2, BITS_4 = 4, BITS_6 = 6 };


struct GradientDescriptor {
    uint8_t shape;      // 0=sharp, 1=linear, 2=ease-in/out, 3=S-curve
    uint8_t direction;  // 0=horizontal, 1=vertical, 2=diag-left, 3=diag-right
    uint8_t width;      // 0=1px, 1=3px, 2=6px, 3=12px

    uint8_t pack(GradientPrecision prec) const {
        switch (prec) {
            case GradientPrecision::BITS_2:
                // 2 bits: store width only (most impactful field)
                return width & 0x03;
            case GradientPrecision::BITS_4:
                // 4 bits: store width (2 bits) + shape (2 bits)
                return ((width & 0x03) << 2) | (shape & 0x03);
            case GradientPrecision::BITS_6:
            default:
                // 6 bits: store all three fields
                return ((shape & 0x03) << 4) | ((direction & 0x03) << 2) | (width & 0x03);
        }
    }

    static GradientDescriptor unpack(uint8_t bits, GradientPrecision prec) {
        GradientDescriptor d{0, 0, 0};
        switch (prec) {
            case GradientPrecision::BITS_2:
                // width recovered, shape and direction hardcoded
                d.width     = bits & 0x03;
                d.shape     = 1;  // linear
                d.direction = 0;  // horizontal
                break;
            case GradientPrecision::BITS_4:
                // width and shape recovered, direction hardcoded
                d.width     = (bits >> 2) & 0x03;
                d.shape     =  bits       & 0x03;
                d.direction = 0;  // horizontal
                break;
            case GradientPrecision::BITS_6:
            default:
                d.shape     = (bits >> 4) & 0x03;
                d.direction = (bits >> 2) & 0x03;
                d.width     =  bits       & 0x03;
                break;
        }
        return d;
    }
};


struct ChangePoint {
    uint16_t x, y;
    uint32_t queueIdx;
};

struct GradientData {
    GradientPrecision precision;
    std::vector<GradientDescriptor> queue;
    std::vector<ChangePoint> changePoints;
    bool valid = true;
};

