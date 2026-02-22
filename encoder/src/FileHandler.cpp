#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>

class BitWriter {
    std::ofstream& out;
    uint8_t buffer = 0;
    int bitCount = 0;

public:
    BitWriter(std::ofstream& f) : out(f) {}

    // Writes 'value' using only 'numBits'
    void write(uint32_t value, int numBits) {
        for (int i = numBits - 1; i >= 0; --i) {
            bool bit = (value >> i) & 1;
            buffer = (buffer << 1) | bit;
            bitCount++;
            if (bitCount == 8) {
                out.put(buffer);
                buffer = 0;
                bitCount = 0;
            }
        }
    }

    // Fills the last byte with zeros if needed
    void flush() {
        if (bitCount > 0) {
            buffer <<= (8 - bitCount);
            out.put(buffer);
        }
    }
};

struct PaletteEntry { float L, a, b, error; };
struct CompactEntry { int8_t L, a, b, e; };

    