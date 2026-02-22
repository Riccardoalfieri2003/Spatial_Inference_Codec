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

void saveSIF(const std::string& path, int width, int height, 
             const std::vector<PaletteEntry>& palette, 
             const std::vector<int>& indexMatrix) {
    
    std::ofstream file(path, std::ios::binary);

    // 1. Process Palette with your rounding rules
    std::vector<CompactEntry> compactPalette;
    for (const auto& p : palette) {
        compactPalette.push_back({
            (int8_t)std::round(p.L),     // Nearest integer
            (int8_t)std::round(p.a),     // Nearest integer
            (int8_t)std::round(p.b),     // Nearest integer
            (int8_t)std::floor(p.error)  // Floor as requested
        });
    }

    // 2. Write Header
    uint32_t w = width, h = height;
    uint16_t palSize = (uint16_t)compactPalette.size();
    file.write((char*)&w, 4);
    file.write((char*)&h, 4);
    file.write((char*)&palSize, 2);

    // 3. Write Palette Data
    file.write((char*)compactPalette.data(), compactPalette.size() * 4);

    // 4. Calculate bit-depth
    // If you have 256 colors, bitsNeeded = 8. If 100 colors, bitsNeeded = 7.
    int bitsNeeded = (palSize <= 1) ? 1 : (int)std::ceil(std::log2(palSize));
    file.put((unsigned char)bitsNeeded);

    // 5. Delta Encoding + RLE Pipeline
    // We store the Delta, and then how many times that Delta repeats.
    BitWriter bw(file);
    
    int lastIdx = 0;
    for (size_t i = 0; i < indexMatrix.size(); ) {
        int currentDelta = indexMatrix[i] - lastIdx;
        
        // Count how many times this specific Delta repeats (RLE)
        uint32_t runLength = 1;
        while (i + runLength < indexMatrix.size() && 
               (indexMatrix[i + runLength] - indexMatrix[i + runLength - 1]) == currentDelta && 
               runLength < 255) { // Cap run at 255 for simplicity
            runLength++;
        }

        // Write the Delta value (using bitsNeeded)
        // Note: Delta can be negative, so we add an offset to keep it positive for bit-writing
        // Offset is palSize, making the range [0, 2*palSize]
        bw.write(currentDelta + palSize, bitsNeeded + 1); 
        
        // Write the Run Length (fixed 8 bits for 1-255)
        bw.write(runLength, 8);

        i += runLength;
        lastIdx = indexMatrix[i-1];
    }

    bw.flush();
    file.close();
    std::cout << "Successfully saved: " << path << " using " << bitsNeeded << " bits per index." << std::endl;
}