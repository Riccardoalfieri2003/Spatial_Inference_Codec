#include <iostream>
#include <string>
#include "sif_decoder.hpp"

int main(int argc, char* argv[]) {

    // ── Get file path from command line or use a default ────────────────────
    std::string filePath = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif";
    if (argc > 1) {
        filePath = argv[1];
    }

    std::cout << "Loading: " << filePath << "\n";

    // ── Decode ───────────────────────────────────────────────────────────────
    SIFData data = loadSIF(filePath);

    if (!data.valid) {
        std::cerr << "Failed to decode file.\n";
        return 1;
    }

    // ── At this point you have: ──────────────────────────────────────────────
    //   data.width         → image width
    //   data.height        → image height
    //   data.palette       → vector of PaletteEntry { L, a, b, error }
    //   data.indexMatrix   → one palette index per pixel, row-major order

    // ── Example: print the first 10 pixel colors ────────────────────────────
    std::cout << "\nFirst 10 pixels (Lab values):\n";
    int limit = std::min(10, (int)data.indexMatrix.size());
    for (int i = 0; i < limit; i++) {
        int idx = data.indexMatrix[i];
        const PaletteEntry& p = data.palette[idx];
        std::cout << "  Pixel " << i
                  << " → palette[" << idx << "]"
                  << "  L=" << p.L
                  << "  a=" << p.a
                  << "  b=" << p.b
                  << "  err=" << p.error
                  << "\n";
    }

    // ── TODO: add your reconstruction logic here ─────────────────────────────
    // e.g. convert Lab → RGB, write to PNG, display, etc.

    return 0;
}
