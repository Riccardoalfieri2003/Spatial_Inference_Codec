#ifndef VOXEL_GRID_HPP
#define VOXEL_GRID_HPP

#include <unordered_map>
#include <vector>
#include "ImageConverter.hpp"
#include <map>

// The Key: Represents the 3D grid position
struct VoxelCoord {
    int l_idx, a_idx, b_idx;

    bool operator==(const VoxelCoord& other) const {
        return l_idx == other.l_idx && a_idx == other.a_idx && b_idx == other.b_idx;
    }
};

// Custom Hash Function for VoxelCoord (Required for unordered_map)
struct VoxelHash {
    std::size_t operator()(const VoxelCoord& v) const {
        // Using prime numbers to minimize collisions (standard spatial hashing)
        return ((std::hash<int>()(v.l_idx) ^ 
               (std::hash<int>()(v.a_idx) << 1)) >> 1) ^ 
               (std::hash<int>()(v.b_idx) << 1);
    }
};

// We need a custom comparator for LabPixel to use it as a key in a std::map
struct LabComparator {
    bool operator()(const LabPixel& a, const LabPixel& b) const {
        if (a.L != b.L) return a.L < b.L;
        if (a.a != b.a) return a.a < b.a;
        return a.b < b.b;
    }
};

struct VoxelData {
    // Key: The exact Lab color | Value: How many pixels had this color
    std::map<LabPixel, int, LabComparator> colorFrequencies;
    int totalPixelCount = 0;

    void addPixel(const LabPixel& pixel) {
        colorFrequencies[pixel]++;
        totalPixelCount++;
    }
};
typedef std::unordered_map<VoxelCoord, VoxelData, VoxelHash> VoxelMap;

#endif