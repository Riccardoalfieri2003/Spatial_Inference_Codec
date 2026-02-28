#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "VoxelGrid_mod.hpp"
#include <vector>


struct Cluster {
    // Total aggregated frequency table for this cluster
    std::map<LabPixel, int, LabComparator> colorFrequencies;
    int totalPixels = 0;

    void addVoxel(const VoxelData& voxel) {
        for (auto const& [color, count] : voxel.colorFrequencies) {
            colorFrequencies[color] += count;
        }
        totalPixels += voxel.totalPixelCount;
    }

    struct ClusterStats {
        LabPixel centroid;
        float maxError;
    };

    ClusterStats getStats() const {
        if (totalPixels == 0) return {{0, 0, 0}, 0.0f};

        // 1. Calculate Weighted Mean (Centroid)
        double sumL = 0, sumA = 0, sumB = 0;
        for (auto const& [color, count] : colorFrequencies) {
            sumL += (double)color.L * count;
            sumA += (double)color.a * count;
            sumB += (double)color.b * count;
        }

        LabPixel centroid = {
            static_cast<float>(sumL / totalPixels),
            static_cast<float>(sumA / totalPixels),
            static_cast<float>(sumB / totalPixels)
        };

        // 2. Calculate Maximum Error (Distance to farthest color)
        float maxDistSq = 0.0f;
        for (auto const& [color, count] : colorFrequencies) {
            float dL = color.L - centroid.L;
            float da = color.a - centroid.a;
            float db = color.b - centroid.b;
            
            float currentDistSq = dL*dL + da*da + db*db;
            if (currentDistSq > maxDistSq) {
                maxDistSq = currentDistSq;
            }
        }

        return {centroid, std::sqrt(maxDistSq)};
    }
};

#include <unordered_set>
#include <queue>
#include <vector>

// A simple wrapper to return both the clusters and the mapping
struct ClusteringResult {
    std::vector<Cluster> clusters;
    std::unordered_map<VoxelCoord, int, VoxelHash> voxelToClusterIdx;
};

class Clusterer {
public:
    static ClusteringResult run(VoxelMap& grid, int maxSteps) {
        ClusteringResult result;
        std::unordered_set<VoxelCoord, VoxelHash> visited;
        int currentClusterIdx = 0;

        for (auto const& [startCoord, data] : grid) {
            if (visited.count(startCoord)) continue;

            Cluster currentCluster;
            std::queue<std::pair<VoxelCoord, int>> q;

            q.push({startCoord, 0});
            visited.insert(startCoord);

            while (!q.empty()) {
                auto [current, distFromRoot] = q.front();
                q.pop();

                // Assign this voxel to the current cluster
                currentCluster.addVoxel(grid[current]);
                
                // CRITICAL ADDITION: Record the link for the Index Matrix
                result.voxelToClusterIdx[current] = currentClusterIdx;

                if (distFromRoot >= maxSteps) continue;

                for (int dl = -1; dl <= 1; ++dl) {
                    for (int da = -1; da <= 1; ++da) {
                        for (int db = -1; db <= 1; ++db) {
                            if (dl == 0 && da == 0 && db == 0) continue;

                            VoxelCoord neighbor = {current.l_idx + dl, current.a_idx + da, current.b_idx + db};

                            if (grid.count(neighbor) && !visited.count(neighbor)) {
                                visited.insert(neighbor);
                                q.push({neighbor, distFromRoot + 1});
                            }
                        }
                    }
                }
            }
            result.clusters.push_back(currentCluster);
            currentClusterIdx++; // Increment index for the next cluster found
        }
        return result;
    }
};


#endif