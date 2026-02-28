
#include <queue>
#include <bitset>
#include <fstream>   // For std::ofstream and std::ifstream
#include <iostream>  // For std::cout
#include <vector>    // For std::vector
#include <string>    // For std::string
#include <filesystem> // For creating directories (C++17 or later)
#include <iomanip>
#include <cstdint>
#include <map>

class BitWriter {
    std::ofstream& out;
    uint8_t buffer;   // Temporary storage for bits
    int bitCount;     // How many bits are currently in the buffer

public:
    BitWriter(std::ofstream& f) : out(f), buffer(0), bitCount(0) {}

    // value: the data to write
    // numBits: how many bits of that data to actually use
    void write(uint32_t value, int numBits) {
        for (int i = numBits - 1; i >= 0; --i) {
            // Extract the i-th bit from the value
            bool bit = (value >> i) & 1;
            
            // Push it into our 8-bit buffer
            buffer = (buffer << 1) | bit;
            bitCount++;

            // If we have a full byte (8 bits), write it to the file
            if (bitCount == 8) {
                out.put(static_cast<char>(buffer));
                buffer = 0;
                bitCount = 0;
            }
        }
    }

    // Very important: writes the last remaining bits
    void flush() {
        if (bitCount > 0) {
            buffer <<= (8 - bitCount); // Move bits to the start of the byte
            out.put(static_cast<char>(buffer));
            buffer = 0;
            bitCount = 0;
        }
    }
};



// Internal structure for Huffman Tree
struct HuffmanNode {
    int value;
    unsigned freq;
    HuffmanNode *left, *right;
    HuffmanNode(int v, unsigned f) : value(v), freq(f), left(nullptr), right(nullptr) {}
};




// Node for the Huffman Tree
struct Node {
    int id;
    int freq;
    Node *left, *right;
    Node(int i, int f) : id(i), freq(f), left(nullptr), right(nullptr) {}
};

// Comparison for the priority queue
struct Compare {
    bool operator()(Node* l, Node* r) { return l->freq > r->freq; }
};

// Recursive function to build the code table
void buildCodes(Node* root, std::string str, std::map<int, std::pair<uint32_t, int>>& huffmanTable) {
    if (!root) return;
    if (!root->left && !root->right) {
        // Convert bit-string "101" to an integer 5 and length 3
        huffmanTable[root->id] = { (uint32_t)std::bitset<32>(str).to_ulong(), (int)str.length() };
    }
    buildCodes(root->left, str + "0", huffmanTable);
    buildCodes(root->right, str + "1", huffmanTable);
}
