# C++ jpegli Huffman Tree Construction Analysis

## The Pseudo-Symbol 256 Trick

### Problem
When building optimal Huffman tables from symbol frequencies, the code space can become completely full (Kraft sum = 2^16). This causes libjpeg-turbo to crash and some decoders (zune-jpeg) to reject the tables.

### C++ Solution
The C++ jpegli implementation uses a **pseudo-symbol 256** approach to guarantee slack space in the code tree.

## Key Files

- `lib/jpegli/huffman.cc` - Tree construction algorithm
- `lib/jpegli/entropy_coding.cc` - Huffman table building

## Algorithm Walkthrough

### Step 1: Add Pseudo-Symbol (entropy_coding.cc:671-677)

```cpp
void BuildJpegHuffmanTable(const Histogram& histo, JHUFF_TBL* table) {
  std::vector<uint32_t> counts(kJpegHuffmanAlphabetSize + 1);  // 257 elements
  std::vector<uint8_t> depths(kJpegHuffmanAlphabetSize + 1);   // 257 elements

  for (size_t j = 0; j < kJpegHuffmanAlphabetSize; ++j) {
    counts[j] = histo.count[j];  // Copy symbols 0-255
  }
  counts[kJpegHuffmanAlphabetSize] = 1;  // Add pseudo-symbol 256 with frequency 1
```

**Key insight**: `kJpegHuffmanAlphabetSize = 256`, so we create arrays of size 257.

### Step 2: Build Tree with 257 Symbols (entropy_coding.cc:678-679)

```cpp
  CreateHuffmanTree(counts.data(), counts.size(), kJpegHuffmanMaxBitLength,
                    depths.data());
```

This calls the tree construction algorithm with:
- `data` = 257 symbol frequencies (0-255 + pseudo-symbol 256)
- `length` = 257
- `tree_limit` = 15 (kJpegHuffmanMaxBitLength)
- `depth` = output array for 257 code lengths

### Step 3: CreateHuffmanTree Algorithm (huffman.cc:157-234)

#### Data Structures

```cpp
struct HuffmanTree {
  HuffmanTree(uint32_t count, int16_t left, int16_t right)
      : total_count(count), index_left(left), index_right_or_value(right) {}
  uint32_t total_count;   // Frequency (for leaves) or sum (for internal nodes)
  int16_t index_left;     // Left child index (-1 for leaf nodes)
  int16_t index_right_or_value;  // Right child index OR symbol value (for leaves)
};
```

#### Retry Loop with count_limit

```cpp
void CreateHuffmanTree(const uint32_t* data, const size_t length,
                       const int tree_limit, uint8_t* depth) {
  // Retry loop: if tree is too deep, increase minimum count and retry
  for (uint32_t count_limit = 1;; count_limit *= 2) {
```

The `count_limit` is the minimum frequency assigned to any symbol. If the tree exceeds `tree_limit` depth, we increase this minimum, which forces rarer symbols to merge earlier, creating a shallower tree.

#### Build Initial Leaf Nodes (huffman.cc:167-173)

```cpp
    std::vector<HuffmanTree> tree;
    tree.reserve(2 * length + 1);

    for (size_t i = length; i != 0;) {
      --i;
      if (data[i]) {  // Only include symbols with non-zero frequency
        const uint32_t count = std::max(data[i], count_limit - 1);
        tree.emplace_back(count, -1, static_cast<int16_t>(i));
      }
    }
```

Creates leaf nodes for non-zero frequency symbols. `index_left = -1` marks this as a leaf, `index_right_or_value` stores the symbol value.

#### Special Case: Single Symbol (huffman.cc:176-180)

```cpp
    const size_t n = tree.size();
    if (n == 1) {
      // Fake value; will be fixed on upper level.
      depth[tree[0].index_right_or_value] = 1;
      break;
    }
```

If only one symbol has non-zero frequency, assign it depth 1. The pseudo-symbol 256 prevents this case in practice (minimum 2 symbols).

#### Package-Merge Algorithm (huffman.cc:182-224)

```cpp
    std::sort(tree.begin(), tree.end(), Compare);  // Sort by frequency

    // Add sentinels
    const HuffmanTree sentinel(std::numeric_limits<uint32_t>::max(), -1, -1);
    tree.push_back(sentinel);  // After sorted leaves
    tree.push_back(sentinel);  // Will be parent node

    // Array layout:
    // [0, n): sorted leaf nodes
    // [n]: sentinel
    // [n+1, 2n): new parent nodes (added during merge)
    // [2n]: final sentinel

    size_t i = 0;      // Points to next leaf node
    size_t j = n + 1;  // Points to next non-leaf node

    for (size_t k = n - 1; k != 0; --k) {
      // Pick two nodes with smallest total_count
      size_t left, right;

      if (tree[i].total_count <= tree[j].total_count) {
        left = i++;
      } else {
        left = j++;
      }

      if (tree[i].total_count <= tree[j].total_count) {
        right = i++;
      } else {
        right = j++;
      }

      // Create parent node
      size_t j_end = tree.size() - 1;
      tree[j_end].total_count = tree[left].total_count + tree[right].total_count;
      tree[j_end].index_left = static_cast<int16_t>(left);
      tree[j_end].index_right_or_value = static_cast<int16_t>(right);
      tree.push_back(sentinel);  // Add new sentinel
    }
```

This is a classic Huffman algorithm optimization using two sorted arrays:
- Leaf nodes (sorted once at start)
- Parent nodes (naturally sorted as we create them)

By maintaining two sorted arrays, we can find the two minimum elements in O(1) instead of O(log n).

#### Compute Depths (huffman.cc:226)

```cpp
    SetDepth(tree[2 * n - 1], tree.data(), depth, 0);
```

Recursively traverse the tree to compute code lengths:

```cpp
void SetDepth(const HuffmanTree& p, HuffmanTree* pool, uint8_t* depth,
              uint8_t level) {
  if (p.index_left >= 0) {
    // Internal node: recurse on children
    ++level;
    SetDepth(pool[p.index_left], pool, depth, level);
    SetDepth(pool[p.index_right_or_value], pool, depth, level);
  } else {
    // Leaf node: set depth for this symbol
    depth[p.index_right_or_value] = level;
  }
}
```

#### Check Depth Limit (huffman.cc:231-233)

```cpp
    if (*std::max_element(&depth[0], &depth[length]) <= tree_limit) {
      break;  // Success! All code lengths <= 15 bits
    }
  }  // Otherwise, retry with count_limit *= 2
```

### Step 4: Build JPEG Huffman Table - EXCLUDE Pseudo-Symbol (entropy_coding.cc:680-694)

```cpp
  memset(table, 0, sizeof(JHUFF_TBL));

  // Count codes per length (EXCLUDE symbol 256!)
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {  // Only 0-255
    if (depths[i] > 0) {
      ++table->bits[depths[i]];
    }
  }

  // Build offset array
  int offset[kJpegHuffmanMaxBitLength + 1] = {0};
  for (size_t i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
    offset[i] = offset[i - 1] + table->bits[i - 1];
  }

  // Populate huffval array (EXCLUDE symbol 256!)
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {  // Only 0-255
    if (depths[i] > 0) {
      table->huffval[offset[depths[i]]++] = i;
    }
  }
}
```

**Critical**: Both loops iterate only up to `kJpegHuffmanAlphabetSize` (256), excluding the pseudo-symbol.

## Why This Works

1. **Pseudo-symbol participates in tree construction**:
   - With frequency 1 (minimal), it typically gets assigned the longest code length
   - Prevents single-symbol edge case
   - Influences tree structure

2. **Pseudo-symbol excluded from final table**:
   - We build the DHT marker with only symbols 0-255
   - The codeword assigned to symbol 256 becomes **unused**
   - This creates slack space in the code tree

3. **Kraft sum < 2^16**:
   - Because we have one unused codeword, the sum of all used codewords is strictly less than 2^16
   - libjpeg-turbo accepts the table
   - zune-jpeg accepts the table

## Example

Suppose symbol 256 gets assigned depth 15 (longest code):

- Total 15-bit codewords available: 2^(16-15) = 2
- Symbol 256 uses one 15-bit codeword
- That codeword is NOT in the DHT table
- Kraft sum contribution: (count_at_length_15 - 1) * 2^(16-15)
- Result: Slack space = one unused 15-bit codeword

## Rust Implementation Strategy

1. **Modify frequency counting**: Add pseudo-symbol 256 with frequency 1
2. **Modify tree construction**: Accept 257-element input array
3. **Modify table building**: Only include symbols 0-255 in output
4. **Keep depth limiting**: Existing retry loop with count_limit

## Key Constants

```cpp
constexpr size_t kJpegHuffmanAlphabetSize = 256;
constexpr size_t kJpegHuffmanMaxBitLength = 15;
```
