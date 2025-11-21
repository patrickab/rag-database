---
title: Stride
aliases:
  - step size
  - memory access stride
  - memory stride
tags:
  - memory
  - cache
  - performance
  - data-locality
  - high-performance-computing
  - parallel-processing
  - memory-management
summary: The step size (in memory units) between successive memory accesses, directly impacting spatial locality and cache hierarchy performance.
---
**Stride** refers to the step size (in memory units, often bytes or elements) between successive memory accesses in a sequence. For example, accessing every element in an array has a stride of 1 (element size), while accessing every other element has a stride of 2.

**Connection to [[Memory Hierarchy|Cache Hierarchy]] & [[Data Locality]]**

1.  **Spatial Locality**: Stride directly impacts spatial locality.
    *   **Small Stride (e.g., stride-1)**: Maximizes spatial locality. When a cache line is fetched due to a miss, subsequent accesses with a small stride are likely to find their data already in the cache, leading to cache hits.
    *   **Large Stride**: Destroys spatial locality. Successive accesses are far apart in memory, often falling into different cache lines. This means each access is more likely to result in a cache miss, even if the data is "nearby" in the logical array structure.

2.  **Cache Hierarchy**: The performance of the multi-level cache hierarchy (L1, L2, L3) is highly dependent on stride.
    *   When spatial locality is high (small stride), data is efficiently loaded into the fastest, smallest caches (L1). Subsequent accesses hit in L1, providing very low latency and high throughput.
    *   When spatial locality is low (large stride), each access often misses in L1, then L2, and potentially L3, forcing a fetch from the much slower main memory. This significantly increases access latency and reduces overall memory throughput, as the benefits of the faster cache levels are negated.

**💡 Key Takeaways**
- **Stride**: The distance between consecutive memory accesses.
- **Small Stride (e.g., 1)**: Enhances spatial locality, leading to more cache hits and higher performance across the cache hierarchy.
- **Large Stride**: Reduces spatial locality, causing more cache misses and forcing slower main memory accesses, thus degrading performance.