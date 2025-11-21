---
title: Data Level Parallelism
aliases:
  - DLP
  - SIMD
  - Vectorization
tags:
  - computer-architecture
  - SIMD
  - CPU
  - high-performance-computing
  - hardware
  - parallel-processing
summary: Simultaneous execution of the same operation on multiple, independent data items, often using SIMD instructions, distinct from instruction-level parallelism.
---
**Data-Level Parallelism (DLP)** refers to the simultaneous execution of the same operation on multiple, independent data items.

**Key Aspects of DLP**
-   **Single Instruction, Multiple Data (SIMD)**: The primary mechanism for exploiting DLP, where a single instruction operates on a vector or array of data elements.
-   **Vectorization**: Compilers and programmers transform loops to operate on multiple data elements in parallel using SIMD instructions.
-   **Examples**: Operations on arrays, image processing (applying a filter to many pixels), scientific computing (matrix operations).

**💡 Key Takeaways**
-   **DLP**: Same operation on multiple data items (SIMD).
-   **Issue Width**: Multiple instructions dispatched per cycle (ILP).
-   **Complementary**: A wide-issue CPU can dispatch SIMD instructions, combining ILP and DLP.
-   **Distinct**: DLP is about *data* parallelism within an instruction; issue width is about *instruction* parallelism.