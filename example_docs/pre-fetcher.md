---
title: hardware prefetcher
aliases:
  - prefetcher
  - memory prefetcher
  - data prefetcher
tags:
  - prefetcher
  - computer-architecture
  - memory-subsystem
  - cache
  - performance-optimization
  - high-performance-computing
  - CPU
  - computing
summary: A hardware component that predicts future memory accesses and proactively fetches data into caches to hide memory latency and improve CPU performance.
---
Component in the memory subsystem that predicts future memory accesses based on past patterns.

It proactively fetches data from slower memory (e.g., DRAM) into faster caches *before* the CPU explicitly requests it.

This hides memory access latency, improving performance by reducing stalls.

**💡 Key Takeaways**
- **Predictive Loading**: Prefetchers are speculative hardware units that anticipate data needs.
- **Latency Hiding**: Their primary goal is to mask the high latency of main memory accesses.
- **Pattern Recognition**: They operate by identifying simple access patterns, such as sequential or strided memory requests.