---
title: Memory Hierarchy
aliases: [Memory Tiers, Memory Levels, CPU Memory Hierarchy, Computer Memory Hierarchy]
tags: [computer-architecture, memory, cpu, caching, performance]
summary: A tiered structure of memory types that bridges the speed gap between CPU and main memory by exploiting locality of reference.
---
The memory hierarchy is a multi-tiered storage system designed to bridge [[Memory Wall]].

It exploits [[Data Locality]] by
- placing smaller / faster / expensive memory technologies closer to the CPU
- and larger, slower, cheaper technologies further away from the CPU.


**The Tiers**
- **Registers**: (On-CPU) Fastest, smallest. Hold data for the current instruction.
- **CPU Caches (L1, L2, L3)**: (SRAM) Extremely fast, on-chip memory. Stores frequently used data from RAM to hide latency.
- **Main Memory**: (DRAM) The system's "working memory." Much larger than caches but significantly slower.
- **Storage**: (SSD/HDD) Slowest and largest. Provides persistent, non-volatile storage.

**Governing Principle: Locality of Reference**
The hierarchy is effective because programs exhibit predictable access patterns.
- **Temporal Locality**: If an item is accessed, it's likely to be accessed again soon. *(e.g., variables in a loop)*
- **Spatial Locality**: If an item is accessed, items at nearby memory addresses are likely to be accessed soon. *(e.g., sequential array elements)*

**Mechanism**
- **Cache Hit**: The CPU finds the data it needs in a cache (e.g., L1). This is the fastest case.
- **Cache Miss**: The data is not in the cache. The CPU must fetch it from a lower, slower level (e.g., Main Memory). When this happens, a *block* of data (not just the requested byte) is copied into the higher-level cache, anticipating future use due to spatial locality.

**💡 Key Takeaways**
- The hierarchy exploits a trade-off: speed vs. cost vs. capacity.
- Its goal is to create the illusion of a memory that is as large as the slowest level but as fast as the fastest level.
- The entire system works because of the *principle of locality*, which makes data caching highly effective.