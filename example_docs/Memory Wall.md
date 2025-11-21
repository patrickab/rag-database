---
title: Memory Wall
aliases:
  - CPU-memory gap
  - processor-memory bottleneck
  - memory bottleneck
  - CPU stalling
tags:
  - CPU
  - DRAM
  - Bottleneck
  - computer-architecture
  - high-performance-computing
summary: The growing performance gap between increasingly fast CPUs and relatively slower DRAM, causing memory access to become a primary system bottleneck.
---
The "**memory wall**" refers to the growing performance gap between increasingly fast Central Processing Units (CPUs) and relatively slower Dynamic Random-Access Memory (DRAM). 

Computational power has advanced much faster than data transmission speed.

**Reasons for the Memory Wall**
*   **Divergent Scaling**:
    *   **CPU Speed**: Driven by [[Moores Law]], CPU clock frequencies and core counts have increased exponentially, allowing more instructions per second.
    *   **Memory Speed**: [[DRAM]] latency (time to access data) and bandwidth (data transfer rate) have improved at a much slower pace. This comes from physical limitations in charge/discharge times and signal propagation.
*   **Physical Limitations**:
    *   **DRAM Physics**: Accessing DRAM involves charging/discharging capacitors, which has inherent physical delays.
    *   **Interconnects**: The electrical pathways between CPU and memory also introduce latency and bandwidth constraints.
*   **Power Consumption**: Increasing DRAM speed significantly often leads to disproportionately higher power consumption, which is undesirable.

**Consequences**
*   **CPU Stalling**: CPUs frequently sit idle, waiting for data to arrive.
*   **Performance Bottleneck**: Memory access becomes the primary bottleneck.
*   **Increased Cache Reliance**: Architects heavily rely on complex multi-level cache hierarchies to bridge the gap.

**💡 Key Takeaways**
*   **Speed Disparity**: CPU processing speed outpaces DRAM access speed significantly.
*   **Bottleneck**: Memory access, not computation, limits overall system performance for many workloads.
*   **Architectural Challenge**: Requires sophisticated caching and memory management techniques to mitigate.