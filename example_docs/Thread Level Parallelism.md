---
title: Thread-level parallelism
aliases:
  - TLP
  - Thread Level Parallelism
tags:
  - parallel-processing
  - high-performance-computing
  - CPU
summary: A form of concurrent execution where multiple threads within a single process run simultaneously on different processor cores to improve performance.
---
Thread-level parallelism (**TLP**) is a form of concurrent execution where multiple threads within a single process run simultaneously on different processor cores.

**💡 Key Takeaways**
*   **Concurrent Execution**: Multiple threads of a single process execute simultaneously.
*   **Shared Resources**: Threads within a process share the same memory space and resources.
*   **Performance Improvement**: Achieved by utilizing multi-core processors for faster task completion.
*   **Contrast to [[Instruction Level Parallelism]] (ILP)**: TLP operates at a higher level, managing entire threads rather than individual instructions.