---
title: Instruction-Level Parallelism
aliases:
  - ILP
  - Instruction Level Parallelism
tags:
  - computer-architecture
  - parallel-processing
  - CPU
  - high-performance-computing
summary: Techniques in computer architecture that allow a processor to execute multiple independent instructions concurrently to maximize throughput.
---
Instruction-level parallelism (ILP) is a measure of how many operations in a computer program can be executed simultaneously by a processor.

**Mechanisms for Achieving ILP**
*   **Pipelining**: Overlapping the execution phases of multiple instructions (e.g., fetch, decode, execute, write-back).
*   **Superscalar Architectures**: Processors with multiple execution units that can issue and execute several instructions per clock cycle.
*   **Out-of-Order Execution**: Reordering instructions dynamically at runtime to fill execution unit pipelines, respecting data dependencies.
*   **Speculative Execution**: Executing instructions before their control dependencies (e.g., branch outcomes) are resolved, rolling back if speculation is incorrect.

**💡 Key Takeaways**
*   **Concurrent Instruction Execution**: Multiple independent instructions are executed simultaneously within a single processor core.
*   **Hardware-Driven**: Primarily managed by the CPU's microarchitecture (e.g., pipelines, multiple execution units).
*   **Goal**: Maximize CPU throughput by keeping execution units busy.
*   **Contrast to TLP**: ILP focuses on parallelizing individual instructions within a single thread, whereas Thread-Level Parallelism (TLP) parallelizes entire threads.