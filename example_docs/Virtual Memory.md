---
title: Virtual Memory
aliases:
  - VM
  - virtual address space
tags:
  - memory-management
  - computer-architecture
  - parallel-processing
summary: An operating system technique that provides applications with an illusion of a contiguous, large, and private memory space, abstracting the underlying physical memory.
---
Virtual memory is an operating system technique that provides an application with an illusion of a contiguous, large, and private memory space, abstracting the underlying physical memory.

**Core Concepts**
-   **Abstraction**: Programs see a logical address space (virtual addresses) independent of the physical RAM layout.
-   **Address Translation**: A Memory Management Unit (MMU) translates virtual addresses to physical addresses using **page tables**.
-   **Paging**: The virtual address space is divided into fixed-size blocks called **pages**, and physical memory into **frames**. Pages can be swapped between RAM and disk (swap space) as needed.
-   **Protection**: Isolates processes, preventing one from accessing another's memory or the OS's memory directly.
-   **Memory Sharing**: Allows multiple processes to share common code or data pages efficiently.

**💡 Key Takeaways**
-   **Illusion of Infinite Memory**: Provides each process with its own large, contiguous address space, even if physical RAM is fragmented or limited.
-   **Hardware-Software Collaboration**: Relies on both OS software (managing page tables, swapping) and CPU hardware (MMU for translation).
-   **Performance Trade-off**: Enables efficient memory utilization and multi-tasking but introduces overhead due to address translation and potential page faults (disk I/O).