# CUDA Programming: Complete Guide from Basic to Advanced

> A comprehensive reference covering every concept from the CUDA course — with code examples, explanations, and analogies.

---

## Table of Contents

1. [The Deep Learning Ecosystem](#1-the-deep-learning-ecosystem)
2. [C/C++ Essentials for CUDA](#2-cc-essentials-for-cuda)
3. [Understanding GPUs — Why They Matter](#3-understanding-gpus--why-they-matter)
4. [CUDA Fundamentals](#4-cuda-fundamentals)
5. [Writing Your First Kernels](#5-writing-your-first-kernels)
6. [Thread Indexing Deep Dive](#6-thread-indexing-deep-dive)
7. [Vector Addition — Your First Real Kernel](#7-vector-addition--your-first-real-kernel)
8. [Matrix Multiplication — Naive GPU](#8-matrix-multiplication--naive-gpu)
9. [Profiling CUDA Kernels](#9-profiling-cuda-kernels)
10. [Tiled Matrix Multiplication with Shared Memory](#10-tiled-matrix-multiplication-with-shared-memory)
11. [Atomic Operations](#11-atomic-operations)
12. [CUDA Streams & Concurrency](#12-cuda-streams--concurrency)
13. [CUDA APIs — cuBLAS](#13-cuda-apis--cublas)
14. [CUDA APIs — cuDNN](#14-cuda-apis--cudnn)
15. [Optimizing Matrix Multiplication](#15-optimizing-matrix-multiplication)
16. [Triton — Python GPU Programming](#16-triton--python-gpu-programming)
17. [Custom PyTorch CUDA Extensions](#17-custom-pytorch-cuda-extensions)
18. [Multi-GPU & Distributed Computing](#18-multi-gpu--distributed-computing)
19. [Extras & Advanced Topics](#19-extras--advanced-topics)
20. [CUDA Cheatsheet](#20-cuda-cheatsheet)

---

## 1. The Deep Learning Ecosystem

Before diving into CUDA, it's essential to understand **where CUDA sits** in the deep learning world.

### Analogy: The Restaurant Kitchen
Think of the deep learning ecosystem like a restaurant:
- **PyTorch/TensorFlow** = The recipe book (high-level frameworks)
- **cuBLAS/cuDNN** = The professional kitchen tools (optimized libraries)
- **CUDA** = Knowing how to build and maintain the kitchen tools yourself (low-level programming)
- **Triton** = A simpler way to craft custom tools without needing to be a full blacksmith

### Frameworks (Research)
| Framework | Key Point |
|-----------|-----------|
| **PyTorch** | Most popular for research. Dynamic computation graphs. Great HuggingFace integration |
| **TensorFlow** | Google's framework. Designed for TPUs. Most used overall |
| **JAX** | JIT-compiled, feels like NumPy, uses XLA compiler |
| **MLX** | Apple's framework for Apple Silicon |

### Production / Inference
| Tool | Purpose |
|------|---------|
| **vLLM** | Fast LLM inference |
| **TensorRT** | NVIDIA's inference optimizer — quantization, sparsity, kernel optimization |
| **torch.compile** | Compiles PyTorch models to optimized binaries |
| **ONNX Runtime** | Microsoft's cross-platform inference engine |
| **Triton (OpenAI)** | Python-based GPU kernel language matching cuBLAS-level perf |

### Low-Level
| Tool | Purpose |
|------|---------|
| **CUDA** | NVIDIA GPU programming language (this course!) |
| **ROCm** | CUDA equivalent for AMD GPUs |
| **OpenCL** | Open standard for CPUs, GPUs, and other hardware |

### Key Takeaway
> CUDA is the **foundation** that powers nearly every deep learning framework on NVIDIA GPUs. Understanding it lets you optimize performance, build custom kernels for cutting-edge research, and understand GPU bottlenecks — especially memory bandwidth.

---

## 2. C/C++ Essentials for CUDA

CUDA is built on top of C/C++. Here are the critical concepts you need.

### 2.1 Pointers

**Analogy:** A pointer is like a **home address**. The address itself isn't the house — it tells you where the house is located. The `&` operator gets the address, and `*` lets you visit the house (dereference).

```c
#include <stdio.h>

int main() {
    int x = 10;
    int* ptr = &x;           // ptr stores the ADDRESS of x
    printf("Address: %p\n", ptr);   // Print the address
    printf("Value: %d\n", *ptr);    // Dereference: get the VALUE at that address → 10
}
```

### 2.2 Multi-level Pointers

A pointer to a pointer — like writing down the address of the piece of paper that has another address on it.

```c
int value = 42;
int* ptr1 = &value;      // ptr1 → value
int** ptr2 = &ptr1;       // ptr2 → ptr1 → value
int*** ptr3 = &ptr2;      // ptr3 → ptr2 → ptr1 → value

printf("Value: %d\n", ***ptr3);  // Output: 42
```

### 2.3 Void Pointers

A "generic" pointer that can hold the address of any type. You must cast it before dereferencing.

```c
int num = 10;
float fnum = 3.14;
void* vptr;

vptr = &num;
printf("Integer: %d\n", *(int*)vptr);    // Cast to int*, then dereference

vptr = &fnum;
printf("Float: %.2f\n", *(float*)vptr);  // Cast to float*, then dereference
```

> **Fun fact:** `malloc()` returns a void pointer! That's why you cast it: `(int*)malloc(sizeof(int))`

### 2.4 NULL Pointers & Safe Memory Practices

Always initialize pointers to `NULL` and check before using:

```c
int* ptr = NULL;  // Safe initialization

// Always check before dereferencing
if (ptr == NULL) {
    printf("ptr is NULL, cannot dereference\n");
}

// After malloc
ptr = malloc(sizeof(int));
if (ptr == NULL) {
    printf("Memory allocation failed!\n");
    return 1;
}
*ptr = 42;

// Always free and nullify
free(ptr);
ptr = NULL;  // Prevent "use after free" bugs
```

### 2.5 Pointer Arithmetic & Arrays

**Analogy:** Arrays are like a row of numbered lockers in a hallway. Pointer arithmetic lets you walk from one locker to the next.

```c
int arr[] = {12, 24, 36, 48, 60};
int* ptr = arr;  // Points to first element

for (int i = 0; i < 5; i++) {
    printf("%d at address %p\n", *ptr, ptr);
    ptr++;  // Move to next element (advances by sizeof(int) = 4 bytes)
}
```

> **Key:** Arrays are contiguous in memory. Each `ptr++` advances by `sizeof(type)` bytes. For `int` (4 bytes), addresses increase by 4.

### 2.6 Matrix as Array of Pointers

```c
int arr1[] = {1, 2, 3, 4};
int arr2[] = {5, 6, 7, 8};
int* matrix[] = {arr1, arr2};  // Array of pointers

for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
        printf("%d ", *(matrix[i] + j));  // Pointer arithmetic for 2D access
    }
    printf("\n");
}
```

### 2.7 `size_t` — The Safe Size Type

```c
int arr[] = {12, 24, 36, 48, 60};
size_t size = sizeof(arr) / sizeof(arr[0]);  // 5 elements
printf("Size: %zu\n", size);                  // %zu for size_t
printf("size_t bytes: %zu\n", sizeof(size_t)); // 8 bytes (64-bit, memory-safe)
```

### 2.8 Structs (Custom Types)

```c
typedef struct {
    float x;
    float y;
} Point;

Point p = {1.1, 2.5};
printf("Size of Point: %zu\n", sizeof(Point));  // 8 bytes (4 + 4)
```

### 2.9 Type Casting

```c
float f = 69.69;
int i = (int)f;        // Truncates decimal → 69
char c = (char)i;      // ASCII 69 → 'E'
```

**C++ style casts:**
- `static_cast<int>(3.14)` — compile-time checked, most common
- `dynamic_cast` — runtime-checked downcasting in inheritance
- `const_cast` — add/remove `const`
- `reinterpret_cast` — dangerous, converts between unrelated types

### 2.10 Macros & Preprocessor Directives

```c
#define PI 3.14159
#define AREA(r) (PI * r * r)

#ifndef radius
#define radius 7
#endif

// Conditional compilation
#if radius > 10
    #define radius 10
#elif radius < 5
    #define radius 5
#endif

printf("Area: %f\n", AREA(radius));
```

### 2.11 Compilers
- **gcc** — C compiler
- **g++** — C++ compiler
- **nvcc** — NVIDIA CUDA Compiler (compiles `.cu` files)

### 2.12 Makefiles

**Analogy:** A Makefile is like a **recipe card** — it tells the computer exactly how to build your program step by step.

```makefile
.PHONY: 01 02 03 clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_86

01:
    @$(GCC) -o 01 01.c

03:
    @$(NVCC) $(CUDA_FLAGS) -o 03_cu 03.cu

clean:
    rm -f 01 02 03_cu *.o
```

Key Makefile concepts:
- `.PHONY` — prevents conflicts with files/directories of the same name
- `:=` — immediate assignment (evaluated once)
- `=` — recursive assignment (re-evaluated each use)
- `@` — suppresses command echo

### 2.13 Debuggers

- **gdb** — for debugging C/C++ programs
- **cuda-gdb** — for debugging CUDA programs

Key commands: `run`, `break`, `next`, `step`, `print`, `continue`, `quit`

---

## 3. Understanding GPUs — Why They Matter

### CPU vs GPU: The Fundamental Difference

**Analogy: Professor vs Army of Students**
- **CPU** = A brilliant professor who can solve any math problem extremely fast, but works on one problem at a time
- **GPU** = An army of 10,000 students, each less clever, but together they can solve 10,000 simple math problems simultaneously

| Feature | CPU (Host) | GPU (Device) |
|---------|------------|--------------|
| Purpose | General purpose | Specialized parallel |
| Clock Speed | High (~5 GHz) | Lower (~1.5 GHz) |
| Cores | Few (8-64) | Many (thousands) |
| Cache | Large | Small per core |
| Latency | Low | Higher |
| Throughput | Low | Very High |
| Metric | Seconds per task | Tasks per second |

### Other Processing Units
- **TPU** — Google's Tensor Processing Unit, specialized for matrix ops
- **FPGA** — Reconfigurable hardware, very low latency, high cost

### Why GPUs for Deep Learning?
Deep learning is mostly matrix multiplication → massively parallel → perfect for GPUs!

### The Typical CUDA Program Flow

```
1. CPU allocates memory
2. CPU copies data → GPU
3. CPU launches kernel on GPU (processing happens here)
4. CPU copies results GPU → CPU
```

**Analogy: The Jigsaw Puzzle**
> Imagine solving a jigsaw puzzle. You give each thread one puzzle piece with its target location. Each thread independently places its piece. As long as all pieces end up in the right spot, the order doesn't matter. You can place many pieces simultaneously — that's GPU parallelism!

### Key Terminology
| Term | Meaning |
|------|---------|
| **Kernel** | A function that runs on the GPU (not Linux kernel, not convolution kernel) |
| **Host** | The CPU + system RAM |
| **Device** | The GPU + VRAM |
| **GEMM** | **GE**neral **M**atrix **M**ultiplication |
| **SGEMM** | **S**ingle precision (fp32) GEMM |
| **HGEMM** | **H**alf precision (fp16) GEMM |

---

## 4. CUDA Fundamentals

### 4.1 Function Qualifiers

| Qualifier | Called From | Runs On | Purpose |
|-----------|-----------|---------|---------|
| `__global__` | Host (CPU) | Device (GPU) | Your CUDA kernels — the main GPU functions |
| `__device__` | Device (GPU) | Device (GPU) | Helper functions called by kernels |
| `__host__` | Host (CPU) | Host (CPU) | Regular C/C++ functions |

**Analogy:**
- `__global__` = The manager's instructions broadcast to all factory workers
- `__device__` = A tool that only factory workers can use among themselves
- `__host__` = Office work that only the manager does

```cpp
// A GPU kernel — launched from CPU, runs on GPU
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}

// A device helper — only callable from GPU code
__device__ float square(float x) {
    return x * x;
}
```

### 4.2 Naming Convention
- `h_A` — host variable (CPU)
- `d_A` — device variable (GPU)

### 4.3 Memory Management

**Analogy: Moving Boxes Between Buildings**
- `cudaMalloc` = Reserving shelf space in the GPU warehouse
- `cudaMemcpy` = Moving boxes between the office (CPU) and warehouse (GPU)
- `cudaFree` = Clearing the shelf space

```cpp
float *d_a, *d_b, *d_c;

// Allocate GPU memory
cudaMalloc(&d_a, N * sizeof(float));
cudaMalloc(&d_b, N * sizeof(float));
cudaMalloc(&d_c, N * sizeof(float));

// Copy data: CPU → GPU
cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

// ... run kernel ...

// Copy results: GPU → CPU
cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

**Memory copy directions:**
- `cudaMemcpyHostToDevice` — CPU → GPU
- `cudaMemcpyDeviceToHost` — GPU → CPU
- `cudaMemcpyDeviceToDevice` — GPU → GPU

### 4.4 The CUDA Hierarchy

**Analogy: The Apartment Complex**

```
Grid          = Entire apartment complex (city)
  Blocks      = Individual apartments
    Threads   = People living in each apartment
      Warps   = Groups of 32 people who always move together
```

1. A **Kernel** executes in a thread
2. **Threads** are grouped into **Thread Blocks** (Blocks)
3. **Blocks** are grouped into a **Grid**
4. A Kernel launches as a **Grid of Blocks of Threads**

### 4.5 The Four Magic Variables

| Variable | Meaning | Analogy |
|----------|---------|---------|
| `gridDim` | Number of blocks in the grid | How many apartments in the complex |
| `blockIdx` | This block's index in the grid | Which apartment we're in |
| `blockDim` | Number of threads per block | How many people per apartment |
| `threadIdx` | This thread's index within its block | Which person in the apartment |

**The Global Thread ID Formula:**
```cpp
int globalId = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` = starting index of the current block
`threadIdx.x` = offset within the block

### 4.6 Threads

- Each thread has **private local memory** (registers)
- Each thread does one piece of the larger computation

```
To add a = [1, 2, 3, ..., N] and b = [2, 4, 6, ..., N]:
  Thread 0: a[0] + b[0]
  Thread 1: a[1] + b[1]
  Thread 2: a[2] + b[2]
  ...
```

### 4.7 Warps

**Analogy: From Textile Weaving**
> In weaving, the **warp** is the set of yarns stretched on a loom. Similarly, in CUDA, a warp is a group of 32 threads that execute **in lockstep** — they all do the same instruction at the same time.

- Each warp = exactly **32 threads**
- Instructions are issued to **warps**, not individual threads
- A **warp scheduler** decides which warps run
- 4 warp schedulers per SM (Streaming Multiprocessor)
- Max 1024 threads per block → max 32 warps per block

### 4.8 Blocks

- Threads within a block share **shared memory** (fast on-chip SRAM)
- Threads in a block can **synchronize** with each other
- CUDA execution is **scalable** because blocks are independent — Block 3 might run before Block 0

**Analogy:**
> Each block is like one team in a relay race. Team members (threads) can pass the baton (shared memory) to each other, but teams work independently.

### 4.9 Grids

- The entire collection of blocks for one kernel launch
- During execution, all threads can access **global memory** (VRAM)
- Great for batch processing — each block handles one batch element

### 4.10 Hardware Mapping

| Software | Hardware |
|----------|----------|
| Threads | CUDA Cores |
| Blocks | Streaming Multiprocessors (SMs) |
| Grid | Entire GPU |

### 4.11 Memory Hierarchy

From fastest to slowest:

| Memory | Speed | Scope | Size |
|--------|-------|-------|------|
| **Registers** | Fastest | Per-thread | Very small |
| **Shared Memory / L1** | Very fast (SRAM) | Per-block | ~48-164 KB |
| **L2 Cache** | Fast (SRAM) | Across all SMs | ~6-50 MB |
| **Global Memory (VRAM)** | Slow (DRAM) | Entire GPU | 8-80 GB |
| **Host Memory (RAM)** | Slowest (relative) | CPU | 16-128 GB |

> **Goal:** Keep data in registers and shared memory as much as possible. Register spills to local memory (which is actually in global DRAM) — avoid this!

### 4.12 The `nvcc` Compiler

The NVIDIA CUDA Compiler flow:
1. **Host code** → compiled to x86 binary (runs on CPU)
2. **Device code** → compiled to PTX (Parallel Thread Execution)
3. **PTX → JIT compiled** to native GPU instructions at runtime

This JIT step enables **forward compatibility** — your code works on future GPU architectures.

### 4.13 SIMT (Single Instruction, Multiple Threads)

Similar to CPU's SIMD (Single Instruction, Multiple Data):
- Instead of a `for` loop running sequentially, each thread runs **one iteration**
- Simpler than CPU: in-order execution, no branch prediction
- Less control logic → more room for compute cores

### 4.14 Math Intrinsics

Special hardware-accelerated math functions on the GPU:
- Use `logf()`, `expf()`, `sqrtf()` (device) instead of `log()`, `exp()`, `sqrt()` (host)
- Compiler flag `-use_fast_math` converts automatically (tiny precision loss)
- `--fmad=true` enables fused multiply-add

---

## 5. Writing Your First Kernels

### 5.1 dim3 Type — Specifying Dimensions

```cpp
// 3D dimensions
dim3 blockSize(16, 16, 1);   // 16x16x1 = 256 threads per block
dim3 gridSize(8, 8, 1);      // 8x8x1 = 64 blocks

// 1D shorthand (also valid)
int gridDim = 16;    // 16 blocks
int blockDim = 256;  // 256 threads per block
```

### 5.2 Kernel Launch Syntax: `<<<>>>`

```cpp
// Launch a kernel
addNumbers<<<gridSize, blockSize>>>(a, b, result);

// Full syntax: <<<gridDim, blockDim, sharedMemBytes, stream>>>
kernel<<<grid, block, 0, 0>>>(args);  // 0 shared mem, default stream
```

### 5.3 `cudaDeviceSynchronize()`

By default, CPU and GPU run asynchronously. This function makes the CPU **wait** until all GPU operations complete.

```cpp
kernel<<<gridSize, blockSize>>>(data);
cudaDeviceSynchronize();  // Block CPU until GPU finishes
printf("Kernel completed!\n");
```

### 5.4 Thread Synchronization Inside Kernels

```cpp
__syncthreads();   // All threads in block must reach here before any proceed
__syncwarps();     // Sync all threads within a warp
```

**Why synchronize?**
> Imagine workers painting a wall. Worker A is still applying primer while Worker B starts painting over it. The paint job is ruined. `__syncthreads()` says: "Everyone finish priming BEFORE anyone starts painting."

### 5.5 Thread Safety & Race Conditions

A **race condition** is when threads compete to modify the same memory:

```
Thread 1: read counter (= 5)
Thread 2: read counter (= 5)  ← reads BEFORE Thread 1 writes!
Thread 1: write counter = 6
Thread 2: write counter = 6   ← Should be 7, but overwrites to 6!
```

Solution: Use `cudaDeviceSynchronize()` or atomic operations (covered later).

---

## 6. Thread Indexing Deep Dive

### 6.1 The 3D Indexing Kernel

**Analogy: Apartment Complex Addressing**
- `blockIdx.x` = apartment number on this floor
- `blockIdx.y * gridDim.x` = floor number
- `blockIdx.z * gridDim.x * gridDim.y` = building number

```cuda
__global__ void whoami(void) {
    // Block ID in the grid (apartment number)
    int block_id =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Offset: how many threads came before this block
    int block_offset = block_id * (blockDim.x * blockDim.y * blockDim.z);

    // Thread offset within the block
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // Global unique thread ID
    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main() {
    dim3 blocksPerGrid(2, 3, 4);    // 24 blocks
    dim3 threadsPerBlock(4, 4, 4);  // 64 threads per block
    // Total: 24 * 64 = 1,536 threads
    
    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}
```

### 6.2 Grid Size Calculation — The Ceiling Division

To ensure all elements are covered:

```cpp
#define N 1024
#define BLOCK_SIZE 256

int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
// = (1024 + 255) / 256 = 1279 / 256 = 4 blocks (integer division)
```

This formula guarantees we launch enough blocks even when `N` isn't perfectly divisible.

---

## 7. Vector Addition — Your First Real Kernel

### 7.1 CPU Version (for comparison)

```cpp
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

### 7.2 GPU Version (1D)

```cuda
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {          // Bounds check!
        c[i] = a[i] + b[i];
    }
}
```

### 7.3 Complete Program

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000000   // 10 million elements
#define BLOCK_SIZE 256

__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *h_a, *h_b, *h_c;       // Host pointers
    float *d_a, *d_b, *d_c;       // Device pointers
    size_t size = N * sizeof(float);

    // 1. Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 2. Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // 3. Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 4. Copy data: Host → Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 5. Launch kernel
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // 6. Copy results: Device → Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 7. Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

### 7.4 3D Vector Addition

For 3D data (e.g., volumetric data), use 3D indexing:

```cuda
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        c[idx] = a[idx] + b[idx];
    }
}

// Launch with 3D grid and block:
dim3 block_size(16, 8, 8);
dim3 num_blocks(
    (nx + block_size.x - 1) / block_size.x,
    (ny + block_size.y - 1) / block_size.y,
    (nz + block_size.z - 1) / block_size.z
);
vector_add_gpu_3d<<<num_blocks, block_size>>>(d_a, d_b, d_c, nx, ny, nz);
```

---

## 8. Matrix Multiplication — Naive GPU

### 8.1 The Math

```
A (M x K) × B (K x N) = C (M x N)

C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
```

Example:
```
A = [[1, 2],      B = [[7, 8, 9, 10],     C = [[29,  32,  35,  38],
     [3, 4],           [11,12,13, 14]]          [65,  72,  79,  86],
     [5, 6]]                                    [101, 112, 123, 134]]
```

### 8.2 CPU Implementation

```cpp
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
```

### 8.3 GPU Implementation (Naive)

**Analogy:** Each thread computes **one element** of the output matrix C. It's like assigning each cell of a spreadsheet to one worker.

```cuda
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Launch with 2D grid:
dim3 blockDim(32, 32);  // 32x32 = 1024 threads per block
dim3 gridDim((N + 31) / 32, (M + 31) / 32);
matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
```

**Problem:** Each thread reads an entire row of A and column of B from **slow global memory**. Very inefficient!

---

## 9. Profiling CUDA Kernels

### 9.1 Why Profile?

**Analogy:** You wouldn't try to make your car faster without first checking what's slowing it down — is it the engine, tires, or drag? Profiling tells you **where** your GPU code is bottlenecked.

### 9.2 NVIDIA Profiling Tools

| Tool | Level | Purpose |
|------|-------|---------|
| `nvidia-smi` | System | GPU utilization, memory usage |
| `nvitop` | System | Better interactive GPU monitor |
| `nsys` (Nsight Systems) | High-level | System-wide bottlenecks, timeline |
| `ncu` (Nsight Compute) | Low-level | Kernel-specific optimization |
| `compute-sanitizer` | Debug | Memory leak detection |

### 9.3 Profiling Workflow

```bash
# Step 1: Compile
nvcc -o kernel kernel.cu

# Step 2: System-level profile (find bottleneck kernels)
nsys profile --stats=true ./kernel

# Step 3: Kernel-level deep dive
ncu --kernel-name myKernel --launch-skip 0 --launch-count 1 ./kernel
```

**Strategy:**
1. Start with `nsys` → find which kernels take the most time
2. Use `ncu` → optimize those specific kernels

### 9.4 NVTX — Custom Profiling Markers

NVTX (NVIDIA Tools Extension) lets you label sections of your code for profiling:

```cuda
#include <nvtx3/nvToolsExt.h>

void matrixMul(float* A, float* B, float* C, int N) {
    nvtxRangePush("Matrix Multiplication");
    
    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePop();  // End of Matrix Multiplication
}
```

Compile with: `nvcc -o matmul matmul.cu -lnvToolsExt`

### 9.5 CUPTI — Build Your Own Profiler

The CUDA Profiling Tools Interface lets you build custom profiling tools with APIs for activities, callbacks, events, metrics, etc.

---

## 10. Tiled Matrix Multiplication with Shared Memory

### 10.1 The Problem with Naive Matmul

In the naive version, every thread reads data from **global memory** (slow DRAM). For a 1024×1024 matmul, each thread reads 1024 floats from A and 1024 from B, all from global memory.

### 10.2 The Solution: Tiling

**Analogy: The Library Study Group**
> Instead of everyone going to the (slow) library individually to get the same book, one person brings the book to the (fast) study room and everyone shares it. That study room is **shared memory**.

**How tiling works:**
1. Divide the matrix into tiles (e.g., 16×16)
2. Load one tile from A and B into **shared memory** (fast)
3. All threads in the block compute using the shared tile
4. Move to the next tile and repeat

```cuda
#define TILE_SIZE 16

__global__ void matmulTiled(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile into shared memory (with bounds checking)
        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();  // Wait for all threads to finish loading

        // Compute partial sum from this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        __syncthreads();  // Wait before loading next tile
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**Key insight:** `__syncthreads()` is called **twice** per tile iteration:
1. After loading data → ensure all data is in shared memory before computing
2. After computing → ensure all threads are done before overwriting shared memory with next tile

---

## 11. Atomic Operations

### 11.1 What Are Atomics?

**Analogy: The Bathroom Door Lock**
> An atomic operation is like a bathroom with a lock. Only one person can use it at a time. While someone is inside (modifying memory), everyone else waits. This prevents conflicts but is slower than everyone going simultaneously.

An **atomic operation** ensures that a memory read-modify-write is completed entirely by one thread before another can access the same location.

### 11.2 The Race Condition Problem

```cuda
// WITHOUT atomics — BROKEN!
__global__ void incrementNonAtomic(int* counter) {
    int old = *counter;     // Thread A reads 5
    int new_val = old + 1;  // Thread B also reads 5 (race!)
    *counter = new_val;     // Both write 6 instead of 7
}

// WITH atomics — CORRECT!
__global__ void incrementAtomic(int* counter) {
    atomicAdd(counter, 1);  // Hardware-guaranteed safe increment
}
```

Result with 1,000,000 threads:
- Non-atomic: `~10000-50000` (very wrong!)
- Atomic: `1000000` (correct!)

### 11.3 Available Atomic Operations

**Integer atomics:**
| Function | Operation |
|----------|-----------|
| `atomicAdd(addr, val)` | `*addr += val` |
| `atomicSub(addr, val)` | `*addr -= val` |
| `atomicExch(addr, val)` | `*addr = val` (exchange) |
| `atomicMax(addr, val)` | `*addr = max(*addr, val)` |
| `atomicMin(addr, val)` | `*addr = min(*addr, val)` |
| `atomicAnd(addr, val)` | `*addr &= val` |
| `atomicOr(addr, val)` | `*addr \|= val` |
| `atomicXor(addr, val)` | `*addr ^= val` |
| `atomicCAS(addr, cmp, val)` | if `*addr == cmp` then `*addr = val` |

**Float atomics:**
- `atomicAdd(float* addr, float val)` — available from CUDA 2.0
- `atomicAdd(double* addr, double val)` — from Compute Capability 6.0

### 11.4 How Atomics Work Under the Hood

```
1. lock(memory_location)
2. old_value = *memory_location
3. *memory_location = old_value + increment
4. unlock(memory_location)
5. return old_value
```

### 11.5 Mutex Implementation in CUDA

```cuda
struct Mutex {
    int *lock;
};

__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // Spin-wait until lock is acquired
    }
}

__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    *counter = *counter + 1;  // Critical section
    unlock(m);
}
```

---

## 12. CUDA Streams & Concurrency

### 12.1 What Are Streams?

**Analogy: River Streams**
> Think of each stream as a river flowing forward in time. Operations in one stream happen sequentially, but **different streams flow simultaneously**. While one stream loads data, another can compute, and a third can copy results back.

```
Stream 1: [Copy A→GPU] [Compute on A] [Copy A result←GPU]
Stream 2:               [Copy B→GPU]  [Compute on B]     [Copy B result←GPU]
                                ↑ overlap! ↑
```

### 12.2 Default Stream

```cpp
// These are equivalent — both use the null/default stream (stream 0)
myKernel<<<gridSize, blockSize>>>(args);
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

### 12.3 Creating and Using Streams

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Async memory copies on different streams
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

// Launch kernel on stream1
vectorAdd<<<blocks, threads, 0, stream1>>>(d_A, d_B, d_C, N);

// Copy result back on stream1
cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);

// Synchronize
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// Cleanup
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### 12.4 Stream Priorities

```cuda
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority);
cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority);
```

### 12.5 Pinned (Page-Locked) Memory

**Analogy:** Regular memory is like a book on a library cart that can be moved at any time. Pinned memory is a book **bolted to the shelf** — the GPU always knows where to find it, making transfers faster.

```cpp
float* h_data;
cudaMallocHost((void**)&h_data, size);  // Pinned memory — faster transfers
// ... use it ...
cudaFreeHost(h_data);  // Free pinned memory
```

### 12.6 Events — Timing & Synchronization

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel took %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 12.7 Inter-Stream Dependencies with Events

```cuda
// Stream2 waits for stream1 to reach the event
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);
```

### 12.8 Callbacks

Execute CPU code when a stream operation completes:

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed!\n");
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```

---

## 13. CUDA APIs — cuBLAS

### 13.1 What Are CUDA APIs?

**Analogy: Using a Power Tool vs Building One**
> CUDA APIs are like professional power tools. You can't see inside them (opaque binaries), but they're extremely optimized. The documentation tells you which buttons to press. If you need something custom, you build your own tool (write your own kernel).

### 13.2 Error Checking Macros

```cpp
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}
```

### 13.3 cuBLAS — Matrix Multiplication

cuBLAS uses **column-major** order (like Fortran), not row-major (like C). You need to account for this!

```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;

// SGEMM: Single-precision (fp32) matrix multiplication
// Note: cuBLAS is column-major, so we swap A and B
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
    N, M, K,                    // Dimensions (swapped for col-major)
    &alpha,
    d_B, N,                     // B matrix
    d_A, K,                     // A matrix
    &beta,
    d_C, N);                    // C matrix (result)

cublasDestroy(handle);
```

### 13.4 HGEMM — Half Precision (fp16)

```cuda
#include <cuda_fp16.h>

// Convert float to half
half A_h[M * K];
for (int i = 0; i < M * K; i++)
    A_h[i] = __float2half(A[i]);

__half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N);

// Convert back to float
for (int i = 0; i < M * N; i++)
    result[i] = __half2float(C_h[i]);
```

### 13.5 cuBLAS Variants

| Variant | Purpose |
|---------|---------|
| **cuBLAS** | Standard BLAS on single GPU |
| **cuBLAS-Lt** | Lightweight — supports fp16/fp8/int8, deep learning focused |
| **cuBLAS-Xt** | Multi-GPU support — distributes across GPUs |
| **cuBLAS-Dx** | Device-side API — call BLAS from inside kernels |
| **CUTLASS** | Open-source templates — allows custom fusion |

> **cuBLAS-Lt note:** Dimensions M and K must be multiples of 4!

---

## 14. CUDA APIs — cuDNN

### 14.1 What cuDNN Provides

NVIDIA's Deep Neural Network library with optimized implementations of:

- Convolution (forward/backward)
- Pooling
- Softmax
- Activation functions (ReLU, tanh, sigmoid, GELU, etc.)
- Batch/Layer/Instance normalization
- Tensor transformations
- And more...

### 14.2 cuDNN Descriptor Pattern

cuDNN uses **opaque descriptor types** to describe tensors, operations, and algorithms:

```cuda
cudnnHandle_t cudnn;
cudnnCreate(&cudnn);

// Describe your input tensor
cudnnTensorDescriptor_t inputDesc;
cudnnCreateTensorDescriptor(&inputDesc);
cudnnSetTensor4dDescriptor(inputDesc,
    CUDNN_TENSOR_NCHW,      // Format: batch, channels, height, width
    CUDNN_DATA_FLOAT,        // Data type
    batch_size, channels, height, width);

// Describe the operation, algorithm, workspace...
// Then execute:
cudnnActivationForward(cudnn, activDesc,
    &alpha, inputDesc, d_input,
    &beta, outputDesc, d_output);

cudnnDestroy(cudnn);
```

### 14.3 cuDNN Tanh Example

```cuda
// Naive CUDA kernel for tanh
__global__ void naiveTanhKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// cuDNN will likely be faster for large tensors because it uses
// optimized, pre-compiled kernels with better memory access patterns
```

### 14.4 cuDNN Engine Types

| Engine Type | Description |
|-------------|-------------|
| Pre-compiled Single Op | Maximum performance, one operation |
| Generic Runtime Fusion | Dynamically fuse operations at runtime |
| Specialized Runtime Fusion | Optimized for specific patterns (e.g., Conv + ReLU) |
| Specialized Pre-compiled Fusion | Pre-compiled multi-op sequences (e.g., Conv + BN + ReLU) |

### 14.5 Graph API (cuDNN 8+)

Instead of fixed API calls, define a **computation graph** where:
- **Nodes** = Operations
- **Edges** = Tensors

This enables flexible kernel fusion without modifying source code.

---

## 15. Optimizing Matrix Multiplication

### 15.1 Optimization Hierarchy

From naive to expert:

```
Naive
  ↓ Coalesced Memory Access (load data in GPU-optimal order)
  ↓ Shared Memory / Tiling (reduce global memory reads)
  ↓ 1D/2D Block Tiling (distribute work evenly across SMs)
  ↓ Vectorized Memory Access (128-bit loads instead of 32-bit)
  ↓ Loop Unrolling (reduce loop overhead)
  ↓ Autotuning (grid search for optimal parameters)
  ↓ cuBLAS (NVIDIA's hand-tuned closed-source kernels)
```

### 15.2 Row Major vs Column Major

cuBLAS expects **column-major** format:

```python
# Row Major:  A[i][j] stored at A[i * N + j]
A = [[1, 2, 3],    # Memory: [1, 2, 3, 4, 5, 6, 7, 8, 9]
     [4, 5, 6],
     [7, 8, 9]]

# Column Major: A[i][j] stored at A[j * M + i]
# Memory: [1, 4, 7, 2, 5, 8, 3, 6, 9]
```

### 15.3 Loop Unrolling with `#pragma unroll`

**Analogy:** Instead of reading instructions one at a time from a checklist, you memorize the next 4 steps and do them all at once.

```cuda
// Without unrolling — compiler checks loop condition each iteration
__global__ void vectorAddNoUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float sum = 0;
        for (int j = 0; j < 100; j++) {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum;
    }
}

// With unrolling — compiler expands the loop body
__global__ void vectorAddUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float sum = 0;
        #pragma unroll
        for (int j = 0; j < 100; j++) {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum;
    }
}
```

Check the PTX assembly: `nvcc -ptx kernel.cu -o - | less`

### 15.4 Occupancy

$$\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Maximum possible warps per SM}}$$

Three limits to active blocks per SM:
1. **Register count** — each thread uses registers; too many → fewer threads
2. **Warp count** — hardware limit on concurrent warps
3. **Shared memory** — limited per SM; large blocks → fewer blocks

### 15.5 Assembly Instructions

- **PTX** — Parallel Thread Execution (virtual ISA, portable)
- **SASS** — Shader Assembly (actual hardware instructions)

View PTX: `nvcc -ptx kernel.cu`

---

## 16. Triton — Python GPU Programming

### 16.1 CUDA vs Triton Design Philosophy

| | CUDA | Triton |
|---|------|--------|
| Level | Scalar program + blocked threads | Blocked program + scalar threads |
| You write | Per-thread logic | Per-block logic |
| Memory mgmt | Manual (cudaMalloc, cudaMemcpy) | Automatic |
| Tiling | Manual | Compiler handles it |
| Language | C/C++ | Python |

**Analogy:** CUDA is like driving a manual transmission car — full control but complex. Triton is like driving an automatic — the compiler handles the gears (memory, tiling, caching) while you focus on the road (algorithm).

### 16.2 Triton Vector Addition

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    # Which block am I? (like blockIdx.x in CUDA)
    pid = tl.program_id(axis=0)
    
    # Calculate which elements this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Bounds mask (like the if(idx < n) in CUDA)
    mask = offsets < n_elements
    
    # Load, compute, store — entire blocks at once!
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 16.3 Triton Softmax

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, 
                   output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Load entire row into SRAM
    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE),
                  mask=tl.arange(0, BLOCK_SIZE) < n_cols,
                  other=-float('inf'))
    
    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store result
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(out_row_start_ptr + tl.arange(0, BLOCK_SIZE),
             softmax_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
```

### 16.4 Softmax — CPU Reference (C)

```c
void softmax(float *x, int n) {
    // Step 1: Find max (for numerical stability)
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    // Step 2: Exponentiate and sum
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - max);  // Subtract max to prevent overflow
        sum += x[i];
    }

    // Step 3: Normalize
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}
// Input:  [1.0, 2.0, 3.0]
// Exp:    [2.71, 7.39, 20.10]  (sum = 30.2)
// Output: [0.09, 0.24, 0.67]
```

### 16.5 Can You Skip CUDA and Only Learn Triton?

**No!** Because:
- Triton is an **abstraction on top of CUDA**
- You need CUDA concepts to understand what Triton does under the hood
- Custom optimizations may require raw CUDA
- Understanding threads, warps, shared memory, etc. is essential for debugging

---

## 17. Custom PyTorch CUDA Extensions

### 17.1 Why Build Extensions?

When PyTorch's built-in operations aren't fast enough, you can write a CUDA kernel and call it from Python!

### 17.2 The CUDA Kernel

```cuda
// polynomial_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void polynomial_activation_kernel(
    const scalar_t* __restrict__ x,    // __restrict__: promise no pointer aliasing
    scalar_t* __restrict__ output,
    size_t size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = x[idx];
        output[idx] = val * val + val + 1;  // x² + x + 1
    }
}

torch::Tensor polynomial_activation_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int threads = 1024;
    int blocks = (x.numel() + threads - 1) / threads;

    // AT_DISPATCH_FLOATING_TYPES handles float/double automatically
    AT_DISPATCH_FLOATING_TYPES(x.type(), "polynomial_activation_cuda", ([&] {
        polynomial_activation_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            x.numel()
        );
    }));
    return output;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polynomial_activation", &polynomial_activation_cuda,
          "Polynomial activation (CUDA)");
}
```

### 17.3 Key Concepts

**`scalar_t`** — A template type that gets compiled to the correct floating-point type (fp32 or fp64) based on the input tensor.

**`__restrict__`** — Tells the compiler that pointers don't overlap (alias), enabling aggressive optimization. Without it, the compiler must assume arrays might overlap and can't reorder operations.

**`AT_DISPATCH_FLOATING_TYPES`** — A PyTorch macro that generates code for both `float` and `double` types.

### 17.4 setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polynomial_cuda',
    ext_modules=[
        CUDAExtension('polynomial_cuda', ['polynomial_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

Build: `python setup.py install`

### 17.5 Using the Extension in Python

```python
import torch
import polynomial_cuda  # The compiled extension

class CUDAPolynomialActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return polynomial_cuda.polynomial_activation(x)

class PolynomialActivation(torch.nn.Module):
    def __init__(self, implementation='pytorch'):
        super().__init__()
        self.implementation = implementation

    def forward(self, x):
        if self.implementation == 'pytorch':
            return x**2 + x + 1
        elif self.implementation == 'cuda':
            return CUDAPolynomialActivation.apply(x)

# Benchmark
x = torch.randn(1000000, device='cuda')
pytorch_act = PolynomialActivation('pytorch').cuda()
cuda_act = PolynomialActivation('cuda').cuda()
```

---

## 18. Multi-GPU & Distributed Computing

### 18.1 cuBLAS-Mp (Multi-Process)

For computations that **don't fit on a single GPU**:
- Multi-process, multi-GPU BLAS on a single node
- Distributes matrix operations across GPUs (e.g., 8×H100s)

### 18.2 NCCL — Collective Communications

**Analogy:** cuBLAS-Mp is the workers doing the heavy lifting. NCCL is the **walkie-talkie system** coordinating them.

- All-reduce, broadcast, gather, scatter across multiple GPUs/nodes
- Used by PyTorch's DistributedDataParallel (DDP)
- Essential for training large models across clusters

**Types of parallelism:**
- **Data Parallelism** — Same model, different data batches on each GPU
- **Model Parallelism** — Model weights split across GPUs

### 18.3 MIG — Multi-Instance GPU

Slice a large GPU into smaller, independent virtual GPUs:
- Useful in datacenters where customers don't need full GPU capacity
- Each slice has its own memory and compute resources

---

## 19. Extras & Advanced Topics

### 19.1 Warp Divergence

**Analogy:** In a marching band, all members should be in sync. If one person does a different move (takes a different if/else branch), everyone else has to wait.

```cuda
// BAD: Different threads take different branches
if (threadIdx.x % 2 == 0) {
    // Even threads: complex work
} else {
    // Odd threads: simple work — but they WAIT for even threads
}
```

Within a warp (32 threads), if threads take different branches, they execute **serially** — the warp runs both paths and masks results. This is **warp divergence**. Vector addition is fast because there's no divergence.

### 19.2 Unified Memory

```cuda
// Managed memory — accessible from both CPU and GPU
int* data;
cudaMallocManaged(&data, N * sizeof(int));

// Use on CPU
for (int i = 0; i < N; i++) data[i] = i;

// Use on GPU directly — no explicit cudaMemcpy needed!
kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();

cudaFree(data);
```

**Pros:** Simpler code, automatic prefetching via streams
**Cons:** Can be slower than explicit memory management in some cases

### 19.3 Memory Architecture

- **DRAM** (global memory / VRAM) — capacitor + transistor cells, large but slow
- **SRAM** (shared memory / L1/L2 cache) — 6T or 8T transistor cells, small but fast
- 6T cells: compact, good performance
- 8T cells: better stability, lower power, larger area

### 19.4 Topics for Further Study

| Topic | Description |
|-------|-------------|
| **Quantization** | fp32 → fp16 → int8 for faster inference |
| **Tensor Cores** | Special hardware for matrix ops (WMMA) |
| **Sparsity** | Skip zero-valued elements for speedup |
| **Flash Attention** | Fused, memory-efficient attention mechanism |
| **CUTLASS** | Open-source CUDA templates for custom matmul |

---

## 20. CUDA Cheatsheet

### Memory Transfer

```cpp
cudaMalloc((void**)&d_arr, size);                                    // Allocate GPU memory
cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);              // CPU → GPU
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);              // GPU → CPU
cudaFree(d_arr);                                                      // Free GPU memory
cudaMallocHost((void**)&h_arr, size);                                 // Pinned host memory
cudaFreeHost(h_arr);                                                  // Free pinned memory
cudaMemcpyAsync(dst, src, size, kind, stream);                        // Async copy
```

### Kernel Launch

```cpp
__global__ void kernel(params) { /* ... */ }
kernel<<<gridDim, blockDim>>>(args);
kernel<<<gridDim, blockDim, sharedMemSize>>>(args);
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
```

### Thread Indexing

```cpp
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx_2d = row * width + col;

// 3D
int idx_3d = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x)
             * blockDim.x + threadIdx.x;
```

### Function Qualifiers

```cpp
__global__ void f() { }   // CPU calls, GPU runs (kernel)
__device__ void f() { }   // GPU calls, GPU runs (helper)
__host__   void f() { }   // CPU calls, CPU runs (normal)
__noinline__ void f() { } // Don't inline
__forceinline__ void f(){ }// Force inline
```

### Variable Qualifiers

```cpp
__shared__  float s;  // Shared within block (fast SRAM)
__device__  float d;  // Global device memory
__constant__ float c; // Read-only constant memory (cached)
__managed__  float m; // Unified Memory (CPU + GPU accessible)
```

### Synchronization

```cpp
__syncthreads();                        // Sync all threads in block
__syncthreads_and(predicate);           // Sync + all satisfy predicate?
__syncthreads_or(predicate);            // Sync + any satisfy predicate?
__syncthreads_count(predicate);         // Sync + count satisfying threads
__threadfence();                        // Device-wide memory fence
__threadfence_block();                  // Block-wide memory fence
__threadfence_system();                 // System-wide memory fence
cudaDeviceSynchronize();                // CPU waits for all GPU work
cudaStreamSynchronize(stream);          // CPU waits for stream
cudaStreamWaitEvent(stream, event);     // Stream waits for event
```

### Atomic Operations

```cpp
atomicAdd(addr, val);         atomicSub(addr, val);
atomicExch(addr, val);        atomicMin(addr, val);
atomicMax(addr, val);         atomicInc(addr, val);
atomicDec(addr, val);         atomicCAS(addr, compare, val);
atomicAnd(addr, val);         atomicOr(addr, val);
atomicXor(addr, val);
```

### Error Handling

```cpp
cudaError_t error = cudaGetLastError();
const char* msg = cudaGetErrorString(error);

#define CUDA_CHECK(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}
```

### Device Management

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, deviceId);   // Get GPU properties
cudaSetDevice(deviceId);                     // Select GPU
cudaGetDevice(&deviceId);                    // Get active GPU id
cudaGetDeviceCount(&count);                  // Total GPU count
cudaDeviceReset();                           // Reset GPU state
```

### Stream & Event Management

```cpp
cudaStreamCreate(&stream);
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
cudaStreamDestroy(stream);

cudaEventCreate(&event);
cudaEventRecord(event, stream);
cudaEventSynchronize(event);
cudaEventElapsedTime(&ms, start, stop);
cudaEventDestroy(event);
```

### Warp-Level Operations

```cpp
// Vote functions
__all_sync(mask, predicate);     // All threads satisfy?
__any_sync(mask, predicate);     // Any thread satisfies?
__ballot_sync(mask, predicate);  // Bit mask of satisfying threads

// Shuffle functions (exchange data within warp WITHOUT shared memory)
__shfl_sync(mask, var, srcLane);       // Get from specific lane
__shfl_up_sync(mask, var, delta);      // Get from lane - delta
__shfl_down_sync(mask, var, delta);    // Get from lane + delta
__shfl_xor_sync(mask, var, laneMask);  // Butterfly exchange
```

### Common Macros

```cpp
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define GRID_SIZE(n, b) ((n + b - 1) / b)
```

### Cache Configuration

```cpp
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);  // More shared mem
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);      // More L1 cache
```

### Template Kernels

```cpp
template<typename T>
__global__ void genericKernel(T* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2;
}

// Launch with specific type:
genericKernel<float><<<grid, block>>>(data);
```

---

## Compilation Quick Reference

```bash
# Compile CUDA program
nvcc -o output kernel.cu

# With architecture flag
nvcc -arch=sm_86 -o output kernel.cu

# With cuBLAS
nvcc -o output kernel.cu -lcublas

# With cuDNN
nvcc -o output kernel.cu -lcudnn

# With NVTX profiling
nvcc -o output kernel.cu -lnvToolsExt

# Profile with Nsight Systems
nsys profile --stats=true ./output

# Profile with Nsight Compute
ncu --kernel-name myKernel ./output

# Check for memory errors
compute-sanitizer ./output

# View PTX assembly
nvcc -ptx kernel.cu

# Build PyTorch extension
python setup.py install
```

---

> **Course Philosophy:** Lower the barrier to HPC, understand GPU bottlenecks (especially memory bandwidth), and build the foundation for projects like Karpathy's llm.c. Getting uncomfortable and breaking things is the best way to learn!
