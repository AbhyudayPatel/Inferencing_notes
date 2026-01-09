import modal
import subprocess

# 1. DEFINE THE IMAGE
# We need an image that includes the NVIDIA CUDA compiler (nvcc).
# The "devel" tag is crucial; "runtime" images only let you run apps, not compile them.
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
)

app = modal.App("cuda-learner")

# 2. DEFINE THE CUDA CODE
# We'll store the C code in a Python string for simplicity.
cuda_code = r"""
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void cuda_hello() {
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Standard C: Host is preparing to launch kernel...\n");
    
    // Launch kernel with 1 block and 5 threads
    cuda_hello<<<1, 5>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("Standard C: Kernel finished.\n");
    return 0;
}
"""

# 3. DEFINE THE MODAL FUNCTION
@app.function(image=cuda_image, gpu="T4")
def run_cuda_script():
    # Step A: Write the C code to a file inside the container
    with open("hello.cu", "w") as f:
        f.write(cuda_code)

    print("--- Compiling ---")
    # Step B: Compile using nvcc
    # This runs the shell command: nvcc -o hello hello.cu
    compile_process = subprocess.run(
        ["nvcc", "-o", "hello", "hello.cu"], 
        capture_output=True, 
        text=True
    )

    if compile_process.returncode != 0:
        print("Compilation Failed:")
        print(compile_process.stderr)
        return

    print("--- Running ---")
    # Step C: Run the compiled binary
    # This runs the shell command: ./hello
    run_process = subprocess.run(
        ["./hello"], 
        capture_output=True, 
        text=True
    )
    
    # Print the output from the C program
    print(run_process.stdout)
    if run_process.stderr:
        print("Errors:", run_process.stderr)

# 4. ENTRYPOINT
@app.local_entrypoint()
def main():
    run_cuda_script.remote()