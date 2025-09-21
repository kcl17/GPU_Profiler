// cuda_app.cu
#include <iostream>
#include <cstring>
#include <cassert>
#include <cuda.h>

#define CU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true) {
    if (code != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(code, &err_str);
        fprintf(stderr, "CUDA Driver Error %d: %s at %s:%d\n", code, err_str, file, line);
        if (abort) exit(code);
    }
}

__global__ void dummyKernel() {}

int main(int argc, char **argv) {
    size_t size = 1024 * 1024; // default 1MB
    int n_iter = 5;             // default 5 iterations
    if (argc > 1) {
        int user_size = std::stoi(argv[1]);
        if (user_size > 0) {
            size = user_size;
            std::cout << "Using user-provided size: " << size << " bytes" << std::endl;
        }
    }
    if (argc > 2) {
        n_iter = std::stoi(argv[2]);
        std::cout << "Using user-provided iteration count: " << n_iter << std::endl;
    }

    std::cout << "Starting with size=" << size << "B, n_iter=" << n_iter << std::endl;

    // Initialize CUDA driver API
    std::cout << "Initializing CUDA..." << std::endl;
    CU_CHECK(cuInit(0));

    // Get device 0 and create a context
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, device));
    std::cout << "Created CUDA context" << std::endl;

    for (int i = 0; i < n_iter; ++i) {
        CUdeviceptr devPtr;
        // Allocate device memory (+print)
        std::cout << i << ": Allocating " << size << "B on device..." << std::flush;
        CU_CHECK(cuMemAlloc(&devPtr, size));
        std::cout << " ptr=" << devPtr << std::endl;

        // Launch a dummy kernel (with cudaDeviceSynchronize to ensure timing is visible)
        std::cout << i << ": Launching dummyKernel..." << std::flush;
        dummyKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        std::cout << " kernel launched" << std::endl;

        // Free device memory (+print)
        std::cout << i << ": Freeing " << size << "B @ 0x" << std::hex << devPtr << std::dec << std::endl;
        CU_CHECK(cuMemFree(devPtr));
    }

    // Destroy context
    CU_CHECK(cuCtxDestroy(ctx));
    std::cout << "CUDA App Finished" << std::endl;
    return 0;
}
