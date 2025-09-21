#include <iostream>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define CU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true) {
    if (code != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(code, &err_str);
        fprintf(stderr, "CUDA Driver Error %d: %s at %s:%d\n", code, err_str, file, line);
        if (abort) exit(code);
    }
}

// Simple vector add kernel, mapped to CUDA Driver API
extern "C" __global__ void vecAdd(float *a, float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

int main(int argc, char **argv) {
    srand(time(0));
    int n_iter = (argc > 1) ? atoi(argv[1]) : 5;
    size_t max_alloc_size = (argc > 2) ? atoi(argv[2]) : 10*1024*1024;
    int seed = (argc > 3) ? atoi(argv[3]) : 42;

    CUdevice device;
    CUcontext ctx;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuCtxCreate(&ctx, 0, device));

    // Create a few CUDA streams for concurrency
    CUstream streams[2];
    for (int i = 0; i < 2; ++i) {
        CU_CHECK(cuStreamCreate(&streams[i], 0));
    }

    for (int i = 0; i < n_iter; ++i) {
        // Randomize allocation size and pattern
        size_t alloc_size = (rand() % (max_alloc_size - 1024)) + 1024;
        size_t n_elems = alloc_size / sizeof(float);
        bool use_pinned = rand() % 2;
        bool inject_oom = (i == n_iter - 1); // inject OOM on last iteration
        bool inject_double_free = (i == 1);  // inject double-free to test profiler

        // Allocate device memory
        CUdeviceptr d_a, d_b, d_out;
        CU_CHECK(cuMemAlloc(&d_a, alloc_size));
        CU_CHECK(cuMemAlloc(&d_b, alloc_size));
        CU_CHECK(cuMemAlloc(&d_out, alloc_size));

        // Allocate host memory (optionally pinned)
        void *h_a = use_pinned ? nullptr : malloc(alloc_size);
        void *h_b = use_pinned ? nullptr : malloc(alloc_size);
        void *h_out = use_pinned ? nullptr : malloc(alloc_size);

        if (use_pinned) {
            CU_CHECK(cuMemHostAlloc(&h_a, alloc_size, CU_MEMHOSTALLOC_DEVICEMAP));
            CU_CHECK(cuMemHostAlloc(&h_b, alloc_size, CU_MEMHOSTALLOC_DEVICEMAP));
            CU_CHECK(cuMemHostAlloc(&h_out, alloc_size, CU_MEMHOSTALLOC_DEVICEMAP));
        }

        // Fill host buffers with random data
        for (size_t j = 0; j < n_elems; ++j) {
            ((float*)h_a)[j] = (float)rand() / RAND_MAX;
            ((float*)h_b)[j] = (float)rand() / RAND_MAX;
        }

        // Copy to device (random stream)
        CUstream strm = streams[rand() % 2];
        CU_CHECK(cuMemcpyHtoDAsync(d_a, h_a, alloc_size, strm));
        CU_CHECK(cuMemcpyHtoDAsync(d_b, h_b, alloc_size, strm));

        // Load module, get kernel
        CUmodule module;
        CU_CHECK(cuModuleLoad(&module, "vecAdd.cubin"));
        CUfunction vecAdd_kernel;
        CU_CHECK(cuModuleGetFunction(&vecAdd_kernel, module, "vecAdd"));

        // Setup kernel launch arguments
        float *d_a_float = reinterpret_cast<float*>(d_a);
        float *d_b_float = reinterpret_cast<float*>(d_b);
        float *d_out_float = reinterpret_cast<float*>(d_out);
        int n = static_cast<int>(n_elems); // explicit conversion to int
        void *kernel_args[] = { &d_a_float, &d_b_float, &d_out_float, &n };

        // Launch kernel
        dim3 block(256);
        dim3 grid((n_elems + block.x - 1) / block.x);
        CU_CHECK(cuLaunchKernel(vecAdd_kernel, grid.x, grid.y, 1, block.x, 1, 1, 0, strm, kernel_args, NULL));

        // Unload module
        CU_CHECK(cuModuleUnload(module));

        // Copy result back (async)
        CU_CHECK(cuMemcpyDtoHAsync(h_out, d_out, alloc_size, strm));

        // Synchronize stream
        CU_CHECK(cuStreamSynchronize(strm));

        // Free device memory
        CU_CHECK(cuMemFree(d_a));
        CU_CHECK(cuMemFree(d_b));
        CU_CHECK(cuMemFree(d_out));

        // Free host memory
        if (use_pinned) {
            CU_CHECK(cuMemFreeHost(h_a));
            CU_CHECK(cuMemFreeHost(h_b));
            CU_CHECK(cuMemFreeHost(h_out));
        } else {
            free(h_a); free(h_b); free(h_out);
        }

        // Optionally inject OOM or double-free for testing
        if (inject_oom) {
            CUdeviceptr dummy;
            printf("Testing allocation failure (OOM):\n");
            CUresult res = cuMemAlloc(&dummy, 1ULL << 40); // Attempt 1TB allocation
            printf("cuMemAlloc returned: %d\n", res);
        } else if (inject_double_free) {
            printf("Testing double-free of pointer d_a\n");
            CU_CHECK(cuMemFree(d_a)); // This will crash, but shows up in profiler logs
        }
    }

    // Clean up
    CU_CHECK(cuCtxDestroy(ctx));
    std::cout << "CUDA App Finished" << std::endl;
    return 0;
}
