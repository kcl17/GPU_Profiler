// File: vecAdd.cu
#include <cuda.h>
extern "C"
__global__ void vecAdd(float *a, float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] + b[idx];
}
