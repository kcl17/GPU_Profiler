# CUDA eBPF Profiler with OpenTelemetry Integration

A prototype system for tracing CUDA Driver API calls (memory, kernels, transfers) using **eBPF uprobes** and exporting observability data as **OpenTelemetry spans**.  
Enables low-overhead profiling of GPU workloads with per-operation timing, error tracking, and span visualization.

## **Features**

- **eBPF tracing** of `cuMemAlloc`, `cuMemFree`, `cuMemcpy`, `cuLaunchKernel`, and more.
- **Dynamic symbol resolution** for CUDA driver library compatibility.
- **Per-operation latency & status** captured in OpenTelemetry spans.
- **Error injection & edge case detection** (double-free, OOM).
- **Test workload** with real GPU computation, memory transfer, and concurrency.
- **Intended for observability**, debugging, and performance analysis of production GPU workloads.

## **Prerequisites**

- **Linux x86_64** with kernel 5.8+ (for modern eBPF features).
- **NVIDIA GPU** with CUDA 11+ and compatible driver.
- **Go 1.18+** (for the collector).
- **OpenTelemetry SDK** (integrated in Go collector).
- **root privileges** (for eBPF uprobe attachment).
- **External tools**: `nm` or `objdump` (for dynamic symbol lookup).

## **Quick Start**

1. **Compile the CUDA test kernel:**

``` nvcc --cubin --gpu-architecture=sm_61 -o vecAdd.cubin vecAdd.cu ```

2. **Build the CUDA test application:**

``` nvcc cuda_advanced.cu -o cuda_advanced -lcuda   ```

3. **Build and run the profiler:**

```CGO_LDFLAGS="-lbpf -lz" go build -o cuda-profiler```
```sudo ./cuda-profiler```


4. **Run the CUDA workload** (in a separate terminal):

```./cuda_advanced     ```

5. **View spans** in your OpenTelemetry-compatible collector (Jaeger, Tempo, CLI).

## **Project Structure**
```
    ├── bpf
    │   ├── bpf_trace.bpf.c
    │   └── vmlinux.h
    ├── build
    │   └── cuda_trace.bpf.o
    ├── cuda-profiler
    ├── go.mod
    ├── go.sum
    ├── main.go
    ├── makefile
    ├── Readme.md
    └── test
        ├── cuda_advanced
        ├── cuda_advanced.cu
        ├── cuda_ebpf
        ├── cuda_ebpf.cu
        ├── vecAdd.cu
        └── vecAdd.cubin
```


## **Configuration**

- **CUDA library path** is auto-detected; override via environment if needed.
- **OpenTelemetry endpoint** can be configured in the Go collector.
- **Symbol resolution** defaults to `/usr/lib/x86_64-linux-gnu/libcuda.so.1`.

## **Contributing**

- **Report issues** on GitHub.
- **Extend functionality** for new CUDA APIs, multi-GPU support, or advanced tracing.
- **Contribute dashboards** for Grafana, Tempo, or Jaeger.


