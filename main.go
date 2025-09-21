package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/aquasecurity/libbpfgo"
	"github.com/honeycombio/otel-config-go/otelconfig"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	oteltrace "go.opentelemetry.io/otel/trace"
)

// Implement an HTTP Handler func to be instrumented
func httpHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, World")
}

// Wrap the HTTP handler func with OTel HTTP instrumentation
func wrapHandler() {
	handler := http.HandlerFunc(httpHandler)
	wrappedHandler := otelhttp.NewHandler(handler, "hello")
	http.Handle("/hello", wrappedHandler)
}

// Event defines the structure sent from your BPF program via the perf buffer.
type Event struct {
	Latency uint64
}

// CUDALibraryPaths are common CUDA library locations — auto-discovered at runtime.
var CUDALibraryPaths = []string{
	"/usr/lib/x86_64-linux-gnu/libcuda.so.1", // Ubuntu/Debian system install
	"/usr/lib/x86_64-linux-gnu/libcuda.so",   // Alternative system path
	"/usr/local/cuda/lib64/libcuda.so",       // Standard CUDA install
	"/usr/local/cuda-11.5/lib64/libcuda.so",  // Version-specific path
	"/usr/lib/nvidia/libcuda.so.1",           // NVIDIA-specific location
	"/opt/cuda/lib64/libcuda.so",             // Alternative install location
	"/usr/lib64/nvidia/libcuda.so.1",         // 64-bit specific path
}

// findCUDALibrary searches for libcuda.so.1 on the system.
func findCUDALibrary() string {
	for _, path := range CUDALibraryPaths {
		if _, err := os.Stat(path); err == nil {
			fmt.Printf("Found CUDA library at: %s\n", path)
			return path
		}
	}
	// Fallback to Debian/Ubuntu default
	return "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
}

// getSymbolOffsetCmd resolves the offset of a symbol in a shared library using `nm`.
// Returns (offset, nil) on success, (0, error) on failure.
func getSymbolOffsetCmd(libPath, symbol string) (uint64, error) {
	cmd := exec.Command("nm", "-D", "--defined-only", libPath)
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("failed to run nm: %w", err)
	}

	lines := bytes.Split(output, []byte{'\n'})
	for _, line := range lines {
		if bytes.Contains(line, []byte(" T "+symbol)) {
			fields := strings.Fields(string(line))
			if len(fields) < 3 {
				continue
			}
			offset, err := strconv.ParseUint(fields[0], 16, 64)
			if err != nil {
				return 0, fmt.Errorf("failed to parse offset: %w", err)
			}
			return offset, nil
		}
	}
	return 0, fmt.Errorf("symbol %s not found in %s", symbol, libPath)
}

// getSymbolOffset is the top-level symbol resolver.
// Tries nm first, falls back to Go debug/elf (usually fails on stripped libcuda.so.1).
// Finally, accepts a manually provided offset via env var CUDA_SYMBOL_OFFSET.
func getSymbolOffset(libPath, symbol string) (uint64, error) {
	// 1. Try nm -D
	offset, err := getSymbolOffsetCmd(libPath, symbol)
	if err == nil {
		return offset, nil
	}
	log.Printf("warning: nm failed for symbol resolution: %v", err)

	// 2. Try Go debug/elf package (usually fails on stripped libcuda.so.1)
	/* Uncomment only if you know your libcuda.so.1 is not stripped or corrupted
	file, err := elf.Open(libPath)
	if err != nil {
		log.Printf("warning: failed to open ELF file: %v", err)
	} else {
		defer file.Close()
		syms, err := file.Symbols()
		if err != nil {
			log.Printf("warning: failed to read ELF symbols: %v", err)
		} else {
			for _, sym := range syms {
				if sym.Name == symbol {
					return sym.Value, nil
				}
			}
		}
	}
	*/

	// 3. Fallback to manual offset from env var
	if manualOffset := os.Getenv("CUDA_SYMBOL_OFFSET"); manualOffset != "" {
		offset, err = strconv.ParseUint(manualOffset, 0, 64)
		if err != nil {
			return 0, fmt.Errorf("failed to parse CUDA_SYMBOL_OFFSET: %w", err)
		}
		log.Printf("warning: using manually provided symbol offset 0x%x for %s", offset, symbol)
		return offset, nil
	}

	return 0, fmt.Errorf("symbol %s not found in %s and no manual offset provided", symbol, libPath)
}

// setupTracer configures OpenTelemetry tracing with resource attributes.
func setupTracer() (*trace.TracerProvider, error) {
	tp := trace.NewTracerProvider(
		trace.WithResource(createResource()),
		trace.WithSampler(trace.AlwaysSample()),
	)
	otel.SetTracerProvider(tp)
	return tp, nil
}

// createResource defines OpenTelemetry resource attributes.
func createResource() *resource.Resource {
	return resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceNameKey.String("cuda-profiler"),
		semconv.ServiceVersionKey.String("1.0.0"),
		attribute.String("gpu.model", "NVIDIA GeForce MX330"),
		attribute.String("driver.version", "575.64.03"),
		attribute.String("cuda.version", "11.5.119"),
	)
}

func main() {
	// Create root context for spans
	ctx := context.Background()

	// Use otelconfig to setup OpenTelemetry SDK
	otelShutdown, err := otelconfig.ConfigureOpenTelemetry()
	if err != nil {
		log.Fatalf("error setting up OTel SDK - %v", err)
	}
	defer otelShutdown()

	// Initialize HTTP handler instrumentation
	wrapHandler()
	go func() {
		log.Println("Starting HTTP server at :3030")
		log.Fatal(http.ListenAndServe(":3030", nil))
	}()

	// Setup OpenTelemetry tracer provider for custom spans
	tp, err := setupTracer()
	if err != nil {
		panic(fmt.Errorf("failed to setup tracer: %w", err))
	}
	defer func() {
		if err := tp.Shutdown(ctx); err != nil {
			fmt.Printf("error shutting down tracer: %v\n", err)
		}
	}()

	tracer := tp.Tracer("cuda-profiler")

	// Load BPF object
	bpfPath := "build/cuda_trace.bpf.o"
	bpfModule, err := libbpfgo.NewModuleFromFile(bpfPath)
	if err != nil {
		panic(fmt.Errorf("failed to load BPF object: %w", err))
	}
	defer bpfModule.Close()

	// Load BPF program
	if err = bpfModule.BPFLoadObject(); err != nil {
		panic(fmt.Errorf("failed to load BPF program: %w", err))
	}

	// Find CUDA library
	cudaLibPath := findCUDALibrary()

	// Resolve symbol offset for cuMemAlloc_v2
	symbol := "cuMemAlloc_v2"
	offset, err := getSymbolOffset(cudaLibPath, symbol)
	if err != nil {
		log.Fatalf("failed to get symbol offset for %s: %v", symbol, err)
	}
	fmt.Printf("Resolved %s offset: %d (0x%x)\n", symbol, offset, offset)

	// Attach uprobes
	entryProg, err := bpfModule.GetProgram("trace_entry")
	if err != nil {
		log.Fatalf("failed to get BPF entry program: %v", err)
	}
	if _, err = entryProg.AttachUprobe(-1, cudaLibPath, uint32(offset)); err != nil {
		log.Fatalf("failed to attach uprobe: %v", err)
	}

	retProg, err := bpfModule.GetProgram("trace_return")
	if err != nil {
		log.Fatalf("failed to get BPF return program: %v", err)
	}
	if _, err = retProg.AttachURetprobe(-1, cudaLibPath, uint32(offset)); err != nil {
		log.Printf("warning: failed to attach uretprobe: %v", err)
	}

	// Initialize perf buffer for event collection
	eventsChannel := make(chan []byte, 256)
	perfMap, err := bpfModule.InitPerfBuf("events", eventsChannel, nil, 1024)
	if err != nil {
		panic(fmt.Errorf("failed to initialize perf buffer: %w", err))
	}
	perfMap.Start()
	defer func() {
		perfMap.Stop()
		perfMap.Close()
	}()

	fmt.Println("CUDA eBPF profiler started successfully!")
	fmt.Printf("Monitoring %s operations...\n", symbol)
	fmt.Println("Press Ctrl+C to stop.")

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Event processing loop
	for {
		select {
		case data := <-eventsChannel:
			var evt Event
			if err = binary.Read(bytes.NewBuffer(data), binary.LittleEndian, &evt); err != nil {
				fmt.Printf("Failed to parse event: %v\n", err)
				continue
			}

			fmt.Printf("CUDA Memory Allocation Latency: %d ns\n", evt.Latency)

			// Create OpenTelemetry span
			spanCtx, span := tracer.Start(ctx, symbol,
				oteltrace.WithAttributes(
					attribute.Int64("cuda.latency.ns", int64(evt.Latency)),
					attribute.String("cuda.function", symbol),
					attribute.String("gpu.model", "NVIDIA GeForce MX330"),
					attribute.String("driver.version", "575.64.03"),
				),
			)

			// Simulate processing delay for better span visualization
			time.Sleep(500 * time.Microsecond)

			span.End()
			ctx = spanCtx

		case <-sigChan:
			fmt.Println("Received shutdown signal. Exiting...")
			return
		}
	}
}
