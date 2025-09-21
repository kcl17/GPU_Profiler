BPF_CLANG ?= clang
BPF_LLVM_STRIP ?= llvm-strip
KERNEL_HEADERS := /lib/modules/$(shell uname -r)/build
BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_x86

BPF_SRC := bpf/bpf_trace.bpf.c
BPF_OBJ := build/cuda_trace.bpf.o

.PHONY: all clean

all: $(BPF_OBJ)

$(BPF_OBJ): $(BPF_SRC)
	@mkdir -p build
	$(BPF_CLANG) $(BPF_CFLAGS) \
		-I$(KERNEL_HEADERS)/include \
		-I$(KERNEL_HEADERS)/include/uapi \
		-I$(KERNEL_HEADERS)/include/generated \
		-Ibpf \
		-c $< -o $@
	$(BPF_LLVM_STRIP) -g $@

clean:
	rm -rf build/*.o
	rm -rf build/*.bpf.o