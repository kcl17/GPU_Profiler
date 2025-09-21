Here is your fully compiled, professional-quality `TROUBLESHOOTING.md` for the **CUDA eBPF Profiler**, cleaned and ready for direct use in **Canva**, **GitHub**, **Notion**, or documentation platforms:

---


# 🔧 CUDA eBPF Profiler — Troubleshooting Guide

Detailed error scenarios, root causes, and proven fixes encountered during the development and deployment of the **CUDA eBPF Profiler**.

---

## 🐛 Symptom: Profiler sees no events from CUDA app

### 📌 Description:
The Go profiler starts, attaches uprobes, but **no events are logged** even after running the test CUDA application.

### 🧠 Root Causes:
- ❌ Symbol mismatch: e.g. using `cuMemAlloc` instead of `cuMemAlloc_v2`
- ❌ `libcuda.so.1` is stripped — symbols can't be parsed via Go's ELF parser
- ❌ `AttachUprobe` silently fails without error
- ❌ Perf buffer poll loop missing or not triggered

### ✅ Fixes:
- Run:
  ```bash
  nm -D /usr/lib/x86_64-linux-gnu/libcuda.so.1 | grep cuMemAlloc


Use the exact exported symbol (like `cuMemAlloc_v2`).

* Replace Go’s ELF parser with a shell-out to `nm`.
* Log and handle errors when `AttachUprobe` fails.
* Ensure your Go code includes:

  ```go
  events.Poll(1000)
  ```

---

## ❌ ERROR: `failed to load BPF object: no such file`

### 💡 Cause:

Compiled BPF object `cuda_trace.bpf.o` is missing.

### ✅ Fix:

```bash
make
```

Ensure this path exists:

```bash
ls build/cuda_trace.bpf.o
```

---

## ❌ ERROR: `Operation not permitted` (BPF loading)

### 💡 Cause:

Your system is not allowing BPF programs due to memory lock or kernel config.

### ✅ Fixes:

#### 1. Confirm Kernel Support

```bash
grep CONFIG_BPF_SYSCALL /boot/config-$(uname -r)
# Output should be: CONFIG_BPF_SYSCALL=y
```

#### 2. Set Memory Lock Limits

Edit:

```bash
sudo nano /etc/security/limits.conf
```

Add at bottom:

```
ramanujan soft memlock unlimited
ramanujan hard memlock unlimited
```

Then:

```bash
sudo nano /etc/pam.d/common-session
```

Add:

```
session required pam_limits.so
```

Reboot and verify:

```bash
ulimit -l
# Output should be: unlimited
```

---

## ❌ ERROR: `sudo: go: command not found`

### 💡 Cause:

`go` binary is not in sudo's secure PATH.

### ✅ Fix:

1. Check path:

   ```bash
   which go
   # Example: /usr/local/go/bin/go
   ```

2. Update `visudo`:

   ```bash
   sudo visudo
   ```

   Update:

   ```
   Defaults secure_path="/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
   ```

---

## 🐞 Symptom: CUDA Driver Error 301 — File not found

### 📌 Description:

Error from `cuModuleLoad()` about missing `.cubin`.

### ✅ Fix:

Compile your `.cubin` file:

```bash
nvcc --cubin --gpu-architecture=sm_61 -o vecAdd.cubin vecAdd.cu
```

Ensure it exists:

```bash
ls vecAdd.cubin
```

---

## 🐞 Symptom: CUDA Error 1 — Invalid argument

### 📌 Description:

Test app crashes due to double-free or incorrect pointer use.

### ✅ Fix:

* Never call `cuMemFree` twice on same pointer.
* Check return values from all CUDA Driver API calls.

---

## ⚠️ Symptom: Go ELF parser fails — "no symbol section"

### 💡 Cause:

`libcuda.so.1` is stripped and Go’s `debug/elf` fails.

### ✅ Fix:

* Replace Go ELF parsing with:

  ```bash
  nm -D --defined-only /usr/lib/x86_64-linux-gnu/libcuda.so.1
  ```
* Use symbol name APIs instead of offset-based attach.

---

## ⚠️ Symptom: Profiler runs but shows no variation

### 📌 Description:

Output is static; same latency each run.

### 💡 Cause:

* Toy workload (e.g., constant 1MB alloc/free loop)
* No real GPU work, compute, or data movement

### ✅ Fix:

* Vary workload:

  * Random allocation sizes
  * Pinned vs pageable memory
  * Concurrent streams
* Add real GPU compute:

  * Vector addition, matrix multiply
* Include errors (e.g., double-free, large allocs)

---

## ✅ Summary: Most Common Fixes

| Issue                         | Fix                                                              |
| ----------------------------- | ---------------------------------------------------------------- |
| `cuda_trace.bpf.o` missing    | Run `make`, check `build/`                                       |
| Operation not permitted       | Set memlock unlimited in `limits.conf`, add `pam_limits`, reboot |
| `go` not found with sudo      | Add Go binary path in `visudo` secure\_path                      |
| No profiler output            | Use `AttachUprobeSymbol()` and check symbol with `nm`            |
| Stripped libcuda.so ELF error | Use `nm -D` instead of `debug/elf` in Go                         |

---

## 📚 References

* [NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
* [BCC Documentation](https://github.com/iovisor/bcc)
* [libbpfgo by Aqua Security](https://github.com/aquasecurity/libbpfgo)
* [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
* [Linux ulimit and RLIMIT\_MEMLOCK](https://man7.org/linux/man-pages/man2/setrlimit.2.html)

---

## 🆘 Where to Get Help

* Check logs from `dmesg` for BPF load errors
* Validate symbol names with `nm`
* Log all attachment and polling errors in your Go code
* Open issues in your GitHub repo with logs and environment info

```

---

Let me know if you want:

- A **PDF version**
- This converted into a **Canva-ready layout**
- The same as a **Notion template**

Would you also like the corresponding `README.md` cleaned and synced with this?
```
