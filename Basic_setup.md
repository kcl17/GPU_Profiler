## ✅ GPU System Setup for CUDA + eBPF Tracing (Linux)

This README guides you through **setting up your Linux system for GPU tracing using eBPF and CUDA**, starting from bare metal (no GPU modules preloaded).

---

### 🧭 Step 1: Check if You Have an NVIDIA GPU

Run:

```bash
lspci | grep -i nvidia
```

Example output:

```
01:00.0 VGA compatible controller: NVIDIA Corporation GP108M [GeForce MX330] (rev a1)
```
```
# Human-readable verbose entry for each VGA device
lspci -v -s $(lspci | grep -i ' vga ' | cut -d' ' -f1) | less[8]

# Kernel has created display-class devices?
lshw -C display | less[3]
```
If this returns nothing, your system may not have an NVIDIA GPU.

---

### ⚙️ Step 2: Install GPU Drivers & Kernel Modules

> This is **required before using `nvidia-smi`, `nvcc`, or loading CUDA kernels**.

#### 🛠️ Install NVIDIA proprietary drivers:

Ubuntu:

```bash
sudo apt update
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-<535> 
```

> Replace `535` with the recommended version shown.

#### ✅ Check that kernel modules load:

```bash
lsmod | grep nvidia
```

If nothing shows up:

```bash
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm
```

> If `modprobe` fails, you may need to **disable Secure Boot** (see below).

---

### 🚫 Disable Secure Boot (if needed)

Some NVIDIA drivers and eBPF programs won’t work unless Secure Boot is disabled.

1. Reboot into BIOS/UEFI
2. Disable **Secure Boot**
3. Save & exit

---

### 🔁 Reboot Now

This ensures all kernel modules and NVIDIA drivers are correctly loaded:

```bash
sudo reboot
```

---

### 🔍 Step 3: Verify GPU Installation

Now run:

```bash
nvidia-smi
```

Expected output: GPU info table with name, memory, driver version.

---

### 🔎 Step 4: Check Detailed GPU & CUDA Info

#### 📋 GPU capabilities:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,utilization.gpu --format=csv
```

#### 📦 CUDA compiler version:

```bash
nvcc --version
```

#### 🎯 List available CUDA devices (SM architectures):

```bash
nvcc --list-gpus
```

#### 📌 Driver kernel module version:

```bash
modinfo nvidia | grep version
```

---

### 🧠 Step 5: Enable Huge Pages & eBPF Requirements

#### ✅ Enable BPF syscall:

```bash
grep CONFIG_BPF_SYSCALL /boot/config-$(uname -r)
# Should return: CONFIG_BPF_SYSCALL=y
```

If not present, you may need to upgrade or recompile your kernel.

#### 🔓 Set RLIMIT\_MEMLOCK:

Edit:

```bash
sudo nano /etc/security/limits.conf
```

Add:

```
<your-username> soft memlock unlimited
<your-username> hard memlock unlimited
```

Then edit PAM limits:

```bash
sudo nano /etc/pam.d/common-session
```

Add at the end:

```
session required pam_limits.so
```

Now reboot:

```bash
sudo reboot
```

Verify:

```bash
ulimit -l
# Should return: unlimited
```
### Quick Helm Chart For gpustat-exporter
```
replicaCount: 1
image:
  repository: fstab/gpustat-exporter
  tag: latest
  pullPolicy: IfNotPresent
securityContext:
  privileged: true
env:
- name: INTERVAL
  value: "5"
service:
  port: 9102
```
> Deploys a Prometheus-compatible /metrics endpoint with per-GPU gauges (utilisation, mem_used, ecc_errors) sourced from gpustat.
---

### 🧰 Optional: Helpful Tools

* **nvidia-modprobe** — ensures modules are autoloaded
* **nvidia-utils** — includes `nvidia-smi`, `nvidia-settings`, etc.
* **cuda-toolkit** — for `nvcc`, `cuobjdump`, etc.
* **perf** — for performance counters and BPF profiling

Install:

```bash
sudo apt install nvidia-utils-535 nvidia-modprobe nvidia-cuda-toolkit linux-tools-common linux-tools-$(uname -r)
```
---

### 🛡️ Good Practices for Future Debugging

* Always check `dmesg | grep -i nvidia` for driver load errors
* Use `lsmod | grep nvidia` after reboots
* Backup `/etc/modprobe.d/` and `/etc/X11/xorg.conf` if you manually edit configs
* Track CUDA & kernel version compatibility matrix from [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

---

