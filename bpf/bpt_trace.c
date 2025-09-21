#include "vmlinux.h"     // For kernel types (BPF CO-RE)
#include <bpf/bpf_helpers.h>

/* Event sent from BPF to userspace */
struct event {
    u64 pid;     // Process ID calling cuMemAlloc
    u64 latency; // Entry-to-exit latency in nanoseconds
    char comm[16]; // Process name (for debugging)
};

/* BPF Perf Event Map (for userspace collection) */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u32));
} events SEC(".maps");

/* Map to store u64 entry timestamps, keyed by pid */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __uint(key_size, sizeof(u64));  // pid_tgid
    __uint(value_size, sizeof(u64)); // Entry time (ns)
} entry_times SEC(".maps");

/* Entry and exit functions using SEC macros; libbpf will auto-attach */

SEC("uprobe/cuMemAlloc")
int trace_entry(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&entry_times, &pid_tgid, &ts, BPF_ANY);
    return 0;
}

SEC("uretprobe/cuMemAlloc")
int trace_return(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *entry_ts = bpf_map_lookup_elem(&entry_times, &pid_tgid);
    if (!entry_ts) return 0; // No matching entry

    u64 exit_ts = bpf_ktime_get_ns();
    if (exit_ts <= *entry_ts) return 0; // Clock anomaly

    // Prepare event
    struct event evt = {
        .pid = pid_tgid >> 32, // Process ID only (not thread ID)
        .latency = exit_ts - *entry_ts,
    };
    // Capture process name for debugging/disambiguation
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));

    // Send event to userspace Go program via perf buffer
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &evt, sizeof(evt));

    // Clean up
    bpf_map_delete_elem(&entry_times, &pid_tgid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
