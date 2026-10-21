/* ds4_gpu_roles.h — device scoring and role assignment for heterogeneous
 * NVIDIA multi-GPU.
 *
 * This is the decision plane for the optimizations.md plan: probe every
 * configured device, normalize the metrics, score each device for prefill,
 * decode, and capacity, then assign one primary (dense weights), one cache
 * (expert pool / long-context KV), and one secondary (prefill helper) role.
 *
 * The implementation is backend-neutral C so it can be unit-tested with
 * synthetic probes. The actual attribute read is a thin CUDA shim declared in
 * ds4_gpu_args.h (ds4_gpu_args_probe_devices) so this module never includes
 * <cuda_runtime.h>.
 *
 * Graceful degradation is mandatory: fewer than two usable devices, or an
 * explicit --disable-multi-gpu, resolves to single_gpu_mode and the engine's
 * existing single-device path, unchanged.
 *
 * Plan: optimizations.md §1 (scoring), §6.4 (chunk), §11.2/§11.3 (log, flags).
 */
#ifndef DS4_GPU_ROLES_H
#define DS4_GPU_ROLES_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_gpu_mgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* §1.1: per-device metrics. Byte units are bytes, bandwidth is GB/s
 * (1e9, not GiB/s) because PCIe/memory bandwidth is quoted that way. */
typedef struct {
    int      device_id;        /* CUDA visible device index */
    char     name[64];
    int      pcie_gen;
    int      pcie_width;
    double   pcie_bw_gbps;     /* measured pinned D2H+H2D */
    uint64_t vram_total_bytes;
    uint64_t vram_free_bytes;
    double   vram_bw_gbps;     /* memClock(KHz) * busWidth * 2 / 8 / 1e6 */
    double   fp16_tflops;      /* smCount * clock(GHz) * 2 * 128 */
    int      sm_count;
    int      tensor_gen;       /* major * 10 + minor */
    int      p2p_capable;      /* at least one peer among configured devices */
    int      nvlink;           /* NVLink rather than PCIe */
} ds4_gpu_device_probe;

enum {
    DS4_GPU_ROLE_NONE = 0,
    DS4_GPU_ROLE_PRIMARY,
    DS4_GPU_ROLE_CACHE,
    DS4_GPU_ROLE_SECONDARY,
};

/* §11.3 runtime overrides. -1 means "auto / unset" for the integer fields. */
typedef struct {
    int      force_primary_gpu;  /* physical device id, -1 = auto */
    int      force_cache_gpu;    /* physical device id, -1 = auto/none */
    int      disable_multi_gpu;  /* force single_gpu_mode regardless of count */
    int      kv_cache_location;  /* DS4_KV_LOC_* */
    uint32_t chunk_size;         /* §6.4, 0 = auto from free VRAM */
} ds4_gpu_roles_config;

enum {
    DS4_KV_LOC_AUTO = -1,
    DS4_KV_LOC_VRAM = 0,
    DS4_KV_LOC_RAM  = 1,
};

typedef struct {
    int    n;
    int    role[DS4_MAX_GPUS];          /* DS4_GPU_ROLE_* per probe entry */
    int    primary, cache, secondary;   /* probe indices; -1 = unassigned */
    double s_prefill[DS4_MAX_GPUS];
    double s_decode[DS4_MAX_GPUS];
    double s_capacity[DS4_MAX_GPUS];
    int    cache_sharding;              /* cache assigned and distinct */
    int    expert_sharding;             /* §0.2: n >= 2 and enabled */
    int    pipeline_parallel;           /* §0.2: n >= 3 and enabled */
    int    single_gpu_mode;             /* n < 2 or disabled */
    uint32_t chunk_size;                /* §6.4 resolved (vram_free rule) */
    int    kv_cache_location;           /* §5.3 request: DS4_KV_LOC_* or tier */
} ds4_gpu_roles;

void ds4_gpu_roles_config_default(ds4_gpu_roles_config *c);

/* Normalize, score, assign roles. Returns 0 on success, nonzero on a bad
 * argument. Never fails because of device scarcity: it degrades. */
int ds4_gpu_roles_plan(const ds4_gpu_device_probe *probes, int n,
                       const ds4_gpu_roles_config *cfg,
                       ds4_gpu_roles *out);

/* §11.2: emit scoring, role assignment, and the expert/KV placement intent as
 * machine-parseable TSON on the given stream (usually stderr). */
void ds4_gpu_roles_log_tson(FILE *out, const ds4_gpu_roles *r,
                            const ds4_gpu_device_probe *probes, int n);

/* §6.4 chunk-size table. vram_free_bytes is the decode-phase free VRAM on the
 * primary device. Returns one of 512/2048/4096/8192. */
uint32_t ds4_gpu_roles_pick_chunk(uint64_t vram_free_bytes);

/* Apply runtime flags to a role config parsed from the command line. Kept here
 * so the CLI, server, and bench share one parser for the --kv-cache-location
 * value forms (vram | ram | gpu:N). Returns 0 on success, nonzero on a bad
 * value (err populated). */
int ds4_gpu_roles_parse_kv_location(const char *value, int *out);

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
/* CUDA shim: fill up to max_out probes for the requested device filter
 * (NULL = every visible device). Returns 0 on success, nonzero on error
 * (err populated). Defined in ds4_cuda.cu. */
int ds4_gpu_roles_probe_cuda(const int *device_filter, int filter_len,
                             ds4_gpu_device_probe *out, int max_out,
                             int *out_n, char *err, size_t errlen);
#endif

/* Probe cfg's devices, plan roles, and reorder the config so index 0 is the
 * primary, index 1 the expert cache, index 2 the secondary. The engine's
 * layer packer walks devices in order, so this is what makes the roles take
 * effect without a second placement path. A no-op (returns 1) when this is
 * not a CUDA build. Returns 0 on success. */
int ds4_gpu_roles_apply(ds4_gpu_config *cfg,
                        const ds4_gpu_roles_config *rc,
                        char *err, size_t errlen);

/* The plan produced by the most recent successful ds4_gpu_roles_apply, or
 * NULL when none has run. Valid until the next apply. */
const ds4_gpu_roles *ds4_gpu_roles_last(void);

/* Device metrics from the most recent probe (name, VRAM bandwidth, fp16
 * throughput, SM count, PCIe generation/width, P2P/NVLink), for the profiler's
 * device panel. Returns the number written (<= max), 0 when none has run. */
int ds4_gpu_role_probe_snapshot(ds4_gpu_device_probe *out, int max);

#ifdef __cplusplus
}
#endif

#endif /* DS4_GPU_ROLES_H */
