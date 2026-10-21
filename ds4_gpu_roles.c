/* ds4_gpu_roles.c — see ds4_gpu_roles.h.
 *
 * The scoring weights and normalization are a direct transcription of
 * optimizations.md §1.2/§1.3. The tie-breaking is deliberately deterministic
 * (lowest probe index wins) so the same device set always yields the same
 * roles, per §11.1.
 */
#include "ds4_gpu_roles.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void ds4_gpu_roles_config_default(ds4_gpu_roles_config *c) {
    if (!c) return;
    c->force_primary_gpu = -1;
    c->force_cache_gpu   = -1;
    c->disable_multi_gpu = 0;
    c->kv_cache_location = DS4_KV_LOC_AUTO;
    c->chunk_size        = 0;
}

int ds4_gpu_roles_parse_kv_location(const char *value, int *out) {
    if (!value || !out) return 1;
    if (!strcmp(value, "auto")) { *out = DS4_KV_LOC_AUTO; return 0; }
    if (!strcmp(value, "vram")) { *out = DS4_KV_LOC_VRAM; return 0; }
    if (!strcmp(value, "ram"))  { *out = DS4_KV_LOC_RAM;  return 0; }
    if (!strncmp(value, "gpu:", 4)) {
        char *end = NULL;
        long d = strtol(value + 4, &end, 10);
        if (end == value + 4 || *end != '\0' || d < 0 || d >= DS4_MAX_GPUS) {
            return 1;
        }
        /* gpu:N is an explicit VRAM home; the engine maps it to a tier. */
        *out = (int)d;
        return 0;
    }
    return 1;
}

/* §1.2: min-max normalize. A degenerate metric (a single device, or one that
 * is equal on every device) has no ranking to express, so it maps to 1.0 for
 * every entry rather than 0.0: the argmax is trivially 0 either way, but the
 * absolute score stays meaningful for sizing decisions. */
static void normalize(const double *x, int n, double *out) {
    double lo = x[0], hi = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < lo) lo = x[i];
        if (x[i] > hi) hi = x[i];
    }
    const double span = hi - lo;
    for (int i = 0; i < n; i++) {
        out[i] = span > 0.0 ? (x[i] - lo) / span : 1.0;
    }
}

uint32_t ds4_gpu_roles_pick_chunk(uint64_t vram_free_bytes) {
    /* §6.4 table. The comparison is against available free VRAM, in bytes. */
    const uint64_t gb = 1024ull * 1024ull * 1024ull;
    if (vram_free_bytes < 1ull * gb) return 512u;
    if (vram_free_bytes < 2ull * gb) return 2048u;
    if (vram_free_bytes < 4ull * gb) return 4096u;
    return 8192u;
}

/* Lowest-index argmax, optionally skipping already-assigned probes. */
static int argmax_skip(const double *score, int n, const int *skip) {
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (skip && skip[i]) continue;
        if (best < 0 || score[i] > score[best]) best = i;
    }
    return best;
}

int ds4_gpu_roles_plan(const ds4_gpu_device_probe *probes, int n,
                       const ds4_gpu_roles_config *cfg,
                       ds4_gpu_roles *out) {
    if (!probes || !cfg || !out || n < 0 || n > DS4_MAX_GPUS) return 1;

    memset(out, 0, sizeof(*out));
    out->n = n;
    out->primary = out->cache = out->secondary = -1;
    if (n == 0) {
        out->single_gpu_mode = 1;
        return 0;
    }

    double bw[DS4_MAX_GPUS], flops[DS4_MAX_GPUS], freeb[DS4_MAX_GPUS];
    double sm[DS4_MAX_GPUS];
    for (int i = 0; i < n; i++) {
        bw[i]    = probes[i].pcie_bw_gbps;
        flops[i] = probes[i].fp16_tflops;
        freeb[i] = (double)probes[i].vram_free_bytes;
        sm[i]    = (double)probes[i].sm_count;
        out->role[i] = DS4_GPU_ROLE_NONE;
    }
    double nbw[DS4_MAX_GPUS], nflops[DS4_MAX_GPUS], nfree[DS4_MAX_GPUS];
    double nsm[DS4_MAX_GPUS], nvrambw[DS4_MAX_GPUS];
    normalize(bw, n, nbw);
    normalize(flops, n, nflops);
    normalize(freeb, n, nfree);
    normalize(sm, n, nsm);
    {
        double vrambw[DS4_MAX_GPUS];
        for (int i = 0; i < n; i++) vrambw[i] = probes[i].vram_bw_gbps;
        normalize(vrambw, n, nvrambw);
    }

    /* §1.3. Prefill is bound by getting activations in over the link; decode is
     * bound by reading resident weights, so memory bandwidth dominates there. */
    for (int i = 0; i < n; i++) {
        out->s_prefill[i]  = 0.75 * nbw[i] + 0.20 * nvrambw[i] + 0.05 * nflops[i];
        out->s_decode[i]   = 0.60 * nvrambw[i] + 0.20 * nflops[i] +
                             0.15 * nfree[i] + 0.05 * nsm[i];
        out->s_capacity[i] = 0.80 * nfree[i] + 0.10 * nvrambw[i] + 0.10 * nbw[i];
    }

    /* §1.4 role assignment, with §11.3 force overrides taking precedence.
     * A forced device id is resolved against the probe list by physical id. */
    int assigned[DS4_MAX_GPUS] = {0};
    int primary = -1, cache = -1, secondary = -1;

    if (cfg->force_primary_gpu >= 0) {
        for (int i = 0; i < n; i++) {
            if (probes[i].device_id == cfg->force_primary_gpu) { primary = i; break; }
        }
    }
    if (primary < 0) primary = argmax_skip(out->s_decode, n, NULL);
    assigned[primary] = 1;

    if (cfg->force_cache_gpu >= 0) {
        for (int i = 0; i < n; i++) {
            if (probes[i].device_id == cfg->force_cache_gpu &&
                !assigned[i]) {
                cache = i;
                break;
            }
        }
    }
    if (cache < 0 && n >= 2) {
        cache = argmax_skip(out->s_capacity, n, assigned);
    }
    /* §1.4: if the capacity winner is already primary, cache sharding is off. */
    if (cache >= 0 && cache != primary) {
        assigned[cache] = 1;
    } else {
        cache = -1;
    }

    if (n >= 3) {
        secondary = argmax_skip(out->s_prefill, n, assigned);
        if (secondary >= 0) assigned[secondary] = 1;
    }

    out->primary = primary;
    out->cache = cache;
    out->secondary = secondary;
    if (primary >= 0) out->role[primary] = DS4_GPU_ROLE_PRIMARY;
    if (cache >= 0)   out->role[cache]   = DS4_GPU_ROLE_CACHE;
    if (secondary >= 0) out->role[secondary] = DS4_GPU_ROLE_SECONDARY;

    out->cache_sharding = cache >= 0;
    out->expert_sharding = (n >= 2) && !cfg->disable_multi_gpu;
    out->pipeline_parallel = (n >= 3) && !cfg->disable_multi_gpu;
    out->single_gpu_mode = (n < 2) || cfg->disable_multi_gpu;

    /* §5.3: record the requested KV home. The engine's KV cache is device
     * resident; ds4_gpu_roles_apply warns when host RAM is requested. */
    out->kv_cache_location = cfg->kv_cache_location;

    /* §6.4: chunk target follows primary free VRAM unless the user pinned it. */
    out->chunk_size = cfg->chunk_size;
    if (out->chunk_size == 0 && primary >= 0) {
        out->chunk_size = ds4_gpu_roles_pick_chunk(probes[primary].vram_free_bytes);
    }

    /* §0.2 degrade rule: if multi-GPU is disabled, roles collapse to device
     * 0 and the cache/pipeline flags are already clear. Keeping device 0 (the
     * caller's first configured device) also means ds4_gpu_roles_apply leaves
     * the caller's ordering untouched. */
    if (out->single_gpu_mode) {
        out->primary = 0;
        out->cache = -1;
        out->secondary = -1;
        out->cache_sharding = 0;
        out->expert_sharding = 0;
        out->pipeline_parallel = 0;
        for (int i = 0; i < n; i++) {
            out->role[i] = i == out->primary ? DS4_GPU_ROLE_PRIMARY
                                             : DS4_GPU_ROLE_NONE;
        }
    }
    return 0;
}

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
static ds4_gpu_roles g_roles_last;
static int          g_roles_have_last;
static ds4_gpu_device_probe g_probe_last[DS4_MAX_GPUS];
static int          g_probe_last_n;

const ds4_gpu_roles *ds4_gpu_roles_last(void) {
    return g_roles_have_last ? &g_roles_last : NULL;
}

int ds4_gpu_role_probe_snapshot(ds4_gpu_device_probe *out, int max) {
    if (!out || max <= 0) return 0;
    int n = g_probe_last_n < max ? g_probe_last_n : max;
    for (int i = 0; i < n; i++) out[i] = g_probe_last[i];
    return n;
}

int ds4_gpu_roles_apply(ds4_gpu_config *cfg,
                        const ds4_gpu_roles_config *rc,
                        char *err, size_t errlen) {
    if (!cfg || cfg->n_gpus < 1) return 1;
    ds4_gpu_roles_config def;
    if (!rc) {
        ds4_gpu_roles_config_default(&def);
        rc = &def;
    }

    ds4_gpu_device_probe probes[DS4_MAX_GPUS];
    int n = 0;
    if (ds4_gpu_roles_probe_cuda(cfg->device_indices, cfg->n_gpus,
                                 probes, DS4_MAX_GPUS, &n, err, errlen) != 0) {
        return 1;
    }
    /* Keep the raw metrics for the profiler's device panel. */
    g_probe_last_n = n < DS4_MAX_GPUS ? n : DS4_MAX_GPUS;
    for (int i = 0; i < g_probe_last_n; i++) g_probe_last[i] = probes[i];
    ds4_gpu_roles roles;
    if (ds4_gpu_roles_plan(probes, n, rc, &roles) != 0) {
        if (err && errlen) snprintf(err, errlen, "device scoring failed");
        return 1;
    }

    /* Reorder so the in-order layer packer fills primary, then cache, then
     * secondary. Both arrays are permuted together so budget[d] keeps
     * describing device_indices[d]. */
    int    order[DS4_MAX_GPUS];
    int    on = 0;
    if (roles.primary >= 0)   order[on++] = roles.primary;
    if (roles.cache >= 0)     order[on++] = roles.cache;
    if (roles.secondary >= 0) order[on++] = roles.secondary;
    for (int i = 0; i < n; i++) {
        int seen = 0;
        for (int j = 0; j < on; j++) {
            if (order[j] == i) { seen = 1; break; }
        }
        if (!seen) order[on++] = i;
    }
    int    dev[DS4_MAX_GPUS];
    size_t vram[DS4_MAX_GPUS];
    for (int i = 0; i < on; i++) {
        dev[i]  = cfg->device_indices[order[i]];
        vram[i] = cfg->vram_bytes[order[i]];
    }
    for (int i = 0; i < on; i++) {
        cfg->device_indices[i] = dev[i];
        cfg->vram_bytes[i]     = vram[i];
    }

    g_roles_last = roles;
    g_roles_have_last = 1;
    ds4_gpu_roles_log_tson(stderr, &roles, probes, n);
    if (roles.kv_cache_location == DS4_KV_LOC_RAM) {
        fprintf(stderr,
                "ds4: --kv-cache-location=ram requested, but host KV residency "
                "is not implemented; KV stays in device VRAM\n");
    }
    return 0;
}
#endif

/* TSON is emitted as `ds4.roles.<field> = value` lines. It stays one
 * key/value per line so `grep`/`cut` can parse it without a TSON reader. */
static void tson_kv(FILE *f, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
}

static const char *role_name(int role) {
    switch (role) {
    case DS4_GPU_ROLE_PRIMARY:   return "primary";
    case DS4_GPU_ROLE_CACHE:     return "cache";
    case DS4_GPU_ROLE_SECONDARY: return "secondary";
    default:                     return "none";
    }
}

void ds4_gpu_roles_log_tson(FILE *out, const ds4_gpu_roles *r,
                            const ds4_gpu_device_probe *probes, int n) {
    if (!out || !r) return;
    tson_kv(out, "ds4.roles.n_gpus = %d\n", r->n);
    tson_kv(out, "ds4.roles.single_gpu_mode = %d\n", r->single_gpu_mode);
    tson_kv(out, "ds4.roles.expert_sharding = %d\n", r->expert_sharding);
    tson_kv(out, "ds4.roles.cache_sharding = %d\n", r->cache_sharding);
    tson_kv(out, "ds4.roles.pipeline_parallel = %d\n", r->pipeline_parallel);
    tson_kv(out, "ds4.roles.chunk_size = %u\n", r->chunk_size);
    tson_kv(out, "ds4.roles.kv_cache_location = %d\n", r->kv_cache_location);
    tson_kv(out, "ds4.roles.primary = %d\n", r->primary);
    tson_kv(out, "ds4.roles.cache = %d\n", r->cache);
    tson_kv(out, "ds4.roles.secondary = %d\n", r->secondary);
    if (probes) {
        for (int i = 0; i < n && i < r->n; i++) {
            tson_kv(out,
                "ds4.roles.dev[%d] = {id:%d name:%s role:%s "
                "s_prefill:%.4f s_decode:%.4f s_capacity:%.4f "
                "pcie_gen:%d pcie_width:%d pcie_bw_gbps:%.2f "
                "vram_free:%.1f vram_total:%.1f vram_bw_gbps:%.1f "
                "fp16_tflops:%.1f sm:%d tensor_gen:%d p2p:%d nvlink:%d}\n",
                i, probes[i].device_id, probes[i].name,
                role_name(r->role[i]),
                r->s_prefill[i], r->s_decode[i], r->s_capacity[i],
                probes[i].pcie_gen, probes[i].pcie_width, probes[i].pcie_bw_gbps,
                (double)probes[i].vram_free_bytes / 1073741824.0,
                (double)probes[i].vram_total_bytes / 1073741824.0,
                probes[i].vram_bw_gbps, probes[i].fp16_tflops,
                probes[i].sm_count, probes[i].tensor_gen,
                probes[i].p2p_capable, probes[i].nvlink);
        }
    }
    fflush(out);
}
