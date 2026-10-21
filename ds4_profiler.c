/* ds4_profiler.c — see ds4_profiler.h. Numbers only; no terminal rendering. */
#include "ds4_profiler.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

/* ---- shared state -------------------------------------------------------- */

#if !defined(_WIN32)
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
#define LOCK()   pthread_mutex_lock(&g_mtx)
#define UNLOCK() pthread_mutex_unlock(&g_mtx)
#else
#define LOCK()   ((void)0)
#define UNLOCK() ((void)0)
#endif

/* The CUDA backend owns the real snapshot; other backends report no devices. */
#if defined(DS4_NO_GPU) || defined(__APPLE__) || defined(DS4_ROCM_BUILD)
void ds4_gpu_profiler_stats(ds4_profiler_stats *out) {
    if (out) memset(out, 0, sizeof(*out));
}
void ds4_gpu_profiler_init(void) {}
void ds4_gpu_set_streaming_expert_type_name(const char *name) { (void)name; }
void ds4_gpu_profiler_set_device_resident(int device_id, uint64_t dense_bytes,
                                          uint32_t expert_entries,
                                          uint64_t expert_bytes) {
    (void)device_id; (void)dense_bytes; (void)expert_entries; (void)expert_bytes;
}
int ds4_gpu_transfer_snapshot(ds4_gpu_transfer *out, int max_tiers) {
    (void)out; (void)max_tiers;
    return 0;
}
uint64_t ds4_gpu_pcie_peak_bps(int tier) { (void)tier; return 0; }
void ds4_gpu_phase_snapshot(ds4_gpu_phases *out) {
    if (out) memset(out, 0, sizeof(*out));
}
void ds4_gpu_machine_snapshot(ds4_gpu_machine *out) {
    if (out) memset(out, 0, sizeof(*out));
}
#endif

static ds4_profiler_host_stats g_host;
static ds4_profiler_timing     g_timing;
static int                     g_host_set;

static double monotonic_sec(void) {
#if !defined(_WIN32)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static void stat_add(ds4_prof_stat *s, double v) {
    if (s->n == 0) {
        s->min = v;
        s->max = v;
    } else {
        if (v < s->min) s->min = v;
        if (v > s->max) s->max = v;
    }
    s->last = v;
    s->sum += v;
    s->n++;
}

void ds4_profiler_set_host_stats(const ds4_profiler_host_stats *s) {
    if (!s) return;
    LOCK();
    g_host = *s;
    g_host_set = 1;
    UNLOCK();
}

/* One row per kind of cost; see ds4_profiler.h for the category rules. The
 * label is what the dashboard shows, the hint is its tooltip. */
static const struct {
    const char *id;
    const char *label;
    const char *hint;
    const char *category;
} k_steps[DS4_PSTEP_COUNT] = {
    [DS4_PSTEP_HOST_EMBED]        = {"host_embed", "Input & embedding",
        "Turn the sampled token id into the vector the model reads. CPU-side lookup and staging.",
        "host"},
    [DS4_PSTEP_HOST_ROUTER]       = {"host_router", "Router dispatch",
        "CPU work to launch the gate that scores all 256 experts and picks 6 per layer. GPU time is a separate device row.",
        "host"},
    [DS4_PSTEP_HOST_ATTN]         = {"host_attn", "Attention dispatch",
        "CPU work to encode the attention kernels (QKV, RoPE, KV cache). The kernels themselves are device rows.",
        "host"},
    [DS4_PSTEP_HOST_SHARED]       = {"host_shared", "Shared expert dispatch",
        "CPU work to launch the dense expert every token uses, on top of the 6 routed ones.",
        "host"},
    [DS4_PSTEP_HOST_MOE_PREP]     = {"host_moe_prep", "Routed expert staging (CPU)",
        "Choosing expert weights to fetch and pointing the copy at them. Disk/RAM reads themselves are transfer rows.",
        "host"},
    [DS4_PSTEP_HOST_TIER_DISPATCH] = {"host_tier", "Second-GPU dispatch (CPU)",
        "CPU work to hand the experts resident on the 3060 to that card. Does not include waiting for it.",
        "host"},
    [DS4_PSTEP_HOST_COMBINE]      = {"host_combine", "MoE combine dispatch",
        "CPU work to launch the weighted sum of the 6 routed expert outputs plus the shared one.",
        "host"},
    [DS4_PSTEP_HOST_HEAD]         = {"host_head", "Output head dispatch",
        "CPU work to launch the final projection that turns the hidden state into 163,840 vocab logits.",
        "host"},
    [DS4_PSTEP_HOST_SAMPLE]       = {"host_sample", "Sampling (CPU)",
        "Turning logits into a token on the CPU: softmax filtering (top-p/min-p) and the draw. The 22 ms slow path lives here.",
        "host"},
    [DS4_PSTEP_HOST_OTHER]        = {"host_other", "Other CPU work",
        "Everything else the CPU does for a token: scheduling, stop checks. Residual of the host rows.",
        "host"},
    [DS4_PSTEP_HOST_DETOK]        = {"host_detok", "Host: detokenize + display",
        "Turning the sampled id into text and handing it to the frontend. Part of what the user waits for, not model compute.",
        "host"},

    [DS4_PSTEP_DEV_EMBED]         = {"dev_embed", "Embedding kernel (GPU)",
        "GPU work to materialise the token's embedding rows.", "device"},
    [DS4_PSTEP_DEV_ATTN_SCORE]    = {"dev_attn_score", "Attention scores (GPU)",
        "GPU time scoring how much each earlier token matters (QK^T over the KV cache).", "device"},
    [DS4_PSTEP_DEV_ATTN_VALUE]    = {"dev_attn_value", "Attention values (GPU)",
        "GPU time mixing the KV values with those scores (softmax*V) and the output projection.", "device"},
    [DS4_PSTEP_DEV_ROUTER]        = {"dev_router", "Router kernel (GPU)",
        "GPU time for the expert gate scoring and top-6 selection.", "device"},
    [DS4_PSTEP_DEV_SHARED]        = {"dev_shared", "Shared expert (GPU)",
        "GPU time for the always-on dense expert's matmuls.", "device"},
    [DS4_PSTEP_DEV_MOE_GATE_UP]   = {"dev_moe_gate_up", "Routed gate+up (GPU)",
        "GPU time for the gate and up projections of the 6 selected experts.", "device"},
    [DS4_PSTEP_DEV_MOE_DOWN]      = {"dev_moe_down", "Routed down (GPU)",
        "GPU time for the down projections of the 6 selected experts.", "device"},
    [DS4_PSTEP_DEV_TIER]          = {"dev_tier", "Second-GPU kernels",
        "GPU time on the 3060 computing the experts it holds. Overlaps the primary's work.", "device"},
    [DS4_PSTEP_DEV_HEAD]          = {"dev_head", "Output head (GPU)",
        "GPU time for the final projection to vocabulary logits.", "device"},
    [DS4_PSTEP_DEV_IDLE]          = {"dev_idle", "GPU idle",
        "Time the primary GPU is not running any kernel for this token, i.e. waiting for data or launches.", "device"},

    [DS4_PSTEP_XFER_EXPERT_POOL]  = {"xfer_expert_pool", "Expert load: from VRAM pool",
        "Expert weights already in a GPU's VRAM: copied device-to-device, no PCIe or disk.", "transfer"},
    [DS4_PSTEP_XFER_EXPERT_ARENA] = {"xfer_expert_arena", "Expert load: from host RAM",
        "Expert weights held in pinned host memory, copied to the GPU over PCIe. This is the streaming hot path.", "transfer"},
    [DS4_PSTEP_XFER_EXPERT_SSD]   = {"xfer_expert_ssd", "Expert load: from SSD",
        "Expert weights read from the model file on disk, then sent to the GPU. The slowest source.", "transfer"},
    [DS4_PSTEP_XFER_PEER]         = {"xfer_peer", "Inter-GPU copies",
        "Activation or expert traffic between the two cards (partial results, owned-slot data).", "transfer"},
    [DS4_PSTEP_XFER_LOGITS]       = {"xfer_logits", "Logits readback",
        "Copying 163,840 floats of logits from GPU to CPU each token so the sampler can run.", "transfer"},
    [DS4_PSTEP_XFER_PARTIALS]     = {"xfer_partials", "Partial results readback",
        "Copying routed-expert partial outputs from the second GPU back to the primary.", "transfer"},
    [DS4_PSTEP_XFER_KV]           = {"xfer_kv", "KV cache expansion",
        "Expanding the compressed (F16) KV cache to F32 for attention, every layer, every token.", "transfer"},
    [DS4_PSTEP_XFER_ACT_HOST]     = {"xfer_act_host", "Transfers: activations to device",
        "Activation or control tensors copied host-to-device: router ids, weights, handoff buffers. Measured copy time and bytes.",
        "transfer"},
    [DS4_PSTEP_XFER_ACT_READBACK] = {"xfer_act_readback", "Transfers: activations from device",
        "Activation, partial-result or id tensors copied device-to-host. The blocking time is also charged to the readback wait row.",
        "transfer"},

    [DS4_PSTEP_WAIT_UPLOAD]       = {"wait_upload", "Wait: expert upload",
        "CPU blocked until the expert upload stream catches up. Large values mean the GPU is starved for weights.", "wait"},
    [DS4_PSTEP_WAIT_READBACK]     = {"wait_readback", "Wait: device readback",
        "CPU blocked on a device-to-host copy finishing (logits or partials).", "wait"},
    [DS4_PSTEP_WAIT_TIER]         = {"wait_tier", "Wait: second GPU",
        "CPU blocked on the 3060 finishing its expert kernels or copies.", "wait"},
    [DS4_PSTEP_WAIT_SYNC]         = {"wait_sync", "Wait: other synchronisation",
        "CPU blocked on any other device sync (graph completion, event waits).", "wait"},
};

static ds4_prof_step_stat g_steps[DS4_PSTEP_COUNT];

/* Expert-usage ring: one record per (token, layer, expert). The write index
 * wraps, so the snapshot is rotated back into chronological order. */
static ds4_prof_expert_trace g_experts[DS4_PTRACE_EXPERT_RING];
static int      g_experts_head;
static int      g_experts_count;
static uint32_t g_trace_token;
static uint64_t g_expert_next_seq;

/* Defined after the record state below; called with the profiler lock held. */
static uint64_t current_rec_seq_locked(void);
static uint32_t current_rec_token_locked(void);
static void current_rec_note_expert_locked(void);

void ds4_profiler_trace_token(uint32_t token_seq) {
    LOCK();
    g_trace_token = token_seq;
    UNLOCK();
}

void ds4_profiler_trace_expert(uint16_t layer, uint16_t expert,
                               uint8_t source, uint8_t src_gpu,
                               uint8_t dest_gpu, uint32_t bytes) {
    LOCK();
    ds4_prof_expert_trace *e = &g_experts[g_expert_next_seq % DS4_PTRACE_EXPERT_RING];
    memset(e, 0, sizeof(*e));
    e->rec_seq = current_rec_seq_locked();
    e->layer = layer;
    e->expert = expert;
    e->source = source;
    e->src_gpu = src_gpu;
    e->dest_gpu = dest_gpu;
    e->bytes = bytes;
    e->token = current_rec_token_locked();
    g_expert_next_seq++;
    g_experts_head = (int)(g_expert_next_seq % DS4_PTRACE_EXPERT_RING);
    if (g_experts_count < DS4_PTRACE_EXPERT_RING) g_experts_count++;
    current_rec_note_expert_locked();
    UNLOCK();
}

int ds4_profiler_trace_snapshot(ds4_prof_expert_trace *out, int max) {
    if (!out || max <= 0) return 0;
    int n = 0;
    LOCK();
    const int count = g_experts_count;
    const int start = (g_experts_head - count + DS4_PTRACE_EXPERT_RING)
                    % DS4_PTRACE_EXPERT_RING;
    for (int i = 0; i < count && n < max; i++) {
        out[n++] = g_experts[(start + i) % DS4_PTRACE_EXPERT_RING];
    }
    UNLOCK();
    return n;
}

uint32_t ds4_profiler_trace_current_token(void) {
    LOCK();
    const uint32_t t = g_trace_token;
    UNLOCK();
    return t;
}

/* ---- ordered event log + records + all-device state --------------------- */

static ds4_prof_event     g_events[DS4_PEV_RING];
static int                g_events_head;
static int                g_events_count;
static uint64_t           g_event_next_seq;
static uint64_t           g_expert_next_seq;
static double             g_trace_t0;
static ds4_prof_dev_state g_dev_state[DS4_PEV_DEV_MAX];
static int                g_dev_state_n;
static ds4_prof_token_rec g_token_recs[DS4_PTREC_RING];
static int                g_token_rec_head;
static int                g_token_rec_count;

static const char *const k_event_names[DS4_PEV_COUNT] = {
    "expert transfer start", "expert transfer end", "expert cache hit",
    "compute start", "compute end", "activation transfer start",
    "activation transfer end", "choose token", "route experts",
    "stage complete", "detokenize", "host stall",
};

const char *ds4_profiler_event_kind(uint16_t kind) {
    return kind < DS4_PEV_COUNT ? k_event_names[kind] : "";
}

/* ---- stage table --------------------------------------------------------- */

typedef struct {
    int      used;
    char     key[48];
    char     label[56];
    char     part[16];
    char     name[40];
    char     category[16];
    char     hint[256];
    uint64_t total_ns, encode_ns, wait_ns, count, bytes;
    /* The same sums restricted to one record kind, so decode tokens and
     * prefill chunks keep separate denominators and never blur into one
     * "average" for a boundary that ran in both. */
    uint64_t k_total_ns[DS4_PREC_KIND_MAX];
    uint64_t k_encode_ns[DS4_PREC_KIND_MAX];
    uint64_t k_wait_ns[DS4_PREC_KIND_MAX];
    uint64_t k_count[DS4_PREC_KIND_MAX];
    uint64_t k_bytes[DS4_PREC_KIND_MAX];
    uint64_t last_ns;
    /* Per-record accumulation, zeroed at record begin. */
    uint64_t tok_total_ns, tok_encode_ns, tok_wait_ns, tok_count, tok_bytes;
    ds4_prof_stat total_ms;
    ds4_prof_stat encode_ms;
    ds4_prof_stat wait_ms;
} stage_slot;

static stage_slot g_stages[DS4_PROF_MAX_STAGES];
static int        g_stage_n;
static int        g_stage_capture;
static int        g_record_enabled;

/* ---- system sampling ----------------------------------------------------- */

typedef struct {
    uint64_t proc_cpu_ns;
    uint64_t sys_total_ns;
    uint64_t sys_busy_ns;
    uint64_t ram_used;
    uint64_t ram_avail;
} sys_sample;

static uint64_t monotonic_ns(void) {
#if !defined(_WIN32)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#else
    return (uint64_t)(monotonic_sec() * 1e9);
#endif
}

static void sys_sample_now(sys_sample *s) {
    memset(s, 0, sizeof(*s));
#if !defined(_WIN32)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        s->proc_cpu_ns = (uint64_t)ru.ru_utime.tv_sec * 1000000000ull +
                         (uint64_t)ru.ru_utime.tv_usec * 1000ull +
                         (uint64_t)ru.ru_stime.tv_sec * 1000000000ull +
                         (uint64_t)ru.ru_stime.tv_usec * 1000ull;
    }
    FILE *fp = fopen("/proc/stat", "r");
    if (fp) {
        unsigned long long v[10] = {0};
        if (fscanf(fp, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7],
                   &v[8], &v[9]) >= 4) {
            for (int i = 0; i < 10; i++) s->sys_total_ns += v[i];
            const uint64_t idle = v[3] + v[4];
            s->sys_busy_ns = s->sys_total_ns > idle ? s->sys_total_ns - idle : 0;
        }
        fclose(fp);
    }
    if (s->sys_total_ns == 0) s->sys_total_ns = 1;
    fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[128];
        unsigned long long kb = 0, avail = 0, total = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) total = kb;
            else if (sscanf(line, "MemAvailable: %llu kB", &avail) == 1) { }
        }
        fclose(fp);
        s->ram_used = total > avail ? (total - avail) * 1024ull : 0;
        s->ram_avail = avail * 1024ull;
    }
#endif
}

/* Query EVERY NVIDIA device, including ones this process does not use: a
 * token can be slowed by another card (display, browser) and the trace should
 * say so. dlopen keeps NVML an optional runtime dependency. */
static void dev_state_sample(ds4_prof_dev_state *out, int *out_n) {
    typedef struct { unsigned int gpu, memory; } nvml_util;
    typedef struct { unsigned long long total, free, used; } nvml_mem;
    static int tried;
    static void *lib;
    static int (*p_count)(unsigned *) ;
    static int (*p_init)(void);
    static int (*p_handle)(unsigned, void **);
    static int (*p_name)(void *, char *, unsigned);
    static int (*p_util)(void *, nvml_util *);
    static int (*p_mem)(void *, nvml_mem *);
    *out_n = 0;
    if (!tried) {
        tried = 1;
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (lib) {
            *(void **)(&p_init)  = dlsym(lib, "nvmlInit_v2");
            *(void **)(&p_count) = dlsym(lib, "nvmlDeviceGetCount_v2");
            *(void **)(&p_handle) = dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
            *(void **)(&p_name)  = dlsym(lib, "nvmlDeviceGetName");
            *(void **)(&p_util)  = dlsym(lib, "nvmlDeviceGetUtilizationRates");
            *(void **)(&p_mem)   = dlsym(lib, "nvmlDeviceGetMemoryInfo");
        }
    }
    if (!lib || !p_init || !p_count || !p_handle || p_init() != 0) return;
    unsigned n = 0;
    if (p_count(&n) != 0) return;
    if (n > DS4_PEV_DEV_MAX) n = DS4_PEV_DEV_MAX;
    for (unsigned i = 0; i < n; i++) {
        void *h = NULL;
        if (p_handle(i, &h) != 0 || !h) continue;
        ds4_prof_dev_state *s = &out[*out_n];
        memset(s, 0, sizeof(*s));
        s->device = (int)i;
        if (p_name) p_name(h, s->name, sizeof(s->name));
        nvml_util u = {0, 0};
        if (p_util && p_util(h, &u) == 0) s->util_pct = u.gpu;
        nvml_mem m = {0, 0, 0};
        if (p_mem && p_mem(h, &m) == 0) { s->mem_used = m.used; s->mem_total = m.total; }
        (*out_n)++;
    }
}

static void dev_state_publish(const ds4_prof_dev_state *s, int n) {
    LOCK();
    g_dev_state_n = n < DS4_PEV_DEV_MAX ? n : DS4_PEV_DEV_MAX;
    for (int i = 0; i < g_dev_state_n; i++) g_dev_state[i] = s[i];
    UNLOCK();
}

/* ---- record state -------------------------------------------------------- */

typedef struct {
    int      dev_n;
    ds4_prof_dev_state dev[DS4_PROF_REC_MAX_DEV];
} dev_sample;

typedef struct {
    int open;
    uint64_t seq;
    ds4_prof_rec_kind kind;
    uint32_t token;
    int32_t pos;
    uint32_t n_tokens;
    char text[52];
    uint64_t start_ns;
    uint32_t stage_n;
    ds4_prof_stage_use stages[DS4_PROF_REC_MAX_STAGES];
    uint32_t event_n;
    uint32_t expert_n;
    uint64_t event_first;
    uint64_t expert_first;
    sys_sample sys0;
    ds4_gpu_machine m0;
    dev_sample dev0;
} cur_record;

/* Device residency/VRAM sampling touches backend state. Keep it off the
 * engine thread (which may be mid-tier) by refreshing a cache from a detached
 * telemetry thread; the TUI and the HTTP serializer only read the cache. */
static ds4_profiler_stats g_cached_stats;
static int                g_stats_started;
static pthread_once_t     g_stats_once = PTHREAD_ONCE_INIT;

static void *stats_thread_main(void *ud) {
    (void)ud;
    for (;;) {
        ds4_profiler_stats s;
        ds4_gpu_profiler_stats(&s);
        LOCK();
        g_cached_stats = s;
        UNLOCK();
        struct timespec ts = {0, 250000000};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void stats_thread_start_once(void) {
#if !defined(_WIN32)
    pthread_attr_t attr;
    pthread_t th;
    if (pthread_attr_init(&attr) == 0) {
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&th, &attr, stats_thread_main, NULL) == 0)
            g_stats_started = 1;
        pthread_attr_destroy(&attr);
    }
#endif
}

static void stats_thread_ensure(void) {
    if (!g_stats_started) pthread_once(&g_stats_once, stats_thread_start_once);
}

static void stats_snapshot_cached(ds4_profiler_stats *out) {
    stats_thread_ensure();
    LOCK();
    *out = g_cached_stats;
    UNLOCK();
}

static cur_record      g_cur;
static uint64_t        g_rec_next_seq;
static ds4_prof_record g_records[DS4_PROF_REC_RING];
static int             g_rec_head;
static int             g_rec_count;

/* Coarse per-token accounting (host/device split) fed by the backend phase
 * counters. The record above is the detailed view; both are measured. */
static double           g_tok_t0;
static ds4_gpu_phases   g_tok_ph0;
static ds4_gpu_transfer g_tok_prev[DS4_PROFILER_MAX_DEV];
static int              g_tok_prev_n;
static int              g_tok_open;
static void timing_note_token(uint64_t tokens);

static uint64_t current_rec_seq_locked(void) {
    return g_cur.open ? g_cur.seq : 0;
}
static uint32_t current_rec_token_locked(void) {
    return g_cur.open ? g_cur.token : g_trace_token;
}
static void current_rec_note_expert_locked(void) {
    if (g_cur.open && g_cur.expert_n < UINT32_MAX) g_cur.expert_n++;
}


static void stage_note_ex(const char *part, const char *name,
                          const char *label, const char *hint,
                          const char *category, uint32_t layer,
                          uint64_t total_ns, uint64_t encode_ns,
                          uint64_t wait_ns, uint64_t bytes);

void ds4_profiler_trace_begin(void) {
    LOCK();
    g_trace_token++;
    g_trace_t0 = monotonic_sec();
    UNLOCK();
}

void ds4_profiler_trace_end(void) {
    /* Records own the trace now; this entry point is kept for the old
     * sample-to-sample writers and deliberately writes nothing. */
}

/* ---- stage naming -------------------------------------------------------- */

static void pretty_label(const char *name, char *out, size_t cap) {
    size_t w = 0;
    const char *p = name;
    while (*p && w + 1 < cap) {
        const char *e = strchr(p, '_');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        char word[32];
        if (n >= sizeof(word)) n = sizeof(word) - 1;
        memcpy(word, p, n);
        word[n] = '\0';
        const char *fixed = NULL;
        if (!strcmp(word, "attn")) fixed = "Attention";
        else if (!strcmp(word, "ffn")) fixed = "FFN";
        else if (!strcmp(word, "hc")) fixed = "HC";
        else if (!strcmp(word, "moe")) fixed = "MoE";
        else if (!strcmp(word, "rope")) fixed = "RoPE";
        else if (!strcmp(word, "kv")) fixed = "KV";
        else if (!strcmp(word, "qkv")) fixed = "QKV";
        else if (!strcmp(word, "q")) fixed = "Q";
        else if (!strcmp(word, "qat")) fixed = "QAT";
        else if (!strcmp(word, "ssd")) fixed = "SSD";
        else if (!strcmp(word, "gpu")) fixed = "GPU";
        else if (!strcmp(word, "cpu")) fixed = "CPU";
        if (w && w + 1 < cap) out[w++] = ' ';
        if (fixed) {
            for (const char *q = fixed; *q && w + 1 < cap; q++) out[w++] = *q;
        } else {
            for (size_t i = 0; i < n && w + 1 < cap; i++)
                out[w++] = (char)(i == 0 ? toupper((unsigned char)word[i]) : word[i]);
        }
        if (!e) break;
        p = e + 1;
    }
    out[w] = '\0';
}

static const char *stage_category(const char *part, const char *name) {
    if (!strcmp(part, "attn") || !strcmp(part, "decode_attn") ||
        strstr(name, "attn") || strstr(name, "q_") ||
        !strcmp(name, "q_path") || strstr(name, "rope") ||
        strstr(name, "attention") || strstr(name, "kv_path"))
        return "attention";
    if (strstr(name, "routed") || strstr(name, "moe")) return "routed";
    if (strstr(name, "shared")) return "shared";
    if (strstr(name, "router") || strstr(name, "route")) return "router";
    if (strstr(name, "compressor") || strstr(name, "indexer")) return "indexer";
    if (strstr(name, "head") || !strcmp(part, "output")) return "output";
    if (strstr(name, "hc_") || strstr(name, "norm") || strstr(name, "flat"))
        return "norm and mixing";
    if (strstr(name, "sample") || strstr(name, "detok")) return "sampling";
    return "other";
}

static const char *stage_hint(const char *part, const char *name) {
    static const struct { const char *name, *hint; } k[] = {
        {"attn_hc_pre", "Mix the residual streams and project the attention pre-norm weights."},
        {"attn_norm", "Normalise the attention input for this layer."},
        {"q_path", "Project and normalise the query, then apply RoPE."},
        {"kv_path", "Project K/V, normalise, apply RoPE and append to the KV cache."},
        {"compressor_proj", "Project the compressed-attention keys and query for this layer."},
        {"compressor_update", "Update the latent compressor state with the new token."},
        {"compressor_quantize", "Quantise the new compressed KV row."},
        {"compressor_commit", "Commit the compressed KV row and its metadata."},
        {"indexer_compressor_proj", "Project the sparse-indexer query/key for this layer."},
        {"indexer_compressor_update", "Update the indexer compressor state."},
        {"indexer_compressor_qat", "Quantise the indexer state for top-k scoring."},
        {"compressor_indexer", "Run the sparse indexer's top-k selection."},
        {"attn_inv_rope", "Undo the RoPE rotation on the attention output."},
        {"attn_output", "Apply the output projection and fold the attention result back."},
        {"attn_hc_post", "Mix the attention result into the residual streams."},
        {"ffn_hc_pre", "Mix and project the FFN pre-norm weights."},
        {"ffn_norm", "Normalise the FFN input for this layer."},
        {"router", "Score all routed experts and pick the top six."},
        {"shared_gate_up", "Compute gate and up projections of the always-on shared expert."},
        {"routed_moe", "Among the 256 experts per layer, the six picked for this token: their weights are fetched from VRAM, host RAM or SSD and their MLPs run. It holds the most weight bytes and FLOPs of any stage, so it is usually the hot one."},
        {"routed_moe_folded", "Combined routed-expert pass with the shared expert fused in."},
        {"shared_down", "Down-project the shared expert and add it to the routed sum."},
        {"ffn_hc_post", "Mix the FFN result back into the residual streams."},
        {"output_norm", "Final normalisation before the vocabulary projection."},
        {"output_proj", "Project the hidden state to the full vocabulary."},
        {"head_norm", "Normalise the hidden state for the output head."},
        {"attention", "Batched prefill attention over the whole chunk."},
        {"hc_weights", "Compute the hyper-connection mixing weights."},
        {"hc_pre", "Apply the hyper-connection pre-mix."},
        {"hc_post", "Apply the hyper-connection post-mix."},
        {"hc_flat_norm", "Normalise the flattened hyper-connection rows."},
        {"hc_weighted_sum", "Combine the hyper-connection branches."},
        {"norm", "Normalise the token rows."},
        {"rope", "Apply rotary position encoding to the batch."},
        {"pre_q", "Prepare the query rows before projection."},
        {"q_a", "Low-rank query projection (A)."},
        {"q_a_norm", "Normalise the low-rank query."},
        {"q_b", "Low-rank query projection (B)."},
        {"inv_rope", "Undo RoPE on the attention output rows."},
        {"ffn", "Batched routed/shared FFN half of a prefill layer."},
    };
    for (size_t i = 0; i < sizeof(k) / sizeof(k[0]); i++)
        if (!strcmp(name, k[i].name)) return k[i].hint;
    if (!strcmp(part, "transfer"))
        return "Bytes moved across a link. The measured time is the part the CPU waited for; bytes and rate show the rest.";
    if (!strcmp(part, "wait"))
        return "CPU time blocked on a device or read. This is waiting, not work.";
    if (!strcmp(part, "host"))
        return "CPU work on the calling thread.";
    if (!strcmp(part, "device"))
        return "Device time inside this operation.";
    return "Scheduler stage. Total = CPU launch + time blocked on the device.";
}

static void stage_reset_accum(stage_slot *s) {
    s->tok_total_ns = s->tok_encode_ns = s->tok_wait_ns = 0;
    s->tok_count = s->tok_bytes = 0;
}

static int stage_intern_locked(const char *part, const char *name,
                               const char *label, const char *hint,
                               const char *category) {
    char key[48];
    snprintf(key, sizeof(key), "%s/%s", part, name);
    for (int i = 0; i < g_stage_n; i++)
        if (!strcmp(g_stages[i].key, key)) return i;
    if (g_stage_n >= DS4_PROF_MAX_STAGES) return -1;
    stage_slot *s = &g_stages[g_stage_n];
    memset(s, 0, sizeof(*s));
    s->used = 1;
    snprintf(s->key, sizeof(s->key), "%s", key);
    snprintf(s->part, sizeof(s->part), "%s", part);
    snprintf(s->name, sizeof(s->name), "%s", name);
    if (label && label[0]) snprintf(s->label, sizeof(s->label), "%s", label);
    else pretty_label(name, s->label, sizeof(s->label));
    if (hint && hint[0]) snprintf(s->hint, sizeof(s->hint), "%s", hint);
    else snprintf(s->hint, sizeof(s->hint), "%s", stage_hint(part, name));
    if (category && category[0]) snprintf(s->category, sizeof(s->category), "%s", category);
    else snprintf(s->category, sizeof(s->category), "%s", stage_category(part, name));
    return g_stage_n++;
}

static void event_add_locked(uint16_t kind, uint16_t stage, uint16_t layer,
                             uint16_t expert, uint8_t device, uint32_t bytes,
                             uint32_t dur_ns) {
    const uint64_t now = monotonic_ns();
    ds4_prof_event *e = &g_events[g_event_next_seq % DS4_PEV_RING];
    memset(e, 0, sizeof(*e));
    e->seq = g_event_next_seq++;
    e->rec_seq = g_cur.open ? g_cur.seq : 0;
    e->t_us = (g_cur.open && now >= g_cur.start_ns)
        ? (uint32_t)((now - g_cur.start_ns) / 1000ull) : 0;
    e->dur_ns = dur_ns;
    e->kind = kind;
    e->stage = stage;
    e->layer = layer;
    e->expert = expert;
    e->device = device;
    e->bytes = bytes;
    e->token = g_cur.open ? g_cur.token : g_trace_token;
    g_events_head = (int)(g_event_next_seq % DS4_PEV_RING);
    if (g_events_count < DS4_PEV_RING) g_events_count++;
    if (g_cur.open && g_cur.event_n < UINT32_MAX) g_cur.event_n++;
}

static void stage_note_ex(const char *part, const char *name,
                          const char *label, const char *hint,
                          const char *category, uint32_t layer,
                          uint64_t total_ns, uint64_t encode_ns,
                          uint64_t wait_ns, uint64_t bytes) {
    if (!g_stage_capture) return;
    const double total_ms = (double)total_ns / 1e6;
    const double enc_ms = (double)encode_ns / 1e6;
    const double wait_ms = (double)wait_ns / 1e6;
    LOCK();
    const int id = stage_intern_locked(part, name, label, hint, category);
    if (id < 0) { UNLOCK(); return; }
    stage_slot *s = &g_stages[id];
    s->total_ns += total_ns;
    s->encode_ns += encode_ns;
    s->wait_ns += wait_ns;
    s->count++;
    s->bytes += bytes;
    s->last_ns = total_ns;
    stat_add(&s->total_ms, total_ms);
    stat_add(&s->encode_ms, enc_ms);
    stat_add(&s->wait_ms, wait_ms);
    /* Kind-tagged sums for the record kinds the profiler actually records.
     * They stay zero for boundaries only reached outside a record. */
    if (g_cur.open && g_cur.kind < DS4_PREC_KIND_MAX) {
        const int k = (int)g_cur.kind;
        s->k_total_ns[k] += total_ns;
        s->k_encode_ns[k] += encode_ns;
        s->k_wait_ns[k] += wait_ns;
        s->k_count[k]++;
        s->k_bytes[k] += bytes;
    }
    if (g_cur.open) {
        s->tok_total_ns += total_ns;
        s->tok_encode_ns += encode_ns;
        s->tok_wait_ns += wait_ns;
        s->tok_count++;
        s->tok_bytes += bytes;
        ds4_prof_stage_use *u = NULL;
        for (uint32_t i = 0; i < g_cur.stage_n; i++)
            if (g_cur.stages[i].id == (uint16_t)id) { u = &g_cur.stages[i]; break; }
        if (!u && g_cur.stage_n < DS4_PROF_REC_MAX_STAGES) {
            u = &g_cur.stages[g_cur.stage_n++];
            memset(u, 0, sizeof(*u));
            u->id = (uint16_t)id;
        }
        if (u) {
            u->total_ns += total_ns;
            u->encode_ns += encode_ns;
            u->wait_ns += wait_ns;
            u->count++;
            u->bytes += bytes;
        }
        event_add_locked(DS4_PEV_STAGE, (uint16_t)id, (uint16_t)layer, 0, 0,
                         (uint32_t)bytes, (uint32_t)(total_ns > UINT32_MAX
                             ? UINT32_MAX : total_ns));
    }
    UNLOCK();
}

void ds4_profiler_note_stage(const char *part, const char *name,
                             uint32_t layer, uint32_t pos, uint32_t n_tokens,
                             uint64_t total_ns, uint64_t encode_ns,
                             uint64_t wait_ns, uint64_t bytes) {
    (void)pos;
    (void)n_tokens;
    stage_note_ex(part ? part : "stage", name ? name : "unnamed",
                  NULL, NULL, NULL, layer, total_ns, encode_ns, wait_ns, bytes);
}

void ds4_profiler_set_stage_capture(int on) { g_stage_capture = on ? 1 : 0; }
int ds4_profiler_stage_capture_enabled(void) { return g_stage_capture; }

void ds4_profiler_trace_event(uint16_t kind, uint16_t layer, uint16_t expert,
                              uint8_t device, uint32_t bytes) {
    if (!g_stage_capture && !g_record_enabled && !ds4_profiler_trace_enabled())
        return;
    LOCK();
    event_add_locked(kind, 0, layer, expert, device, bytes, 0);
    UNLOCK();
}

int ds4_profiler_trace_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *p = getenv("DS4_PROFILE_TRACE");
        cached = (p && p[0]) ? 1 : 0;
    }
    return cached || g_stage_capture || g_record_enabled;
}

int ds4_profiler_event_snapshot(ds4_prof_event *out, int max) {
    if (!out || max <= 0) return 0;
    int n = 0;
    LOCK();
    const int count = g_events_count;
    const int start = (g_events_head - count + DS4_PEV_RING) % DS4_PEV_RING;
    for (int i = 0; i < count && n < max; i++)
        out[n++] = g_events[(start + i) % DS4_PEV_RING];
    UNLOCK();
    return n;
}

int ds4_profiler_dev_state_snapshot(ds4_prof_dev_state *out, int max) {
    if (!out || max <= 0) return 0;
    LOCK();
    const int n = g_dev_state_n < max ? g_dev_state_n : max;
    for (int i = 0; i < n; i++) out[i] = g_dev_state[i];
    UNLOCK();
    return n;
}

/* ---- records ------------------------------------------------------------- */

static void trace_write_record(const ds4_prof_record *rec);

static uint64_t counter_delta(uint64_t end, uint64_t start) {
    return end >= start ? end - start : 0;
}

int ds4_profiler_record_enabled(void) { return g_record_enabled; }
void ds4_profiler_set_record_enabled(int on) {
    g_record_enabled = on ? 1 : 0;
    if (on) {
        stats_thread_ensure();
        if (!getenv("DS4_PROFILE_TRACE")) g_stage_capture = 1;
    }
}

int ds4_profiler_token_begin_ex(ds4_prof_rec_kind kind, int token,
                                int pos, uint32_t n_tokens) {
    if (!g_record_enabled) return 0;
    LOCK();
    if (g_cur.open) { UNLOCK(); return 0; }
    UNLOCK();

    sys_sample sys;
    ds4_gpu_machine m;
    dev_sample devs;
    sys_sample_now(&sys);
    memset(&m, 0, sizeof(m));
    ds4_gpu_machine_snapshot(&m);
    memset(&devs, 0, sizeof(devs));
    dev_state_sample(devs.dev, &devs.dev_n);

    const uint64_t now = monotonic_ns();
    LOCK();
    if (g_cur.open) { UNLOCK(); return 0; }
    memset(&g_cur, 0, sizeof(g_cur));
    g_cur.open = 1;
    g_cur.seq = g_rec_next_seq++;
    g_cur.kind = kind;
    g_cur.token = token < 0 ? UINT32_MAX : (uint32_t)token;
    g_cur.pos = pos;
    g_cur.n_tokens = n_tokens ? n_tokens : 1;
    g_cur.start_ns = now;
    g_cur.event_first = g_event_next_seq;
    g_cur.expert_first = g_expert_next_seq;
    g_cur.sys0 = sys;
    g_cur.m0 = m;
    g_cur.dev0 = devs;
    for (int i = 0; i < g_stage_n; i++) stage_reset_accum(&g_stages[i]);
    /* The coarse host/device split is a decode statistic. Prefill chunks
     * have their own record and must not be folded into decode ms/token. */
    g_tok_open = kind == DS4_PREC_DECODE_TOKEN ? 1 : 0;
    g_tok_t0 = (double)now / 1e9;
    ds4_gpu_phase_snapshot(&g_tok_ph0);
    g_tok_prev_n = ds4_gpu_transfer_snapshot(g_tok_prev, DS4_PROFILER_MAX_DEV);
    UNLOCK();
    return 1;
}

void ds4_profiler_record_text(const char *text, size_t len) {
    if (!text || len == 0) return;
    size_t w = 0;
    LOCK();
    if (g_cur.open) {
        for (size_t i = 0; i < len && w + 1 < sizeof(g_cur.text); i++) {
            unsigned char c = (unsigned char)text[i];
            g_cur.text[w++] = (c == '\n' || c == '\r' || c == '\t' ||
                               c < 0x20) ? ' ' : (char)c;
        }
        g_cur.text[w] = '\0';
    }
    UNLOCK();
}

void ds4_profiler_token_begin(void) {
    ds4_profiler_token_begin_ex(DS4_PREC_DECODE_TOKEN, -1, -1, 1);
}

void ds4_profiler_token_end(uint64_t tokens) {
    if (!g_cur.open) return;
    if (tokens == 0) tokens = 1;

    sys_sample sys1;
    ds4_gpu_machine m1;
    dev_sample devs1;
    const uint64_t end_ns = monotonic_ns();
    sys_sample_now(&sys1);
    memset(&m1, 0, sizeof(m1));
    ds4_gpu_machine_snapshot(&m1);
    memset(&devs1, 0, sizeof(devs1));
    dev_state_sample(devs1.dev, &devs1.dev_n);

    ds4_prof_record rec;
    memset(&rec, 0, sizeof(rec));
    ds4_prof_token_rec legacy;
    memset(&legacy, 0, sizeof(legacy));
    uint32_t pool = 0, arena = 0, ssd = 0, cpu = 0;

    LOCK();
    if (!g_cur.open) { UNLOCK(); return; }
    g_cur.open = 0;
    rec.seq = g_cur.seq;
    rec.kind = (uint8_t)g_cur.kind;
    rec.closed = 1;
    rec.token = g_cur.token;
    rec.pos = g_cur.pos;
    rec.n_tokens = (uint32_t)tokens;
    memcpy(rec.text, g_cur.text, sizeof(rec.text));
    rec.start_ns = g_cur.start_ns;
    rec.end_ns = end_ns;
    rec.ms = (double)(end_ns - g_cur.start_ns) / 1e6;
    rec.stage_n = g_cur.stage_n;
    for (uint32_t i = 0; i < g_cur.stage_n; i++) rec.stages[i] = g_cur.stages[i];
    rec.event_first = (uint32_t)g_cur.event_first;
    rec.event_n = g_cur.event_n;
    rec.expert_first = (uint32_t)g_cur.expert_first;
    rec.expert_n = g_cur.expert_n;
    rec.dev_n = devs1.dev_n > g_cur.dev0.dev_n ? devs1.dev_n : g_cur.dev0.dev_n;
    if (rec.dev_n > DS4_PROF_REC_MAX_DEV) rec.dev_n = DS4_PROF_REC_MAX_DEV;
    for (uint32_t i = 0; i < rec.dev_n; i++) {
        const ds4_prof_dev_state *a = i < (uint32_t)g_cur.dev0.dev_n
            ? &g_cur.dev0.dev[i] : NULL;
        const ds4_prof_dev_state *b = i < (uint32_t)devs1.dev_n
            ? &devs1.dev[i] : NULL;
        ds4_prof_rec_dev *d = &rec.dev[i];
        d->device = b ? b->device : (a ? a->device : (int)i);
        d->util_start = a ? a->util_pct : 0;
        d->util_end = b ? b->util_pct : 0;
        d->mem_start = a ? a->mem_used : 0;
        d->mem_end = b ? b->mem_used : 0;
        d->mem_total = b ? b->mem_total : (a ? a->mem_total : 0);
    }
    rec.ssd_read_bytes = counter_delta(m1.ssd_read_bytes, g_cur.m0.ssd_read_bytes);
    rec.host_ram_used = sys1.ram_used;
    rec.host_ram_avail = sys1.ram_avail;
    rec.proc_cpu_ns = counter_delta(sys1.proc_cpu_ns, g_cur.sys0.proc_cpu_ns);
    rec.sys_cpu_busy_ns = counter_delta(sys1.sys_busy_ns, g_cur.sys0.sys_busy_ns);
    rec.sys_cpu_total_ns = counter_delta(sys1.sys_total_ns, g_cur.sys0.sys_total_ns);
    const int tiers = m1.tiers < DS4_PROF_MACHINE_TIERS ? m1.tiers : DS4_PROF_MACHINE_TIERS;
    for (int t = 0; t < tiers; t++) {
        rec.tier_weight_bytes[t] = counter_delta(m1.weight_bytes[t], g_cur.m0.weight_bytes[t]);
        rec.tier_activation_bytes[t] = counter_delta(m1.activation_bytes[t], g_cur.m0.activation_bytes[t]);
        rec.tier_tx_bytes[t] = counter_delta(m1.tx_bytes[t], g_cur.m0.tx_bytes[t]);
        rec.tier_rx_bytes[t] = counter_delta(m1.rx_bytes[t], g_cur.m0.rx_bytes[t]);
    }

    for (uint32_t i = 0; i < rec.expert_n; i++) {
        const uint64_t sq = g_cur.expert_first + i;
        const ds4_prof_expert_trace *x = &g_experts[sq % DS4_PTRACE_EXPERT_RING];
        if (x->rec_seq != rec.seq) continue;
        switch (x->source) {
        case DS4_PTRACE_SRC_POOL:  pool++;  break;
        case DS4_PTRACE_SRC_ARENA: arena++; break;
        case DS4_PTRACE_SRC_SSD:   ssd++;   break;
        default:                   cpu++;   break;
        }
    }

    g_records[rec.seq % DS4_PROF_REC_RING] = rec;
    g_rec_head = (int)((rec.seq + 1) % DS4_PROF_REC_RING);
    if (g_rec_count < DS4_PROF_REC_RING) g_rec_count++;
    for (int i = 0; i < g_stage_n; i++) stage_reset_accum(&g_stages[i]);

    legacy.token = rec.token == UINT32_MAX ? 0 : rec.token;
    legacy.ms = rec.ms / (double)tokens;
    legacy.events = rec.event_n;
    legacy.experts_pool = pool;
    legacy.experts_arena = arena;
    legacy.experts_ssd = ssd;
    legacy.experts_cpu = cpu;
    legacy.dev_n = devs1.dev_n;
    for (int i = 0; i < devs1.dev_n; i++) legacy.dev[i] = devs1.dev[i];
    g_token_recs[g_token_rec_head] = legacy;
    g_token_rec_head = (g_token_rec_head + 1) % DS4_PTREC_RING;
    if (g_token_rec_count < DS4_PTREC_RING) g_token_rec_count++;
    UNLOCK();

    dev_state_publish(devs1.dev, devs1.dev_n);
    trace_write_record(&rec);
    timing_note_token(tokens);
}

int ds4_profiler_token_rec_snapshot(ds4_prof_token_rec *out, int max) {
    if (!out || max <= 0) return 0;
    int n = 0;
    LOCK();
    const int count = g_token_rec_count;
    const int start = (g_token_rec_head - count + DS4_PTREC_RING) % DS4_PTREC_RING;
    for (int i = 0; i < count && n < max; i++)
        out[n++] = g_token_recs[(start + i) % DS4_PTREC_RING];
    UNLOCK();
    return n;
}

const char *ds4_profiler_record_kind_name(int kind) {
    switch (kind) {
    case DS4_PREC_PREFILL_CHUNK: return "prefill chunk";
    case DS4_PREC_PREFILL_TOKEN: return "prefill token";
    default: return "decode token";
    }
}

int ds4_profiler_record_list(ds4_prof_record_meta *out, int max) {
    if (!out || max <= 0) return 0;
    int n = 0;
    LOCK();
    for (uint64_t seq = (g_rec_count > 0
             ? g_rec_next_seq - (uint64_t)g_rec_count : g_rec_next_seq);
         seq < g_rec_next_seq && n < max; seq++) {
        const ds4_prof_record *r = &g_records[seq % DS4_PROF_REC_RING];
        if (r->seq != seq) continue;
        ds4_prof_record_meta *m = &out[n++];
        memset(m, 0, sizeof(*m));
        m->seq = r->seq;
        m->kind = r->kind;
        m->closed = r->closed;
        m->token = r->token;
        m->pos = r->pos;
        m->n_tokens = r->n_tokens;
        m->ms = r->ms;
        memcpy(m->text, r->text, sizeof(m->text));
        m->events = r->event_n;
        m->experts = r->expert_n;
        m->hot_stage = 0xFFFF;
        for (uint32_t i = 0; i < r->stage_n; i++) {
            if (m->hot_stage == 0xFFFF ||
                r->stages[i].total_ns > (uint64_t)(m->hot_ms * 1e6)) {
                m->hot_stage = r->stages[i].id;
                m->hot_ms = (double)r->stages[i].total_ns / 1e6;
            }
        }
        if (m->hot_stage == 0xFFFF) { m->hot_stage = 0; m->hot_ms = 0; }
        for (uint32_t i = 0; i < r->expert_n; i++) {
            const ds4_prof_expert_trace *x =
                &g_experts[(r->expert_first + i) % DS4_PTRACE_EXPERT_RING];
            if (x->rec_seq != r->seq) continue;
            switch (x->source) {
            case DS4_PTRACE_SRC_POOL: m->experts_pool++; break;
            case DS4_PTRACE_SRC_ARENA: m->experts_arena++; break;
            case DS4_PTRACE_SRC_SSD: m->experts_ssd++; break;
            default: m->experts_cpu++; break;
            }
        }
        for (uint32_t i = 0; i < r->dev_n; i++)
            if (r->dev[i].util_end > m->util_max) m->util_max = (uint8_t)r->dev[i].util_end;
    }
    UNLOCK();
    return n;
}

int ds4_profiler_record_get(uint64_t seq, ds4_prof_record *out) {
    if (!out) return 0;
    LOCK();
    const ds4_prof_record *r = &g_records[seq % DS4_PROF_REC_RING];
    if (r->seq != seq) { UNLOCK(); return 0; }
    *out = *r;
    UNLOCK();
    return 1;
}

int ds4_profiler_record_events(uint64_t seq, ds4_prof_event *out, int max) {
    if (!out || max <= 0) return 0;
    ds4_prof_record r;
    if (!ds4_profiler_record_get(seq, &r)) return 0;
    int n = 0;
    LOCK();
    for (uint32_t i = 0; i < r.event_n && n < max; i++) {
        const uint64_t sq = (uint64_t)r.event_first + i;
        const ds4_prof_event *e = &g_events[sq % DS4_PEV_RING];
        if (e->seq != sq || e->rec_seq != seq) continue;
        out[n++] = *e;
    }
    UNLOCK();
    return n;
}

int ds4_profiler_record_experts(uint64_t seq, ds4_prof_expert_trace *out, int max) {
    if (!out || max <= 0) return 0;
    ds4_prof_record r;
    if (!ds4_profiler_record_get(seq, &r)) return 0;
    int n = 0;
    LOCK();
    for (uint32_t i = 0; i < r.expert_n && n < max; i++) {
        const uint64_t sq = (uint64_t)r.expert_first + i;
        const ds4_prof_expert_trace *x = &g_experts[sq % DS4_PTRACE_EXPERT_RING];
        if (x->rec_seq != seq) continue;
        out[n++] = *x;
    }
    UNLOCK();
    return n;
}

void ds4_profiler_note_step(ds4_prof_step step, uint64_t ns,
                            uint64_t count, uint64_t bytes) {
    if ((unsigned)step >= (unsigned)DS4_PSTEP_COUNT) return;
    LOCK();
    g_steps[step].ns += ns;
    g_steps[step].count += count;
    g_steps[step].bytes += bytes;
    UNLOCK();
    if (!g_stage_capture) return;
    const char *cat = ds4_profiler_step_category(step);
    const uint64_t enc = !strcmp(cat, "wait") || !strcmp(cat, "transfer") ? 0 : ns;
    const uint64_t wait = enc == 0 ? ns : 0;
    stage_note_ex(cat, ds4_profiler_step_id(step),
                  ds4_profiler_step_label(step),
                  ds4_profiler_step_hint(step), cat, 0,
                  ns, enc, wait, bytes);
}

void ds4_profiler_step_snapshot(ds4_prof_step_stat *out, int n) {
    if (!out || n <= 0) return;
    LOCK();
    for (int i = 0; i < n && i < (int)DS4_PSTEP_COUNT; i++) out[i] = g_steps[i];
    UNLOCK();
}

const char *ds4_profiler_step_id(ds4_prof_step step) {
    return (unsigned)step < (unsigned)DS4_PSTEP_COUNT ? k_steps[step].id : "";
}
const char *ds4_profiler_step_label(ds4_prof_step step) {
    return (unsigned)step < (unsigned)DS4_PSTEP_COUNT ? k_steps[step].label : "";
}
const char *ds4_profiler_step_hint(ds4_prof_step step) {
    return (unsigned)step < (unsigned)DS4_PSTEP_COUNT ? k_steps[step].hint : "";
}
const char *ds4_profiler_step_category(ds4_prof_step step) {
    return (unsigned)step < (unsigned)DS4_PSTEP_COUNT ? k_steps[step].category : "";
}

void ds4_profiler_reset_timing(void) {
    LOCK();
    memset(&g_timing, 0, sizeof(g_timing));
    memset(g_steps, 0, sizeof(g_steps));
    memset(g_stages, 0, sizeof(g_stages));
    g_stage_n = 0;
    g_experts_head = 0;
    g_experts_count = 0;
    g_expert_next_seq = 0;
    memset(&g_cur, 0, sizeof(g_cur));
    g_rec_next_seq = 0;
    g_rec_head = 0;
    g_rec_count = 0;
    memset(g_records, 0, sizeof(g_records));
    g_events_head = 0;
    g_events_count = 0;
    g_event_next_seq = 0;
    memset(g_events, 0, sizeof(g_events));
    g_trace_token = 0;
    g_token_rec_head = 0;
    g_token_rec_count = 0;
    memset(g_token_recs, 0, sizeof(g_token_recs));
    g_tok_open = 0;
    UNLOCK();
}

/* ---- coarse per-token timing -------------------------------------------- */

static void timing_note_token(uint64_t tokens) {
    const double now = monotonic_sec();
    if (!g_tok_open) return;
    g_tok_open = 0;
    if (tokens == 0) tokens = 1;

    ds4_gpu_phases ph1;
    ds4_gpu_phase_snapshot(&ph1);

    ds4_gpu_transfer cur[DS4_PROFILER_MAX_DEV];
    const int n = ds4_gpu_transfer_snapshot(cur, DS4_PROFILER_MAX_DEV);
    const int m = n < g_tok_prev_n ? n : g_tok_prev_n;

    double weight_ms = 0.0, act_ms = 0.0;
    for (int t = 0; t < m; t++) {
        const uint64_t wb = cur[t].weight_into >= g_tok_prev[t].weight_into
            ? cur[t].weight_into - g_tok_prev[t].weight_into : 0;
        const uint64_t ab = cur[t].activation_io >= g_tok_prev[t].activation_io
            ? cur[t].activation_io - g_tok_prev[t].activation_io : 0;
        const uint64_t bps = ds4_gpu_pcie_peak_bps(t);
        /* Bytes to time at the measured link rate. The transfers overlap
         * compute, so these are link-time equivalents; subtracting them from
         * the device time leaves a floor on kernel compute. */
        if (bps != 0) {
            weight_ms += (double)wb / (double)bps * 1000.0;
            act_ms    += (double)ab / (double)bps * 1000.0;
        }
    }

    const double per_token = (now - g_tok_t0) * 1000.0 / (double)tokens;
    const double tk = (double)tokens;
#define PH_DELTA(field)                                                     \
    ((ph1.field >= g_tok_ph0.field)                                         \
        ? (double)(ph1.field - g_tok_ph0.field) / 1e6 / tk : 0.0)
    /* Every term is measured elapsed time, so the parts sum to the total.
     * device_wait is the CPU's blocked-on-device time; host is the remainder
     * (graph encode, launches, sampling, detokenize). */
    double device_wait = PH_DELTA(device_wait_ns);
    double d_upload    = PH_DELTA(upload_wait_ns);
    double d_readback  = PH_DELTA(readback_ns);
    double h_read      = PH_DELTA(host_read_ns);
#undef PH_DELTA
    if (device_wait > per_token) device_wait = per_token;
    if (device_wait < 0.0) device_wait = 0.0;
    double host = per_token - device_wait;
    if (host < 0.0) host = 0.0;
    /* Split the device time into upload, readback and the rest of the syncs. */
    if (d_upload > device_wait) d_upload = device_wait;
    if (d_readback > device_wait - d_upload) d_readback = device_wait - d_upload;
    double d_sync = device_wait - d_upload - d_readback;
    if (d_sync < 0.0) d_sync = 0.0;
    /* Split the host time into the expert read/staging wait and the rest. */
    if (h_read > host) h_read = host;
    if (h_read < 0.0) h_read = 0.0;

    LOCK();
    stat_add(&g_timing.per_token_ms, per_token);
    stat_add(&g_timing.host_ms, host);
    stat_add(&g_timing.host_read_ms, h_read);
    stat_add(&g_timing.device_wait_ms, device_wait);
    stat_add(&g_timing.device_upload_ms, d_upload);
    stat_add(&g_timing.device_readback_ms, d_readback);
    stat_add(&g_timing.device_sync_ms, d_sync);
    stat_add(&g_timing.weight_transfer_ms, weight_ms);
    stat_add(&g_timing.activation_transfer_ms, act_ms);
    stat_add(&g_timing.tokens_per_s, per_token > 0.0 ? 1000.0 / per_token : 0.0);
    g_timing.tokens += tokens;
    UNLOCK();
}

void ds4_profiler_note_prefill(double ms, uint64_t tokens, uint32_t chunks) {
    if (tokens == 0) return;
    LOCK();
    stat_add(&g_timing.prefill_ms, ms);
    /* prefill_ms.n doubles as the chunk counter: per-chunk averages divide the
     * wall clock by it, so a whole-prompt measurement must cover every chunk it
     * timed rather than count as one. */
    if (chunks > 1u) g_timing.prefill_ms.n += chunks - 1u;
    stat_add(&g_timing.prefill_tokens_per_s,
             ms > 0.0 ? (double)tokens * 1000.0 / ms : 0.0);
    g_timing.prefill_tokens += tokens;
    UNLOCK();
}

void ds4_profiler_timing_snapshot(ds4_profiler_timing *out) {
    if (!out) return;
    LOCK();
    *out = g_timing;
    UNLOCK();
}

/* ---- views / JSONL ------------------------------------------------------- */

static int stage_key_copy(int id, char *out, size_t cap) {
    int ok = 0;
    LOCK();
    if (id >= 0 && id < g_stage_n) {
        snprintf(out, cap, "%s", g_stages[id].key);
        ok = 1;
    }
    UNLOCK();
    return ok;
}

int ds4_profiler_view(ds4_prof_view *out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    stats_snapshot_cached(&out->stats);
    sys_sample sys;
    sys_sample_now(&sys);
    out->ssd_read_bytes = 0;
    ds4_gpu_machine m;
    memset(&m, 0, sizeof(m));
    ds4_gpu_machine_snapshot(&m);
    out->ssd_read_bytes = m.ssd_read_bytes;
    out->host_ram_used = sys.ram_used;
    out->host_ram_avail = sys.ram_avail;
    out->proc_cpu_ns = sys.proc_cpu_ns;
    dev_sample devs;
    memset(&devs, 0, sizeof(devs));
    dev_state_sample(devs.dev, &devs.dev_n);
    out->dev_n = devs.dev_n;
    for (int i = 0; i < devs.dev_n; i++) out->dev[i] = devs.dev[i];

    LOCK();
    out->timing = g_timing;
    out->host = g_host;
    out->n_stages = g_stage_n;
    for (int i = 0; i < g_stage_n; i++) {
        const stage_slot *s = &g_stages[i];
        ds4_prof_stage_view *v = &out->stages[i];
        snprintf(v->key, sizeof(v->key), "%s", s->key);
        snprintf(v->label, sizeof(v->label), "%s", s->label);
        snprintf(v->part, sizeof(v->part), "%s", s->part);
        snprintf(v->name, sizeof(v->name), "%s", s->name);
        snprintf(v->category, sizeof(v->category), "%s", s->category);
        snprintf(v->hint, sizeof(v->hint), "%s", s->hint);
        v->total_ms = s->total_ms;
        v->encode_ms = s->encode_ms;
        v->wait_ms = s->wait_ms;
        v->calls = s->count;
        v->bytes = s->bytes;
        for (int k = 0; k < DS4_PREC_KIND_MAX; k++) {
            v->kind[k].total_ms = (double)s->k_total_ns[k] / 1e6;
            v->kind[k].encode_ms = (double)s->k_encode_ns[k] / 1e6;
            v->kind[k].wait_ms = (double)s->k_wait_ns[k] / 1e6;
            v->kind[k].calls = s->k_count[k];
            v->kind[k].bytes = s->k_bytes[k];
        }
    }
    UNLOCK();
    out->n_records = ds4_profiler_record_list(out->records, DS4_PROF_REC_RING);
    return out->n_stages;
}

static FILE *trace_file(void) {
    static FILE *fp;
    static int opened;
    if (!opened) {
        opened = 1;
        const char *p = getenv("DS4_PROFILE_TRACE");
        if (p && p[0]) {
            if (p[0] == '1' && p[1] == '\0') p = "ds4-trace.jsonl";
            fp = fopen(p, "a");
        }
    }
    return fp;
}

/* One self-contained JSON object per decoded token or prefilled chunk. This is
 * deliberately the full record: stages (total/encode/wait), ordered events,
 * expert uses, all-device state at both ends, and byte/CPU deltas. */
static void trace_write_record(const ds4_prof_record *rec) {
    FILE *fp = trace_file();
    if (!fp || !rec) return;
    ds4_prof_event *ev = rec->event_n
        ? malloc((size_t)rec->event_n * sizeof(*ev)) : NULL;
    ds4_prof_expert_trace *ex = rec->expert_n
        ? malloc((size_t)rec->expert_n * sizeof(*ex)) : NULL;
    const int evn = ev ? ds4_profiler_record_events(rec->seq, ev, (int)rec->event_n) : 0;
    const int exn = ex ? ds4_profiler_record_experts(rec->seq, ex, (int)rec->expert_n) : 0;

    fprintf(fp, "{\"seq\":%llu,\"kind\":\"%s\",\"token\":%u,\"pos\":%d,"
                "\"tokens\":%u,\"ms\":%.4f,\"text\":\"%s\","
                "\"start_ns\":%llu,\"end_ns\":%llu,"
                "\"ssd_read_bytes\":%llu,\"host_ram_used\":%llu,"
                "\"host_ram_avail\":%llu,\"proc_cpu_ms\":%.3f,"
                "\"sys_cpu_pct\":%.2f,"
                "\"stages\":[",
            (unsigned long long)rec->seq,
            ds4_profiler_record_kind_name(rec->kind), rec->token, rec->pos,
            rec->n_tokens, rec->ms, rec->text,
            (unsigned long long)rec->start_ns,
            (unsigned long long)rec->end_ns,
            (unsigned long long)rec->ssd_read_bytes,
            (unsigned long long)rec->host_ram_used,
            (unsigned long long)rec->host_ram_avail,
            (double)rec->proc_cpu_ns / 1e6,
            rec->sys_cpu_total_ns
                ? 100.0 * (double)rec->sys_cpu_busy_ns / (double)rec->sys_cpu_total_ns
                : 0.0);
    for (uint32_t i = 0; i < rec->stage_n; i++) {
        char key[48] = {0};
        stage_key_copy(rec->stages[i].id, key, sizeof(key));
        fprintf(fp, "%s{\"id\":%u,\"key\":\"%s\",\"total_ms\":%.4f,"
                    "\"encode_ms\":%.4f,\"wait_ms\":%.4f,\"calls\":%llu,"
                    "\"bytes\":%llu}",
                i ? "," : "", (unsigned)rec->stages[i].id, key,
                (double)rec->stages[i].total_ns / 1e6,
                (double)rec->stages[i].encode_ns / 1e6,
                (double)rec->stages[i].wait_ns / 1e6,
                (unsigned long long)rec->stages[i].count,
                (unsigned long long)rec->stages[i].bytes);
    }
    fprintf(fp, "],\"events\":[");
    for (int i = 0; i < evn; i++) {
        char skey[48] = {0};
        if (ev[i].kind == DS4_PEV_STAGE)
            stage_key_copy(ev[i].stage, skey, sizeof(skey));
        fprintf(fp, "%s{\"us\":%u,\"ns\":%u,\"k\":\"%s\",\"stage\":\"%s\","
                    "\"layer\":%u,\"expert\":%u,\"dev\":%u,\"bytes\":%u}",
                i ? "," : "", ev[i].t_us, ev[i].dur_ns,
                ds4_profiler_event_kind(ev[i].kind), skey,
                (unsigned)ev[i].layer, (unsigned)ev[i].expert,
                (unsigned)ev[i].device, ev[i].bytes);
    }
    fprintf(fp, "],\"experts\":[");
    for (int i = 0; i < exn; i++) {
        fprintf(fp, "%s[%u,%u,%u,%u,%u,%u]", i ? "," : "",
                (unsigned)ex[i].layer, (unsigned)ex[i].expert,
                (unsigned)ex[i].source, (unsigned)ex[i].src_gpu,
                (unsigned)ex[i].dest_gpu, ex[i].bytes);
    }
    fprintf(fp, "],\"devices\":[");
    for (uint32_t i = 0; i < rec->dev_n; i++) {
        fprintf(fp, "%s{\"dev\":%d,\"util_start\":%u,\"util_end\":%u,"
                    "\"mem_start\":%llu,\"mem_end\":%llu,\"mem_total\":%llu}",
                i ? "," : "", rec->dev[i].device, rec->dev[i].util_start,
                rec->dev[i].util_end,
                (unsigned long long)rec->dev[i].mem_start,
                (unsigned long long)rec->dev[i].mem_end,
                (unsigned long long)rec->dev[i].mem_total);
    }
    fprintf(fp, "],\"tiers\":[");
    for (int t = 0; t < DS4_PROF_MACHINE_TIERS; t++) {
        if (rec->tier_weight_bytes[t] == 0 && rec->tier_activation_bytes[t] == 0 &&
            rec->tier_tx_bytes[t] == 0 && rec->tier_rx_bytes[t] == 0) continue;
        fprintf(fp, "%s{\"tier\":%d,\"weight_bytes\":%llu,"
                    "\"activation_bytes\":%llu,\"tx_bytes\":%llu,\"rx_bytes\":%llu}",
                t ? "," : "", t,
                (unsigned long long)rec->tier_weight_bytes[t],
                (unsigned long long)rec->tier_activation_bytes[t],
                (unsigned long long)rec->tier_tx_bytes[t],
                (unsigned long long)rec->tier_rx_bytes[t]);
    }
    fprintf(fp, "]}\n");
    fflush(fp);
    free(ev);
    free(ex);
}

/* ---- JSON ---------------------------------------------------------------- */

typedef struct {
    char  *buf;
    size_t cap;
    size_t n;
} jbuf;

static void jraw(jbuf *j, const char *s) {
    const size_t len = strlen(s);
    if (j->n + len >= j->cap) {
        if (j->n < j->cap) {   /* keep the buffer a valid C string */
            const size_t room = j->cap - 1u - j->n;
            memcpy(j->buf + j->n, s, room);
            j->n = j->cap - 1u;
        }
        return;
    }
    memcpy(j->buf + j->n, s, len);
    j->n += len;
}

static void jfmt(jbuf *j, const char *fmt, ...) {
    if (j->n + 1u >= j->cap) return;
    va_list ap;
    va_start(ap, fmt);
    const int w = vsnprintf(j->buf + j->n, j->cap - j->n, fmt, ap);
    va_end(ap);
    if (w < 0) return;
    if ((size_t)w >= j->cap - j->n) j->n = j->cap - 1u;
    else j->n += (size_t)w;
}

static void jstr(jbuf *j, const char *s) {
    jraw(j, "\"");
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
        case '"':  jraw(j, "\\\""); break;
        case '\\': jraw(j, "\\\\"); break;
        case '\n': jraw(j, "\\n");  break;
        case '\r': jraw(j, "\\r");  break;
        case '\t': jraw(j, "\\t");  break;
        default:
            if (*p < 0x20) jfmt(j, "\\u%04x", *p);
            else { char c[2] = { (char)*p, 0 }; jraw(j, c); }
        }
    }
    jraw(j, "\"");
}

/* Terminate a jbuf and return its length. jraw does not NUL-terminate, so
 * every dump that can end with a literal has to close the string itself. */
static size_t jdone(jbuf *j) {
    if (j->cap) j->buf[j->n < j->cap ? j->n : j->cap - 1u] = '\0';
    return j->n;
}

static void jstat(jbuf *j, const char *key, const ds4_prof_stat *s) {
    jfmt(j, "\"%s\":{\"last\":%.3f,\"min\":%.3f,\"max\":%.3f,\"average\":%.3f,"
             "\"samples\":%llu}",
         key, s->last, s->n ? s->min : 0.0, s->max, ds4_prof_avg(s),
         (unsigned long long)s->n);
}

static void jbytes(jbuf *j, const char *key, uint64_t v) {
    jfmt(j, "\"%s\":%llu", key, (unsigned long long)v);
}

static uint64_t host_ram_total_bytes(void) {
#if !defined(_WIN32)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0) return (uint64_t)pages * (uint64_t)psize;
#endif
    return 0;
}

/* VRAM composition. Every consumer we track gets its own field; the residual
 * ("other") is then only external processes (desktop, browser, the CUDA
 * driver/context) plus anything untracked. `tensor` and `pool` can overlap the
 * specific parts and reserved can exceed used, so they are trimmed first so the
 * parts sum to the measured used VRAM. */
typedef struct {
    uint64_t dense, arena, kv, experts, scratch, temp, tensors, pool, other, sum;
} vram_parts;

static vram_parts vram_split(const ds4_profiler_device *d, uint64_t host_kv) {
    vram_parts p;
    memset(&p, 0, sizeof(p));
    const uint64_t kv = d->device_id == 0 ? host_kv : 0;  /* KV lives on tier 0 */
    uint64_t wgt = d->dense_bytes + d->derived_bytes;
    uint64_t tens = d->tensor_bytes > kv ? d->tensor_bytes - kv : 0;
    uint64_t pool = d->mempool_bytes;
    const uint64_t base = wgt + d->arena_bytes + kv + d->expert_bytes +
                          d->scratch_bytes + d->temp_bytes;
    uint64_t over = base > d->vram_used_bytes ? base - d->vram_used_bytes : 0;
    if (over != 0) { const uint64_t t = tens < over ? tens : over; tens -= t; over -= t; }
    if (over != 0) {
        const uint64_t q = pool < over ? pool : over;
        pool -= q; over -= q;
        if (over != 0) wgt = wgt > over ? wgt - over : 0;
    }
    const uint64_t known = wgt + d->arena_bytes + kv + d->expert_bytes +
                           d->scratch_bytes + d->temp_bytes + tens + pool;
    p.dense = wgt;
    p.arena = d->arena_bytes;
    p.kv = kv;
    p.experts = d->expert_bytes;
    p.scratch = d->scratch_bytes;
    p.temp = d->temp_bytes;
    p.tensors = tens;
    p.pool = pool;
    p.other = d->vram_used_bytes > known ? d->vram_used_bytes - known : 0;
    p.sum = d->vram_used_bytes;
    return p;
}

size_t ds4_profiler_json(char *buf, size_t buflen) {
    if (!buf || buflen == 0) return 0;
    jbuf j = { buf, buflen, 0 };
    buf[0] = '\0';

    ds4_profiler_stats s;
    stats_snapshot_cached(&s);

    ds4_profiler_timing t;
    ds4_profiler_host_stats h;
    LOCK();
    t = g_timing;
    h = g_host;
    UNLOCK();
    if (h.host_ram_total_bytes == 0) h.host_ram_total_bytes = host_ram_total_bytes();

    jraw(&j, "{");

    /* Expert distribution across the three tiers. */
    uint64_t gpu_experts = 0;
    for (int i = 0; i < s.n_devices; i++) gpu_experts += s.dev[i].expert_count;
    const uint64_t total = h.total_experts;
    uint64_t ram = s.host_experts_resident;
    uint64_t ram_bytes = h.ram_bytes;
    if (s.host_arena_capacity_bytes != 0) {
        /* The pinned arena is the host expert cache; report what it actually
         * holds rather than the page-cache budget that only guides reads. */
        ram = s.host_arena_experts;
        ram_bytes = s.host_arena_used_bytes;
    }
    const uint64_t ssd = total > gpu_experts + ram ? total - gpu_experts - ram : 0;
    jraw(&j, "\"expert_distribution\":{");
    jbytes(&j, "total", total);      jraw(&j, ",");
    jbytes(&j, "on_gpu", gpu_experts); jraw(&j, ",");
    jbytes(&j, "in_host_ram", ram);  jraw(&j, ",");
    jbytes(&j, "on_ssd", ssd);       jraw(&j, ",");
    jbytes(&j, "bytes_per_expert", h.expert_bytes);
    jraw(&j, "},");

    jraw(&j, "\"host\":{");
    jfmt(&j, "\"has_ssd\":%d,", h.has_ssd ? 1 : 0);
    jfmt(&j, "\"has_ram\":%d,",
         (h.has_ram || s.host_arena_capacity_bytes != 0) ? 1 : 0);
    jbytes(&j, "model_bytes", h.model_bytes);       jraw(&j, ",");
    jbytes(&j, "ram_bytes", ram_bytes);             jraw(&j, ",");
    jbytes(&j, "ram_total_bytes", h.host_ram_total_bytes); jraw(&j, ",");
    jbytes(&j, "ram_experts", ram);                 jraw(&j, ",");
    jbytes(&j, "arena_capacity_bytes", s.host_arena_capacity_bytes); jraw(&j, ",");
    jbytes(&j, "arena_used_bytes", s.host_arena_used_bytes); jraw(&j, ",");
    jbytes(&j, "arena_experts", s.host_arena_experts); jraw(&j, ",");
    jbytes(&j, "arena_hits", s.host_arena_hits);    jraw(&j, ",");
    jbytes(&j, "arena_misses", s.host_arena_misses); jraw(&j, ",");
    jbytes(&j, "arena_admissions", s.host_arena_admissions); jraw(&j, ",");
    jbytes(&j, "arena_evictions", s.host_arena_evictions); jraw(&j, ",");
    jbytes(&j, "arena_evicted_fetches", s.host_arena_evicted_fetches); jraw(&j, ",");
    jbytes(&j, "kv_bytes", h.kv_bytes);             jraw(&j, ",");
    jbytes(&j, "ssd_read_bytes", s.ssd_read_bytes); jraw(&j, ",");
    jraw(&j, "\"dense\":[");
    for (int c = 0; c < h.dense_n && c < DS4_PROFILER_DENSE_CATS; c++) {
        if (c) jraw(&j, ",");
        jraw(&j, "{\"name\":");
        jstr(&j, h.dense[c].name);
        jfmt(&j, ",\"bytes\":%llu}", (unsigned long long)h.dense[c].bytes);
    }
    jraw(&j, "]},");

    /* Timing. */
    jraw(&j, "\"timing\":{");
    jstat(&j, "per_token_ms", &t.per_token_ms);              jraw(&j, ",");
    jstat(&j, "host_ms", &t.host_ms);                        jraw(&j, ",");
    jstat(&j, "host_read_ms", &t.host_read_ms);              jraw(&j, ",");
    jstat(&j, "device_wait_ms", &t.device_wait_ms);          jraw(&j, ",");
    jstat(&j, "device_upload_ms", &t.device_upload_ms);      jraw(&j, ",");
    jstat(&j, "device_readback_ms", &t.device_readback_ms);  jraw(&j, ",");
    jstat(&j, "device_sync_ms", &t.device_sync_ms);          jraw(&j, ",");
    jstat(&j, "weight_transfer_ms", &t.weight_transfer_ms);  jraw(&j, ",");
    jstat(&j, "activation_transfer_ms", &t.activation_transfer_ms); jraw(&j, ",");
    jstat(&j, "tokens_per_s", &t.tokens_per_s);              jraw(&j, ",");
    jstat(&j, "prefill_ms", &t.prefill_ms);                  jraw(&j, ",");
    jstat(&j, "prefill_tokens_per_s", &t.prefill_tokens_per_s); jraw(&j, ",");
    jfmt(&j, "\"tokens\":%llu", (unsigned long long)t.tokens);
    jfmt(&j, ",\"prefill_tokens\":%llu", (unsigned long long)t.prefill_tokens);
    jraw(&j, "},");

    /* Per-step timeline: one row per kind of cost, carrying its own label,
     * category and tooltip so the dashboard needs no second copy of the table.
     * Rows with no samples are omitted rather than shown as zero. */
    jraw(&j, "\"steps\":[");
    {
        ds4_prof_step_stat st[DS4_PSTEP_COUNT];
        ds4_profiler_step_snapshot(st, DS4_PSTEP_COUNT);
        const double tk = t.tokens ? (double)t.tokens : 0.0;
        int first = 1;
        for (int i = 0; i < DS4_PSTEP_COUNT; i++) {
            if (st[i].ns == 0 && st[i].count == 0 && st[i].bytes == 0) continue;
            const double ms = tk > 0 ? (double)st[i].ns / 1e6 / tk : 0.0;
            if (!first) jraw(&j, ",");
            first = 0;
            jraw(&j, "{");
            jraw(&j, "\"id\":");       jstr(&j, ds4_profiler_step_id((ds4_prof_step)i));
            jraw(&j, ",\"label\":");   jstr(&j, ds4_profiler_step_label((ds4_prof_step)i));
            jraw(&j, ",\"category\":");jstr(&j, ds4_profiler_step_category((ds4_prof_step)i));
            jraw(&j, ",\"hint\":");    jstr(&j, ds4_profiler_step_hint((ds4_prof_step)i));
            jfmt(&j, ",\"ms_per_token\":%.3f", ms);
            jfmt(&j, ",\"pct_of_token\":%.1f",
                 (t.per_token_ms.sum > 0.0 && tk > 0)
                     ? 100.0 * ms / t.per_token_ms.sum : 0.0);
            jfmt(&j, ",\"calls_per_token\":%.1f", tk > 0 ? (double)st[i].count / tk : 0.0);
            jfmt(&j, ",\"bytes_per_token\":%.0f", tk > 0 ? (double)st[i].bytes / tk : 0.0);
            jfmt(&j, ",\"gb_per_s\":%.2f",
                 st[i].ns > 0 ? (double)st[i].bytes / (double)st[i].ns : 0.0);
            jraw(&j, "}");
        }
    }
    jraw(&j, "],");

    /* Stage rows: one row per scheduler stage, carrying its measured
     * CPU-launch / device-wait split. Never merged with another stage. */
    jraw(&j, "\"stage_rows\":[");
    {
        ds4_prof_view view;
        memset(&view, 0, sizeof(view));
        (void)ds4_profiler_view(&view);
        const int n = view.n_stages < DS4_PROF_MAX_STAGES
            ? view.n_stages : DS4_PROF_MAX_STAGES;
        const double tk = t.tokens ? (double)t.tokens : 0.0;
        /* Kind denominators: decode shares divide by decode tokens, prefill
         * shares by prefill tokens. Mixing them was what made the shares of a
         * label sum past 100%. */
        double kind_tokens[DS4_PREC_KIND_MAX];
        kind_tokens[DS4_PREC_DECODE_TOKEN] = tk;
        kind_tokens[DS4_PREC_PREFILL_CHUNK] = (double)t.prefill_tokens;
        kind_tokens[DS4_PREC_PREFILL_TOKEN] = tk;
        double kind_total_ms[DS4_PREC_KIND_MAX] = {0};
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < DS4_PREC_KIND_MAX; k++) {
                kind_total_ms[k] += view.stages[i].kind[k].total_ms;
            }
        }
        int first = 1;
        for (int i = 0; i < n; i++) {
            const ds4_prof_stage_view *sv = &view.stages[i];
            if (sv->total_ms.n == 0 && sv->calls == 0 && sv->bytes == 0) continue;
            if (!first) jraw(&j, ",");
            first = 0;
            jraw(&j, "{");
            jraw(&j, "\"key\":");      jstr(&j, sv->key);
            jraw(&j, ",\"label\":");   jstr(&j, sv->label);
            jraw(&j, ",\"category\":");jstr(&j, sv->category);
            jraw(&j, ",\"hint\":");    jstr(&j, sv->hint);
            /* Per-kind rows: the same boundary measured on a decode token and
             * on a prefill chunk keeps its own average and its own share. */
            jraw(&j, ",\"kinds\":{");
            for (int k = 0; k < DS4_PREC_KIND_MAX; k++) {
                const ds4_prof_stage_kind *kv = &sv->kind[k];
                if (kv->calls == 0 && kv->total_ms <= 0.0) continue;
                const double kt = kind_tokens[k];
                if (k > 0) jraw(&j, ",");
                jraw(&j, "\"");
                jraw(&j, k == DS4_PREC_DECODE_TOKEN ? "decode"
                     : k == DS4_PREC_PREFILL_CHUNK ? "prefill_chunk"
                                                   : "prefill_token");
                jraw(&j, "\":{");
                jfmt(&j, "\"ms\":%.3f", kv->total_ms);
                jfmt(&j, ",\"ms_per_unit\":%.4f", kt > 0 ? kv->total_ms / kt : 0.0);
                jfmt(&j, ",\"encode_ms_per_unit\":%.4f", kt > 0 ? kv->encode_ms / kt : 0.0);
                jfmt(&j, ",\"wait_ms_per_unit\":%.4f", kt > 0 ? kv->wait_ms / kt : 0.0);
                jfmt(&j, ",\"share_pct\":%.2f",
                     kind_total_ms[k] > 0.0
                         ? 100.0 * kv->total_ms / kind_total_ms[k] : 0.0);
                jfmt(&j, ",\"calls\":%llu", (unsigned long long)kv->calls);
                jfmt(&j, ",\"bytes\":%llu", (unsigned long long)kv->bytes);
                jraw(&j, "}");
            }
            jraw(&j, "},");
            jfmt(&j, "\"total_ms_per_token\":%.4f",
                 tk > 0 ? sv->total_ms.sum / tk : 0.0);
            jfmt(&j, ",\"encode_ms_per_token\":%.4f",
                 tk > 0 ? sv->encode_ms.sum / tk : 0.0);
            jfmt(&j, ",\"wait_ms_per_token\":%.4f",
                 tk > 0 ? sv->wait_ms.sum / tk : 0.0);
            jfmt(&j, ",\"last_ms\":%.4f", sv->total_ms.last);
            jfmt(&j, ",\"min_ms\":%.4f", sv->total_ms.n ? sv->total_ms.min : 0.0);
            jfmt(&j, ",\"max_ms\":%.4f", sv->total_ms.max);
            jfmt(&j, ",\"calls_per_token\":%.2f",
                 tk > 0 ? (double)sv->calls / tk : 0.0);
            jfmt(&j, ",\"bytes_per_token\":%.0f",
                 tk > 0 ? (double)sv->bytes / tk : 0.0);
            jfmt(&j, ",\"gb_per_s\":%.3f",
                 sv->total_ms.sum > 0.0
                     ? (double)sv->bytes / (sv->total_ms.sum * 1e6) : 0.0);
            jraw(&j, "}");
        }
    }
    jraw(&j, "],");

    /* Expert trace: the last two tokens verbatim, one record per expert use,
     * with its source and destination so the dashboard can show residency and
     * transfer without reading the engine's internals. */
    jraw(&j, "\"expert_trace\":[");
    {
        const int cap = 4096;
        ds4_prof_expert_trace *tr = malloc((size_t)cap * sizeof(*tr));
        const int n = tr ? ds4_profiler_trace_snapshot(tr, cap) : 0;
        uint32_t last_tokens[2] = {0, 0};
        int seen = 0;
        for (int i = n - 1; i >= 0 && seen < 2; i--) {
            if (seen == 0 || tr[i].token != last_tokens[seen - 1]) {
                last_tokens[seen++] = tr[i].token;
            }
        }
        int first = 1;
        for (int i = 0; i < n; i++) {
            const uint32_t tk = tr[i].token;
            if (tk != last_tokens[0] && (seen < 2 || tk != last_tokens[1])) continue;
            if (!first) jraw(&j, ",");
            first = 0;
            jfmt(&j, "{\"token\":%u,\"layer\":%u,\"expert\":%u,\"source\":%u,"
                     "\"src_gpu\":%u,\"dest_gpu\":%u,\"bytes\":%u}",
                 tk, (unsigned)tr[i].layer, (unsigned)tr[i].expert,
                 (unsigned)tr[i].source, (unsigned)tr[i].src_gpu,
                 (unsigned)tr[i].dest_gpu, (unsigned)tr[i].bytes);
        }
        free(tr);
    }
    jraw(&j, "],");

    /* Compact record list. The TUI reads full records directly; the web page
     * only needs this list plus events for the newest record. */
    jraw(&j, "\"records\":[");
    {
        ds4_prof_record_meta recs[DS4_PROF_REC_RING];
        const int n = ds4_profiler_record_list(recs, DS4_PROF_REC_RING);
        for (int i = 0; i < n; i++) {
            const ds4_prof_record_meta *r = &recs[i];
            if (i) jraw(&j, ",");
            jfmt(&j, "{\"seq\":%llu,\"kind\":\"%s\",\"token\":%u,\"pos\":%d,"
                     "\"tokens\":%u,\"ms\":%.3f,\"events\":%u,\"experts\":%u,"
                     "\"hot_stage\":%u,\"hot_ms\":%.3f,"
                     "\"pool\":%u,\"arena\":%u,\"ssd\":%u,\"cpu\":%u,"
                     "\"util_max\":%u,\"text\":\"%s\"}",
                 (unsigned long long)r->seq,
                 ds4_profiler_record_kind_name(r->kind), r->token, r->pos,
                 r->n_tokens, r->ms, r->events, r->experts, r->hot_stage,
                 r->hot_ms, r->experts_pool, r->experts_arena,
                 r->experts_ssd, r->experts_cpu, r->util_max, r->text);
        }
    }
    jraw(&j, "],");

    /* Per-token records + the newest token's ordered events: the dashboard's
     * token list and timeline read from here. */
    jraw(&j, "\"tokens\":[");
    {
        ds4_prof_token_rec recs[DS4_PTREC_RING];
        const int n = ds4_profiler_token_rec_snapshot(recs, DS4_PTREC_RING);
        uint32_t newest = 0;
        for (int i = 0; i < n; i++) {
            const ds4_prof_token_rec *r = &recs[i];
            if (i) jraw(&j, ",");
            jfmt(&j, "{\"token\":%u,\"ms\":%.2f,\"events\":%u,"
                     "\"pool\":%u,\"arena\":%u,\"ssd\":%u,\"cpu\":%u,"
                     "\"compute\":%u,\"devices\":[",
                 r->token, r->ms, r->events, r->experts_pool,
                 r->experts_arena, r->experts_ssd, r->experts_cpu,
                 r->compute_ops);
            for (int d = 0; d < r->dev_n; d++) {
                if (d) jraw(&j, ",");
                jfmt(&j, "{\"dev\":%d,\"name\":\"%s\",\"util\":%u,"
                         "\"mem_used\":%llu,\"mem_total\":%llu}",
                     r->dev[d].device, r->dev[d].name, r->dev[d].util_pct,
                     (unsigned long long)r->dev[d].mem_used,
                     (unsigned long long)r->dev[d].mem_total);
            }
            jraw(&j, "]}");
            newest = r->token;
        }
        jraw(&j, "],\"events\":[");
        ds4_prof_record_meta rm[DS4_PROF_REC_RING];
        const int rn = ds4_profiler_record_list(rm, DS4_PROF_REC_RING);
        const uint64_t newest_seq = rn > 0 ? rm[rn - 1].seq : UINT64_MAX;
        const int cap = 4096;
        ds4_prof_event *ev = malloc((size_t)cap * sizeof(*ev));
        const int en = ev ? ds4_profiler_record_events(newest_seq, ev, cap) : 0;
        for (int i = 0; i < en; i++) {
            if (i) jraw(&j, ",");
            jfmt(&j, "{\"us\":%u,\"ns\":%u,\"k\":\"%s\",\"layer\":%u,"
                     "\"expert\":%u,\"dev\":%u,\"bytes\":%u}",
                 ev[i].t_us, ev[i].dur_ns,
                 ds4_profiler_event_kind(ev[i].kind),
                 (unsigned)ev[i].layer, (unsigned)ev[i].expert,
                 (unsigned)ev[i].device, ev[i].bytes);
        }
        free(ev);
        (void)newest;
    }
    jraw(&j, "],");

    /* Per-device rows. */
    jraw(&j, "\"devices\":[");
    for (int i = 0; i < s.n_devices; i++) {
        const ds4_profiler_device *d = &s.dev[i];
        const vram_parts p = vram_split(d, h.kv_bytes);
        if (i) jraw(&j, ",");
        jraw(&j, "{");
        jfmt(&j, "\"tier\":%d,", i);
        jfmt(&j, "\"device_id\":%d,", d->device_id);
        jbytes(&j, "vram_used_bytes", d->vram_used_bytes);   jraw(&j, ",");
        jbytes(&j, "vram_total_bytes", d->vram_total_bytes); jraw(&j, ",");
        jbytes(&j, "dense_bytes", p.dense);       jraw(&j, ",");
        jbytes(&j, "arena_bytes", p.arena);       jraw(&j, ",");
        jbytes(&j, "kv_bytes", p.kv);             jraw(&j, ",");
        jbytes(&j, "expert_bytes", p.experts);    jraw(&j, ",");
        jbytes(&j, "scratch_bytes", p.scratch);   jraw(&j, ",");
        jbytes(&j, "temp_bytes", p.temp);         jraw(&j, ",");
        jbytes(&j, "tensor_bytes", p.tensors);    jraw(&j, ",");
        jbytes(&j, "pool_bytes", p.pool);         jraw(&j, ",");
        jbytes(&j, "other_bytes", p.other);       jraw(&j, ",");
        jbytes(&j, "vram_sum_bytes", p.sum);      jraw(&j, ",");
        jbytes(&j, "expert_count", d->expert_count);       jraw(&j, ",");
        jbytes(&j, "expert_size_bytes", d->expert_size_bytes); jraw(&j, ",");
        jbytes(&j, "weight_bytes", d->weight_bytes);       jraw(&j, ",");
        jbytes(&j, "activation_bytes", d->activation_bytes); jraw(&j, ",");
        jbytes(&j, "pcie_tx_bytes", d->pcie_tx_bytes);     jraw(&j, ",");
        jbytes(&j, "pcie_rx_bytes", d->pcie_rx_bytes);     jraw(&j, ",");
        jbytes(&j, "pcie_peak_bytes_per_s", d->pcie_max_bytes_per_s); jraw(&j, ",");
        jraw(&j, "\"name\":"); jstr(&j, d->name);              jraw(&j, ",");
        jbytes(&j, "mem_bw_bytes_per_s", d->mem_bw_bytes_per_s); jraw(&j, ",");
        jbytes(&j, "compute_flops_fp16", d->compute_flops_fp16); jraw(&j, ",");
        jfmt(&j, "\"sm_count\":%u,", d->sm_count);
        jfmt(&j, "\"pcie_gen\":%d,", d->pcie_gen);
        jfmt(&j, "\"pcie_width\":%d,", d->pcie_width);
        jfmt(&j, "\"tensor_gen\":%d,", d->tensor_gen);
        jfmt(&j, "\"p2p\":%d,", d->p2p ? 1 : 0);
        jfmt(&j, "\"nvlink\":%d,", d->nvlink ? 1 : 0);
        jraw(&j, "\"weight_type\":"); jstr(&j, d->type);       jraw(&j, ",");
        jraw(&j, "\"expert_type\":"); jstr(&j, d->expert_type);
        jraw(&j, "}");
    }
    jraw(&j, "]");

    jraw(&j, "}");
    return j.n;
}

/* ---- detail dump ---------------------------------------------------------
 * The profiler's answer to "what happened inside this token". One record is
 * rendered start to end from the record itself plus its event and expert
 * rings: every measured stage with its launch/wait split and its share, every
 * device as it was before and after, every expert use with the tier its bytes
 * came from. The stage capture cannot bracket everything, so the discussed
 * sum is compared against the record's wall time and the difference is printed
 * as an explicit "unmeasured" line rather than being folded into a stage.
 *
 * Two renderings of the same data: text, which a person or a model reads
 * directly, and JSON, which the dashboard and any other process consumes.
 * Both write into a caller-provided buffer, never overflow it, and mark the
 * cut when the buffer was too small.
 */

#define DUMP_TRUNC "\n[truncated: buffer too small]\n"

static const char *dump_src_name(unsigned src) {
    switch (src) {
    case DS4_PTRACE_SRC_POOL:  return "gpu-pool";
    case DS4_PTRACE_SRC_ARENA: return "host-arena";
    case DS4_PTRACE_SRC_SSD:   return "ssd";
    case DS4_PTRACE_SRC_CPU:   return "cpu";
    default:                   return "unknown";
    }
}

void ds4_profiler_format_bytes(char *out, size_t cap, uint64_t bytes) {
    static const char *unit[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
    if (!out || cap == 0) return;
    double v = (double)bytes;
    int u = 0;
    while (v >= 1024.0 && u < 5) { v /= 1024.0; u++; }
    if (u == 0) {
        snprintf(out, cap, "%llu B", (unsigned long long)bytes);
        return;
    }
    char frac[16];
    snprintf(frac, sizeof(frac), "%.3f", v);
    /* Trim the padding zeros so 6.750 MiB reads as 6.75 MiB; three decimals
     * are the ceiling, not a fixed width. */
    char *dot = strchr(frac, '.');
    if (dot) {
        char *e = frac + strlen(frac) - 1;
        while (e > dot && *e == '0') *e-- = '\0';
        if (e == dot) *e = '\0';
    }
    snprintf(out, cap, "%s %s", frac, unit[u]);
}

static void dump_stage_meta(int id, char *key, size_t kcap,
                            char *label, size_t lcap,
                            char *cat, size_t ccap) {
    if (key && kcap) key[0] = '\0';
    if (label && lcap) label[0] = '\0';
    if (cat && ccap) cat[0] = '\0';
    LOCK();
    if (id >= 0 && id < g_stage_n) {
        if (key && kcap) snprintf(key, kcap, "%s", g_stages[id].key);
        if (label && lcap) snprintf(label, lcap, "%s", g_stages[id].label);
        if (cat && ccap) snprintf(cat, ccap, "%s", g_stages[id].category);
    }
    UNLOCK();
    if (label && lcap && !label[0]) snprintf(label, lcap, "stage %d", id);
    if (key && kcap && !key[0]) snprintf(key, kcap, "stage/%d", id);
}

/* Record text is sanitised when it is captured, but a dump may be piped into
 * a terminal or a prompt, so keep control bytes out of it anyway. */
static void dump_qtext(jbuf *o, const char *s) {
    jraw(o, "\"");
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\') { char b[3] = { '\\', (char)*p, 0 }; jraw(o, b); }
        else if (*p < 0x20 || *p == 0x7f) jraw(o, " ");
        else { char b[2] = { (char)*p, 0 }; jraw(o, b); }
    }
    jraw(o, "\"");
}

static int rec_stage_cmp(const void *a, const void *b) {
    const ds4_prof_stage_use *x = *(const ds4_prof_stage_use *const *)a;
    const ds4_prof_stage_use *y = *(const ds4_prof_stage_use *const *)b;
    if (x->total_ns != y->total_ns) return x->total_ns < y->total_ns ? 1 : -1;
    return x->id < y->id ? -1 : x->id > y->id;
}

/* Wall time minus the sum of the measured stages: the part of the token the
 * boundaries did not cover. Negative rounding is clamped to zero. */
static double rec_explained_ms(const ds4_prof_record *r, double *unexplained_ms) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < r->stage_n; i++) sum += r->stages[i].total_ns;
    const double ms = (double)sum / 1e6;
    if (unexplained_ms) {
        const double gap = r->ms - ms;
        *unexplained_ms = gap > 0.0 ? gap : 0.0;
    }
    return ms;
}

static void dump_expert_counts(const ds4_prof_expert_trace *x, int n,
                               uint64_t *pool, uint64_t *arena,
                               uint64_t *ssd, uint64_t *cpu) {
    uint64_t p = 0, a = 0, s = 0, c = 0;
    for (int i = 0; i < n; i++) {
        switch (x[i].source) {
        case DS4_PTRACE_SRC_POOL:  p++; break;
        case DS4_PTRACE_SRC_ARENA: a++; break;
        case DS4_PTRACE_SRC_SSD:   s++; break;
        default:                   c++; break;
        }
    }
    if (pool) *pool = p;
    if (arena) *arena = a;
    if (ssd) *ssd = s;
    if (cpu) *cpu = c;
}

size_t ds4_profiler_record_dump(uint64_t seq, char *buf, size_t buflen) {
    if (!buf || buflen == 0) return 0;
    buf[0] = '\0';
    ds4_prof_record r;
    if (!ds4_profiler_record_get(seq, &r)) {
        return (size_t)snprintf(buf, buflen, "no profiler record with seq %llu\n",
                                (unsigned long long)seq);
    }
    const int max_ev = 16384, max_x = 8192;
    ds4_prof_event *ev = malloc((size_t)max_ev * sizeof(*ev));
    ds4_prof_expert_trace *xs = malloc((size_t)max_x * sizeof(*xs));
    const int n_ev = ev ? ds4_profiler_record_events(seq, ev, max_ev) : 0;
    const int n_x = xs ? ds4_profiler_record_experts(seq, xs, max_x) : 0;

    jbuf o = { buf, buflen, 0 };
    char b1[32], b2[32], kb[48], lb[64], cb[24];
    double unexplained = 0.0;
    const double expl = rec_explained_ms(&r, &unexplained);
    const double units = r.n_tokens ? (double)r.n_tokens : 1.0;

    /* A chunk record has no token of its own and no piece text; naming one would
     * print the sentinel and an empty string. */
    if (r.kind == DS4_PREC_PREFILL_CHUNK) {
        jfmt(&o, "record %llu  %s  %.3f ms  %.3f ms/unit  %u units  pos %d\n",
             (unsigned long long)r.seq, ds4_profiler_record_kind_name(r.kind),
             r.ms, r.ms / units, r.n_tokens, r.pos);
    } else {
        jfmt(&o, "record %llu  %s  %.3f ms  %.3f ms/unit  %u unit%s  pos %d  token %u\n",
             (unsigned long long)r.seq, ds4_profiler_record_kind_name(r.kind),
             r.ms, r.ms / units, r.n_tokens, r.n_tokens == 1 ? "" : "s", r.pos,
             r.token);
        jraw(&o, "  text ");
        dump_qtext(&o, r.text);
        jraw(&o, "\n");
    }
    /* Stage totals are per-device wall clock, and decode splits work across the
     * GPUs, so overlapping stages can sum past the record: report the ratio
     * plainly instead of pretending the excess is unexplained time. */
    const double expl_pct = r.ms > 0.0 ? 100.0 * expl / r.ms : 0.0;
    jfmt(&o, "  stage time %.3f ms over a %.3f ms record (%.1f%%), %.3f ms unattributed\n",
         expl, r.ms, expl_pct, unexplained);
    if (expl_pct > 100.5 && r.stage_n > 0) {
        jraw(&o, "  (over 100% because stages on different devices overlap)\n");
    }
    {
        uint64_t p, a, s, c;
        dump_expert_counts(xs, n_x, &p, &a, &s, &c);
        /* The per-record buffers are capped, so say when the counts describe a
         * prefix: a prefill chunk routes far more experts than fit. */
        jfmt(&o, "  experts %d%s%u: gpu-pool %llu, host-arena %llu, ssd %llu, cpu %llu\n",
             n_x, n_x < (int)r.expert_n ? " of " : "", r.expert_n,
             (unsigned long long)p, (unsigned long long)a,
             (unsigned long long)s, (unsigned long long)c);
    }
    ds4_profiler_format_bytes(b1, sizeof(b1), r.ssd_read_bytes);
    ds4_profiler_format_bytes(b2, sizeof(b2), r.host_ram_used);
    jfmt(&o, "  ssd read %s  host ram %s  events %u%s%u\n", b1, b2, n_ev,
         n_ev < (int)r.event_n ? " of " : "", r.event_n);
    jfmt(&o, "  cpu %.1f%% of a core over the record (process), %.1f%% machine busy\n",
         r.end_ns > r.start_ns
             ? 100.0 * (double)r.proc_cpu_ns / (double)(r.end_ns - r.start_ns) : 0.0,
         r.sys_cpu_total_ns
             ? 100.0 * (double)r.sys_cpu_busy_ns / (double)r.sys_cpu_total_ns : 0.0);
    for (uint32_t i = 0; i < r.dev_n; i++) {
        const ds4_prof_rec_dev *d = &r.dev[i];
        char m0[32], m1[32];
        ds4_profiler_format_bytes(m0, sizeof(m0), d->mem_start);
        ds4_profiler_format_bytes(m1, sizeof(m1), d->mem_end);
        jfmt(&o, "  device %d  util %u%% -> %u%%  vram %s -> %s\n",
             d->device, d->util_start, d->util_end, m0, m1);
    }

    /* Stages, largest first: the order the eye wants, not the order they ran. */
    const ds4_prof_stage_use *sorted[DS4_PROF_REC_MAX_STAGES];
    const uint32_t ns = r.stage_n < DS4_PROF_REC_MAX_STAGES
        ? r.stage_n : DS4_PROF_REC_MAX_STAGES;
    for (uint32_t i = 0; i < ns; i++) sorted[i] = &r.stages[i];
    qsort(sorted, ns, sizeof(sorted[0]), rec_stage_cmp);
    jraw(&o, "  stages (largest first):\n");
    jraw(&o, "      total_ms    launch_ms     wait_ms   calls        bytes  share  category     key\n");
    const uint64_t sum_ns = (uint64_t)(expl * 1e6);
    for (uint32_t i = 0; i < ns; i++) {
        const ds4_prof_stage_use *u = sorted[i];
        dump_stage_meta(u->id, kb, sizeof(kb), lb, sizeof(lb), cb, sizeof(cb));
        ds4_profiler_format_bytes(b1, sizeof(b1), u->bytes);
        jfmt(&o, "  %11.4f  %11.4f  %11.4f  %6llu  %11s  %5.1f%%  %-11s  %s\n",
             (double)u->total_ns / 1e6, (double)u->encode_ns / 1e6,
             (double)u->wait_ns / 1e6, (unsigned long long)u->count, b1,
             sum_ns ? 100.0 * (double)u->total_ns / (double)sum_ns : 0.0, cb, kb);
    }

    /* Expert uses: grouped per layer so the routing of the record is readable
     * at a glance, and in selection order inside a layer. */
    if (n_x > 0) {
        /* One decode token routes ~6 experts per layer, a prefill chunk routes
         * hundreds; listing all of the latter would bury the routing shape, so
         * a run is shown up to the cap and its remainder is counted. */
        const int per_layer_cap = 12;
        jraw(&o, "  expert uses (layer: expert/source/src_gpu->dest_gpu):\n");
        int line = 0;
        for (int i = 0; i < n_x; ) {
            const uint16_t layer = xs[i].layer;
            int end = i;
            while (end < n_x && xs[end].layer == layer) end++;
            const int run = end - i;
            const int shown = run > per_layer_cap ? per_layer_cap : run;
            jfmt(&o, "%s    L%-3u:", line ? "\n" : "", layer);
            line++;
            for (int k = 0; k < shown; k++) {
                const ds4_prof_expert_trace *x = &xs[i + k];
                char sg[8], dg[8];
                if (x->src_gpu == 0xff) snprintf(sg, sizeof(sg), "host");
                else snprintf(sg, sizeof(sg), "g%u", x->src_gpu);
                if (x->dest_gpu == 0xff) snprintf(dg, sizeof(dg), "-");
                else snprintf(dg, sizeof(dg), "g%u", x->dest_gpu);
                jfmt(&o, " %u/%s/%s>%s", x->expert, dump_src_name(x->source), sg, dg);
            }
            if (run > shown) jfmt(&o, " +%d more", run - shown);
            i = end;
        }
        jraw(&o, "\n");
    }

    /* Events, in order: what happened between the boundaries. */
    if (n_ev > 0) {
        jraw(&o, "  events (t_us, kind, layer, expert, device, bytes, dur_us):\n");
        for (int i = 0; i < n_ev; i++) {
            const ds4_prof_event *e = &ev[i];
            jfmt(&o, "  %9u  %-14s L%-3u e%-4u d%-2u %10u %8.1f\n",
                 e->t_us, ds4_profiler_event_kind(e->kind), e->layer, e->expert,
                 e->device, e->bytes, (double)e->dur_ns / 1000.0);
        }
    }
    free(ev);
    free(xs);
    const int cut = o.n + sizeof(DUMP_TRUNC) > o.cap;
    if (cut && o.cap > sizeof(DUMP_TRUNC)) {
        const size_t at = o.cap - sizeof(DUMP_TRUNC);
        memcpy(buf + at, DUMP_TRUNC, sizeof(DUMP_TRUNC) - 1u);
        o.n = o.cap - 1u;
    }
    return jdone(&o);
}

size_t ds4_profiler_record_dump_json(uint64_t seq, char *buf, size_t buflen) {
    if (!buf || buflen == 0) return 0;
    buf[0] = '\0';
    ds4_prof_record r;
    jbuf j = { buf, buflen, 0 };
    if (!ds4_profiler_record_get(seq, &r)) {
        jfmt(&j, "{\"error\":\"no profiler record with seq %llu\"}",
             (unsigned long long)seq);
        return j.n;
    }
    const int max_ev = 16384, max_x = 8192;
    ds4_prof_event *ev = malloc((size_t)max_ev * sizeof(*ev));
    ds4_prof_expert_trace *xs = malloc((size_t)max_x * sizeof(*xs));
    const int n_ev = ev ? ds4_profiler_record_events(seq, ev, max_ev) : 0;
    const int n_x = xs ? ds4_profiler_record_experts(seq, xs, max_x) : 0;
    double unexplained = 0.0;
    const double expl = rec_explained_ms(&r, &unexplained);

    jfmt(&j, "{\"seq\":%llu,", (unsigned long long)r.seq);
    jraw(&j, "\"kind\":"); jstr(&j, ds4_profiler_record_kind_name(r.kind));
    jfmt(&j, ",\"kind_id\":%u", r.kind);
    jfmt(&j, ",\"closed\":%u", r.closed ? 1u : 0u);
    jfmt(&j, ",\"pos\":%d", r.pos);
    jfmt(&j, ",\"n_tokens\":%u", r.n_tokens);
    jfmt(&j, ",\"token\":%u", r.token);
    jraw(&j, ",\"text\":"); jstr(&j, r.text);
    jfmt(&j, ",\"ms\":%.4f", r.ms);
    jfmt(&j, ",\"explained_ms\":%.4f", expl);
    jfmt(&j, ",\"unexplained_ms\":%.4f", unexplained);
    jfmt(&j, ",\"ssd_read_bytes\":%llu", (unsigned long long)r.ssd_read_bytes);
    jfmt(&j, ",\"host_ram_used\":%llu", (unsigned long long)r.host_ram_used);
    jfmt(&j, ",\"host_ram_avail\":%llu", (unsigned long long)r.host_ram_avail);
    jfmt(&j, ",\"proc_cpu_ns\":%llu", (unsigned long long)r.proc_cpu_ns);
    jfmt(&j, ",\"sys_cpu_busy_ns\":%llu", (unsigned long long)r.sys_cpu_busy_ns);
    jfmt(&j, ",\"sys_cpu_total_ns\":%llu", (unsigned long long)r.sys_cpu_total_ns);

    jraw(&j, ",\"devices\":[");
    for (uint32_t i = 0; i < r.dev_n; i++) {
        const ds4_prof_rec_dev *d = &r.dev[i];
        if (i) jraw(&j, ",");
        jfmt(&j, "{\"device\":%d,\"util_start\":%u,\"util_end\":%u,"
                 "\"mem_start\":%llu,\"mem_end\":%llu}",
             d->device, d->util_start, d->util_end,
             (unsigned long long)d->mem_start, (unsigned long long)d->mem_end);
    }
    jraw(&j, "],");

    const ds4_prof_stage_use *sorted[DS4_PROF_REC_MAX_STAGES];
    const uint32_t ns = r.stage_n < DS4_PROF_REC_MAX_STAGES
        ? r.stage_n : DS4_PROF_REC_MAX_STAGES;
    for (uint32_t i = 0; i < ns; i++) sorted[i] = &r.stages[i];
    qsort(sorted, ns, sizeof(sorted[0]), rec_stage_cmp);
    const uint64_t sum_ns = (uint64_t)(expl * 1e6);
    jraw(&j, "\"stages\":[");
    for (uint32_t i = 0; i < ns; i++) {
        const ds4_prof_stage_use *u = sorted[i];
        char kb[48], lb[64], cb[24];
        dump_stage_meta(u->id, kb, sizeof(kb), lb, sizeof(lb), cb, sizeof(cb));
        if (i) jraw(&j, ",");
        jraw(&j, "{\"key\":");   jstr(&j, kb);
        jraw(&j, ",\"label\":"); jstr(&j, lb);
        jraw(&j, ",\"category\":"); jstr(&j, cb);
        jfmt(&j, ",\"total_ms\":%.4f", (double)u->total_ns / 1e6);
        jfmt(&j, ",\"launch_ms\":%.4f", (double)u->encode_ns / 1e6);
        jfmt(&j, ",\"wait_ms\":%.4f", (double)u->wait_ns / 1e6);
        jfmt(&j, ",\"calls\":%llu", (unsigned long long)u->count);
        jfmt(&j, ",\"bytes\":%llu", (unsigned long long)u->bytes);
        jfmt(&j, ",\"share_pct\":%.2f",
             sum_ns ? 100.0 * (double)u->total_ns / (double)sum_ns : 0.0);
        jraw(&j, "}");
    }
    jraw(&j, "],");

    uint64_t p, a, s, c;
    dump_expert_counts(xs, n_x, &p, &a, &s, &c);
    jfmt(&j, "\"experts\":{\"count\":%d,\"total\":%u,\"gpu_pool\":%llu,"
             "\"host_arena\":%llu,\"ssd\":%llu,\"cpu\":%llu,\"uses\":[",
         n_x, r.expert_n, (unsigned long long)p, (unsigned long long)a,
         (unsigned long long)s, (unsigned long long)c);
    for (int i = 0; i < n_x; i++) {
        const ds4_prof_expert_trace *x = &xs[i];
        if (i) jraw(&j, ",");
        jraw(&j, "{\"layer\":");
        jfmt(&j, "%u", x->layer);
        jraw(&j, ",\"expert\":");
        jfmt(&j, "%u", x->expert);
        jraw(&j, ",\"source\":");
        jstr(&j, dump_src_name(x->source));
        jfmt(&j, ",\"src_gpu\":%d,\"dest_gpu\":%d,\"bytes\":%u}",
             x->src_gpu == 0xff ? -1 : (int)x->src_gpu,
             x->dest_gpu == 0xff ? -1 : (int)x->dest_gpu, x->bytes);
    }
    jraw(&j, "]},");

    jfmt(&j, "\"events\":{\"count\":%d,\"total\":%u,\"items\":[", n_ev, r.event_n);
    for (int i = 0; i < n_ev; i++) {
        const ds4_prof_event *e = &ev[i];
        if (i) jraw(&j, ",");
        jraw(&j, "{\"kind\":");
        jstr(&j, ds4_profiler_event_kind(e->kind));
        jfmt(&j, ",\"t_us\":%u,\"dur_ns\":%u,\"stage\":%u,\"layer\":%u,"
                 "\"expert\":%u,\"device\":%u,\"bytes\":%u}",
             e->t_us, e->dur_ns, e->stage, e->layer, e->expert, e->device,
             e->bytes);
    }
    jraw(&j, "]}");
    jraw(&j, "}");
    free(ev);
    free(xs);
    return jdone(&j);
}

/* Newest-first list of live records, optionally filtered by kind. `kind` is a
 * DS4_PREC_* value, or -1 for every kind. */
size_t ds4_profiler_records_dump(char *buf, size_t buflen, int kind, int limit) {
    if (!buf || buflen == 0) return 0;
    buf[0] = '\0';
    ds4_prof_record_meta metas[DS4_PROF_REC_RING];
    const int n = ds4_profiler_record_list(metas, DS4_PROF_REC_RING);
    jbuf o = { buf, buflen, 0 };
    int shown = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (kind >= 0 && (int)metas[i].kind != kind) continue;
        if (limit > 0 && shown >= limit) break;
        if (shown == 0) {
            jraw(&o, "   seq  kind              pos   units       ms  hot stage\n");
        }
        char lb[64], kb[48], cb[24];
        dump_stage_meta(metas[i].hot_stage, kb, sizeof(kb), lb, sizeof(lb),
                        cb, sizeof(cb));
        jfmt(&o, "  %4llu  %-14s %6d  %5u  %8.3f  %s\n",
             (unsigned long long)metas[i].seq,
             ds4_profiler_record_kind_name(metas[i].kind), metas[i].pos,
             metas[i].n_tokens, metas[i].ms, lb);
        shown++;
    }
    if (shown == 0) jraw(&o, "no profiler records match\n");
    else jfmt(&o, "%d record%s shown\n", shown, shown == 1 ? "" : "s");
    return jdone(&o);
}

size_t ds4_profiler_records_dump_json(char *buf, size_t buflen, int kind,
                                      int limit) {
    if (!buf || buflen == 0) return 0;
    buf[0] = '\0';
    ds4_prof_record_meta metas[DS4_PROF_REC_RING];
    const int n = ds4_profiler_record_list(metas, DS4_PROF_REC_RING);
    jbuf j = { buf, buflen, 0 };
    jraw(&j, "{\"kind\":");
    jstr(&j, kind < 0 ? "all" : ds4_profiler_record_kind_name(kind));
    jraw(&j, ",\"records\":[");
    int shown = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (kind >= 0 && (int)metas[i].kind != kind) continue;
        if (limit > 0 && shown >= limit) break;
        if (shown) jraw(&j, ",");
        char kb[48], lb[64], cb[24];
        dump_stage_meta(metas[i].hot_stage, kb, sizeof(kb), lb, sizeof(lb),
                        cb, sizeof(cb));
        jfmt(&j, "{\"seq\":%llu,\"kind\":",
             (unsigned long long)metas[i].seq);
        jstr(&j, ds4_profiler_record_kind_name(metas[i].kind));
        jfmt(&j, ",\"pos\":%d,\"n_tokens\":%u,\"token\":%u,\"ms\":%.4f",
             metas[i].pos, metas[i].n_tokens, metas[i].token, metas[i].ms);
        jraw(&j, ",\"text\":"); jstr(&j, metas[i].text);
        jraw(&j, ",\"hot_stage\":");
        jstr(&j, lb);
        jfmt(&j, ",\"hot_ms\":%.4f", metas[i].hot_ms);
        jfmt(&j, ",\"events\":%u,\"experts\":%u", metas[i].events,
             metas[i].experts);
        jfmt(&j, ",\"experts_pool\":%u,\"experts_arena\":%u}",
             metas[i].experts_pool, metas[i].experts_arena);
        shown++;
    }
    jraw(&j, "]}");
    return jdone(&j);
}

/* ---- full report ---------------------------------------------------------
 * Everything the profiler knows, in the order it is read: how fast, where the
 * time went (split by record kind, never merged), what the expert tiers hold,
 * then the newest records and their full detail. This is the artifact a run
 * leaves behind and the body of the HTTP text endpoint, so a caller never has
 * to assemble the picture from several probes.
 */
static void report_stage_table(jbuf *o, const ds4_prof_view *v, int kind,
                               double units, const char *unit_name) {
    double total = 0.0;
    for (int i = 0; i < v->n_stages; i++) total += v->stages[i].kind[kind].total_ms;
    jfmt(o, "\nstages on %s (ms/%s, share of the %s total of %.2f ms):\n",
         ds4_profiler_record_kind_name(kind), unit_name, unit_name, total);
    jraw(o, "   ms/unit   launch     wait   shares  calls/unit        bytes  category     key\n");
    /* Selection sort by kind total, descending: the list is bounded (160) and
     * rebuilding it in place would need a second copy of every stage row. */
    char done[DS4_PROF_MAX_STAGES];
    memset(done, 0, sizeof(done));
    for (int n = 0; n < v->n_stages; n++) {
        int best = -1;
        for (int i = 0; i < v->n_stages; i++) {
            if (done[i] || v->stages[i].kind[kind].calls == 0) continue;
            if (best < 0 ||
                v->stages[i].kind[kind].total_ms > v->stages[best].kind[kind].total_ms) {
                best = i;
            }
        }
        if (best < 0) break;
        done[best] = 1;
        const ds4_prof_stage_view *s = &v->stages[best];
        const ds4_prof_stage_kind *k = &s->kind[kind];
        char b1[32];
        ds4_profiler_format_bytes(b1, sizeof(b1), k->bytes);
        const double u = units > 0.0 ? units : 1.0;
        jfmt(o, "  %8.3f  %7.3f  %7.3f  %5.1f%%  %10.2f  %11s  %-11s  %s\n",
             k->total_ms / u, k->encode_ms / u, k->wait_ms / u,
             total > 0.0 ? 100.0 * k->total_ms / total : 0.0,
             (double)k->calls / u, b1, s->category, s->key);
    }
}

size_t ds4_profiler_report(char *buf, size_t buflen, int kind, int limit) {
    if (!buf || buflen == 0) return 0;
    buf[0] = '\0';
    ds4_prof_view v;
    memset(&v, 0, sizeof(v));
    (void)ds4_profiler_view(&v);
    const ds4_profiler_timing *t = &v.timing;
    jbuf o = { buf, buflen, 0 };

    jraw(&o, "ds4 profiler report\n");
    const double dtok = (double)t->tokens;
    const double ptok = (double)t->prefill_tokens;
    jfmt(&o, "  decode   %.2f t/s  %.2f ms/token  (%llu tokens)\n",
         ds4_prof_avg(&t->tokens_per_s),
         dtok > 0.0 ? t->per_token_ms.sum / dtok : 0.0,
         (unsigned long long)t->tokens);
    jfmt(&o, "  prefill  %.2f t/s  %.2f ms/token  (%llu tokens in %llu chunks)\n",
         ds4_prof_avg(&t->prefill_tokens_per_s),
         ptok > 0.0 ? t->prefill_ms.sum / ptok : 0.0,
         (unsigned long long)t->prefill_tokens,
         (unsigned long long)t->prefill_ms.n);
    /* The token total is split into CPU-driven and device-blocked parts; the
     * remainder is the gap the measured steps do not cover. */
    const double pt = dtok > 0.0 ? t->per_token_ms.sum / dtok : 0.0;
    const double hm = dtok > 0.0 ? t->host_ms.sum / dtok : 0.0;
    const double dw = dtok > 0.0 ? t->device_wait_ms.sum / dtok : 0.0;
    jfmt(&o, "  token split: host %.2f ms (%.1f%%), device wait %.2f ms (%.1f%%), "
             "other %.2f ms\n",
         hm, pt > 0.0 ? 100.0 * hm / pt : 0.0,
         dw, pt > 0.0 ? 100.0 * dw / pt : 0.0,
         pt - hm - dw > 0.0 ? pt - hm - dw : 0.0);
    for (int d = 0; d < v.dev_n; d++) {
        const ds4_prof_dev_state *s = &v.dev[d];
        char m0[32], m1[32];
        ds4_profiler_format_bytes(m0, sizeof(m0), s->mem_used);
        ds4_profiler_format_bytes(m1, sizeof(m1), s->mem_total);
        jfmt(&o, "  device %d  %-24s util %u%%  vram %s of %s\n",
             s->device, s->name, s->util_pct, m0, m1);
    }
    if (v.stats.host_arena_capacity_bytes) {
        char c0[32], c1[32], c2[32];
        ds4_profiler_format_bytes(c0, sizeof(c0), v.stats.host_arena_used_bytes);
        ds4_profiler_format_bytes(c1, sizeof(c1), v.stats.host_arena_capacity_bytes);
        ds4_profiler_format_bytes(c2, sizeof(c2), v.stats.host_expert_bytes);
        const uint64_t look = v.stats.host_arena_hits + v.stats.host_arena_misses;
        jfmt(&o, "  expert arena: %s of %s (%.1f%%), %llu experts of %llu "
                 "(%s each, %.1f%% of the model's routed experts)\n",
             c0, c1,
             v.stats.host_arena_capacity_bytes
                 ? 100.0 * (double)v.stats.host_arena_used_bytes /
                       (double)v.stats.host_arena_capacity_bytes : 0.0,
             (unsigned long long)v.stats.host_arena_experts,
             (unsigned long long)v.host.total_experts, c2,
             v.host.total_experts
                 ? 100.0 * (double)v.stats.host_arena_experts /
                       (double)v.host.total_experts : 0.0);
        jfmt(&o, "  arena lookups: %llu hits (%.1f%%), %llu misses; "
                 "%llu admissions, %llu evictions, %llu fetches of evicted experts\n",
             (unsigned long long)v.stats.host_arena_hits,
             look ? 100.0 * (double)v.stats.host_arena_hits / (double)look : 0.0,
             (unsigned long long)v.stats.host_arena_misses,
             (unsigned long long)v.stats.host_arena_admissions,
             (unsigned long long)v.stats.host_arena_evictions,
             (unsigned long long)v.stats.host_arena_evicted_fetches);
        jfmt(&o, "  arena headroom left: %llu experts (%s)\n",
             (unsigned long long)(v.stats.host_arena_capacity_bytes >
                                          v.stats.host_arena_used_bytes
                 ? (v.stats.host_arena_capacity_bytes - v.stats.host_arena_used_bytes) /
                       (v.stats.host_expert_bytes ? v.stats.host_expert_bytes : 1)
                 : 0),
             c1);
    } else {
        jraw(&o, "  expert arena: not configured\n");
    }

    report_stage_table(&o, &v, DS4_PREC_DECODE_TOKEN, dtok, "token");
    report_stage_table(&o, &v, DS4_PREC_PREFILL_CHUNK, (double)t->prefill_ms.n,
                       "chunk");

    jraw(&o, "\nrecords:\n");
    {
        char *list = malloc(64u * 1024u);
        if (list) {
            const size_t n = ds4_profiler_records_dump(list, 64u * 1024u, kind,
                                                      limit);
            if (o.n + n < o.cap) { memcpy(buf + o.n, list, n); o.n += n; }
            free(list);
        }
    }
    /* Full detail of each listed record, newest first, as long as it fits. */
    ds4_prof_record_meta metas[DS4_PROF_REC_RING];
    const int n = ds4_profiler_record_list(metas, DS4_PROF_REC_RING);
    char *sub = malloc(DS4_PROF_DUMP_CAP);
    int shown = 0;
    for (int i = n - 1; sub && i >= 0; i--) {
        if (kind >= 0 && (int)metas[i].kind != kind) continue;
        if (limit > 0 && shown >= limit) break;
        const size_t sn = ds4_profiler_record_dump(metas[i].seq, sub,
                                                   DS4_PROF_DUMP_CAP);
        if (o.n + sn + 2u >= o.cap) {
            jraw(&o, "\n[report truncated: raise the buffer to see more records]\n");
            break;
        }
        jraw(&o, "\n");
        memcpy(buf + o.n, sub, sn);
        o.n += sn;
        jraw(&o, "\n");
        shown++;
    }
    free(sub);
    return jdone(&o);
}
