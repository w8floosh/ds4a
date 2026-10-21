/* ds4_profiler.h — engine telemetry: per-token/chunk records, stage costs,
 * transfer/wait counters, expert residency.
 *
 * The consumer is ds4_tui.c, which renders the profiler pane directly from
 * ds4_profiler_view()/record_get(); the report is also what --profile-dump
 * writes at the end of a run. This module owns only data and aggregation; it
 * never writes to the terminal.
 *
 * Producers are split by knowledge:
 *   - the GPU backend fills ds4_profiler_stats (per-device memory, PCIe links);
 *   - the engine fills ds4_profiler_host_stats (model size, host expert cache);
 *   - the decode loop feeds per-token timing through ds4_profiler_note_token().
 *
 * The "no acronyms" rule applies to the human labels, not to these field names.
 */
#ifndef DS4_PROFILER_H
#define DS4_PROFILER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_PROFILER_MAX_DEV  16
#define DS4_PROFILER_TYPE_LEN 24

/* Per-device snapshot. Byte fields are raw; the profiler pane turns them into
 * the human-facing composition. */
typedef struct {
    int      device_id;
    uint64_t vram_used_bytes;
    uint64_t vram_total_bytes;
    uint64_t pcie_tx_bytes;          /* cumulative bytes into the device */
    uint64_t pcie_rx_bytes;          /* cumulative bytes out of the device */
    uint64_t pcie_max_bytes_per_s;   /* measured one-way peak, bar denominator */
    uint64_t dense_bytes;            /* model range cache + q8 derived caches */
    uint64_t arena_bytes;            /* resident non-routed weight arena */
    uint64_t derived_bytes;          /* derived weight artifacts */
    uint64_t temp_bytes;             /* reusable temp + token-tile scratch */
    uint64_t tensor_bytes;           /* all graph tensor allocations */
    uint64_t mempool_bytes;          /* cudaMallocAsync pool reserved */
    uint64_t scratch_bytes;          /* per-device graph scratch */
    uint64_t expert_bytes;           /* resident routed-expert bytes */
    uint64_t expert_size_bytes;      /* bytes per resident expert */
    uint32_t expert_count;           /* resident expert slots */
    uint64_t weight_bytes;           /* cumulative weight/expert bytes loaded in */
    uint64_t activation_bytes;       /* cumulative activation traffic, both ways */
    char     type[DS4_PROFILER_TYPE_LEN];        /* dense weight type label */
    char     expert_type[DS4_PROFILER_TYPE_LEN]; /* routed-expert quant label */
    /* Static specifications from the startup probe, for the device panel. */
    char     name[64];               /* marketing name */
    uint64_t mem_bw_bytes_per_s;     /* VRAM bandwidth */
    uint64_t compute_flops_fp16;     /* fp16 tensor throughput, FLOP/s */
    uint32_t sm_count;
    int      pcie_gen;               /* negotiated link generation */
    int      pcie_width;             /* negotiated link width, lanes */
    int      tensor_gen;             /* compute capability, major*10+minor */
    int      p2p;                    /* peer access validated for some pair */
    int      nvlink;                 /* NVLink rather than PCIe */
} ds4_profiler_device;

typedef struct {
    int n_devices;
    uint64_t ssd_read_bytes;        /* cumulative bytes read from the SSD */
    uint64_t host_experts_resident; /* experts actually loaded into host RAM */
    uint64_t host_expert_bytes;     /* per-expert size, for the RAM row */
    /* Pinned host expert arena: a resident set the kernel cannot evict. When
     * it exists it is the authority on what is held in host RAM; hits/misses
     * split the experts that reach the fetch seam by where their bytes came
     * from, so a shortfall shows up as misses instead of a silent slowdown. */
    uint64_t host_arena_capacity_bytes;
    uint64_t host_arena_used_bytes;
    uint64_t host_arena_experts;
    uint64_t host_arena_hits;
    uint64_t host_arena_misses;
    uint64_t host_arena_admissions;
    uint32_t host_arena_admission;   /* misses may still claim free slots */
    uint64_t host_arena_evictions;
    uint64_t host_arena_evicted_fetches;
    ds4_profiler_device dev[DS4_PROFILER_MAX_DEV];
} ds4_profiler_stats;

/* Resident non-routed weight categories (embedding, router, ...). */
#define DS4_PROFILER_DENSE_CATS 10
typedef struct {
    char     name[12];
    uint64_t bytes;
} ds4_profiler_cat;

/*
 * Non-GPU tiers. The engine owns these because only it knows the model file
 * size and the host expert-cache budget; the JSON serializer combines them with
 * the device rows into a full SSD / RAM / GPU expert distribution.
 */
typedef struct {
    int      has_ssd;                 /* model is streamed from disk */
    int      has_ram;                 /* a host expert cache is configured */
    uint64_t total_experts;           /* routed experts in the model */
    uint64_t expert_bytes;            /* bytes per routed expert */
    uint64_t ram_experts;             /* experts held in host RAM */
    uint64_t ram_bytes;               /* RAM occupied by those experts */
    uint64_t host_ram_total_bytes;    /* physical RAM for the RAM bar */
    uint64_t model_bytes;             /* GGUF size on the SSD */
    uint64_t kv_bytes;                /* KV cache allocated on the compute GPU */
    int      dense_n;                 /* dense breakdown entries */
    ds4_profiler_cat dense[DS4_PROFILER_DENSE_CATS];
} ds4_profiler_host_stats;

void ds4_profiler_set_host_stats(const ds4_profiler_host_stats *s);

/* Backend: fill a snapshot. Zero devices when no GPU backend is active. */
void ds4_gpu_profiler_stats(ds4_profiler_stats *out);
void ds4_gpu_profiler_init(void);

/* Label the routed-expert quantization (e.g. "IQ2_XXS") for the expert column.
 * No-op on backends without a device profiler. */
void ds4_gpu_set_streaming_expert_type_name(const char *name);

/* Push engine-owned per-device residency into the snapshot: the selective-cache
 * slab holding the dense/non-routed set on the compute tier, and the startup
 * hot-expert bank on a cache tier. The engine owns the model map and the
 * placement policy; the backend owns the snapshot, so residency is handed over
 * at install time. `expert_entries` counts resident (layer, expert) slots and
 * `expert_bytes` the slab behind them. No-op without a device profiler. */
void ds4_gpu_profiler_set_device_resident(int device_id,
                                          uint64_t dense_bytes,
                                          uint32_t expert_entries,
                                          uint64_t expert_bytes);

/* Cheap monotone transfer counters per logical tier, read on the token path
 * without touching device state. `weight_into` is weight/expert bytes loaded
 * into the tier; `activation_io` is activation traffic in both directions.
 * Returns the number of tiers written (<= max_tiers). */
typedef struct {
    uint64_t weight_into;
    uint64_t activation_io;
} ds4_gpu_transfer;
int ds4_gpu_transfer_snapshot(ds4_gpu_transfer *out, int max_tiers);

/* Measured one-way PCIe peak for a logical tier, bytes/s (0 if unknown). */
uint64_t ds4_gpu_pcie_peak_bps(int tier);

/* Cumulative per-phase wall time on the token path. Every field is elapsed
 * time measured on the calling thread, so the per-token deltas add up exactly:
 *   host_read + host_other + device_upload + device_readback + device_sync
 *   == per token. `host_other` and `device_sync` are the residuals of the two
 * totals, so the tree closes whatever the split leaves over. */
typedef struct {
    uint64_t device_wait_ns;   /* total blocked on the device (sum of 2+3+4) */
    uint64_t upload_wait_ns;   /* blocked on the expert upload stream        */
    uint64_t readback_ns;      /* blocked on device->host tensor reads       */
    uint64_t host_read_ns;     /* blocked in the host expert read pool        */
    uint64_t boundary_wait_ns; /* of device_wait: stage-profiler boundaries   */
} ds4_gpu_phases;

void ds4_gpu_phase_snapshot(ds4_gpu_phases *out);

/* Bracket a stage-profiler boundary: the profiler ends the command stream and
 * waits there, so the backend charges that wait apart from the engine's own. */
void ds4_gpu_stage_boundary_sync_begin(void);
void ds4_gpu_stage_boundary_sync_end(void);

/* Cheap machine counters for the per-token record. Unlike
 * ds4_gpu_profiler_stats(), this touches no device state (no cudaMemGetInfo,
 * no NVML): only counters the backend already updates. `tiers` is the number
 * of logical tiers written and keeps every array index valid. */
#define DS4_PROF_MACHINE_TIERS 16
typedef struct {
    int      tiers;
    uint64_t ssd_read_bytes;                    /* cumulative model-file reads */
    uint64_t tx_bytes[DS4_PROF_MACHINE_TIERS];   /* into the device          */
    uint64_t rx_bytes[DS4_PROF_MACHINE_TIERS];   /* out of the device        */
    uint64_t weight_bytes[DS4_PROF_MACHINE_TIERS];
    uint64_t activation_bytes[DS4_PROF_MACHINE_TIERS];
} ds4_gpu_machine;

void ds4_gpu_machine_snapshot(ds4_gpu_machine *out);

/* ---------------------------------------------------------------- timing -- */

/* A scalar with its running min/max/average. */
typedef struct {
    double   last;
    double   min;
    double   max;
    double   sum;
    uint64_t n;
} ds4_prof_stat;

static inline double ds4_prof_avg(const ds4_prof_stat *s) {
    return s->n ? s->sum / (double)s->n : 0.0;
}

/*
 * Per-token timing. The decode loop brackets one generated token with
 * begin/end. end() deltas three sources:
 *   - the backend transfer counters  -> transfer milliseconds at link rate;
 *   - the backend device-wait counter -> how long the CPU was blocked on the
 *     device (synchronize/readback), i.e. the device's own time.
 * The decomposition is
 *   per_token   = host + device_wait
 *   host        = CPU: graph encode, kernel launches, sampling, detokenize
 *   device_wait = time the CPU spent blocked on the device
 * The transfer terms are a separate reference: the time the moved bytes would
 * need at the measured link peak. Transfers run asynchronously and overlap both
 * host work and device compute, so they are not part of the two-way sum; compare
 * them against the effective rate to see how close to peak the link runs.
 */
void ds4_profiler_token_begin(void);
void ds4_profiler_token_end(uint64_t tokens);

/* ---- per-step timeline ---------------------------------------------------
 * One row per kind of cost, never two kinds merged.
 *
 *   HOST   CPU wall time spent doing work (not blocked on the device)
 *   DEV    GPU busy time on one kernel group
 *   XFER   bytes crossing a link (time is reported, bytes carry the rate)
 *   WAIT   CPU blocked waiting for the device (kept apart from HOST on purpose:
 *          waiting is not work, and merging them hides the bottleneck)
 *
 * Every row carries a human label and a one-line explanation, so a reader (or a
 * tooltip) never needs a second copy of this table.
 */
typedef enum {
    DS4_PSTEP_HOST_EMBED,
    DS4_PSTEP_HOST_ROUTER,
    DS4_PSTEP_HOST_ATTN,
    DS4_PSTEP_HOST_SHARED,
    DS4_PSTEP_HOST_MOE_PREP,
    DS4_PSTEP_HOST_TIER_DISPATCH,
    DS4_PSTEP_HOST_COMBINE,
    DS4_PSTEP_HOST_HEAD,
    DS4_PSTEP_HOST_SAMPLE,
    DS4_PSTEP_HOST_OTHER,
    DS4_PSTEP_HOST_DETOK,         /* token text + frontend handoff         */

    DS4_PSTEP_DEV_EMBED,
    DS4_PSTEP_DEV_ATTN_SCORE,
    DS4_PSTEP_DEV_ATTN_VALUE,
    DS4_PSTEP_DEV_ROUTER,
    DS4_PSTEP_DEV_SHARED,
    DS4_PSTEP_DEV_MOE_GATE_UP,
    DS4_PSTEP_DEV_MOE_DOWN,
    DS4_PSTEP_DEV_TIER,
    DS4_PSTEP_DEV_HEAD,
    DS4_PSTEP_DEV_IDLE,

    DS4_PSTEP_XFER_EXPERT_POOL,
    DS4_PSTEP_XFER_EXPERT_ARENA,
    DS4_PSTEP_XFER_EXPERT_SSD,
    DS4_PSTEP_XFER_PEER,
    DS4_PSTEP_XFER_LOGITS,
    DS4_PSTEP_XFER_PARTIALS,
    DS4_PSTEP_XFER_KV,
    DS4_PSTEP_XFER_ACT_HOST,      /* H2D activation / control tensors     */
    DS4_PSTEP_XFER_ACT_READBACK,  /* D2H activations, partials, ids       */

    DS4_PSTEP_WAIT_UPLOAD,
    DS4_PSTEP_WAIT_READBACK,
    DS4_PSTEP_WAIT_TIER,
    DS4_PSTEP_WAIT_SYNC,

    DS4_PSTEP_COUNT
} ds4_prof_step;

typedef struct {
    uint64_t ns;
    uint64_t count;
    uint64_t bytes;
} ds4_prof_step_stat;

/* Add one sample. Cheap enough for the token path: a few additions under the
 * lock, no allocation and no device queries. */
void ds4_profiler_note_step(ds4_prof_step step, uint64_t ns,
                            uint64_t count, uint64_t bytes);
void ds4_profiler_step_snapshot(ds4_prof_step_stat *out, int n);
const char *ds4_profiler_step_id(ds4_prof_step step);
const char *ds4_profiler_step_label(ds4_prof_step step);
const char *ds4_profiler_step_hint(ds4_prof_step step);
const char *ds4_profiler_step_category(ds4_prof_step step);

/* ---- per-token expert trace ---------------------------------------------
 * Which routed experts the token actually used, where they came from, and
 * where they were sent. A ring of the last few tokens is kept; older tokens
 * are overwritten so memory stays bounded on long runs.
 */
enum {
    DS4_PTRACE_EXPERT_RING = 131072,  /* ~350 tokens x 258 expert uses */
    DS4_PTRACE_TOKEN_RING  = 16
};

enum {
    DS4_PTRACE_SRC_POOL = 0,   /* already resident in a GPU's VRAM pool */
    DS4_PTRACE_SRC_ARENA = 1,  /* pinned host RAM (expert arena)        */
    DS4_PTRACE_SRC_SSD = 2,    /* read from the model file on disk      */
    DS4_PTRACE_SRC_CPU = 3     /* computed on the CPU, skipped on GPU   */
};

typedef struct {
    uint16_t layer;
    uint16_t expert;
    uint8_t  source;    /* DS4_PTRACE_SRC_* */
    uint8_t  src_gpu;   /* GPU tier it resided on, 0xFF when host/SSD */
    uint8_t  dest_gpu;  /* GPU tier that received it */
    uint8_t  _pad;
    uint32_t bytes;
    uint32_t token;     /* sequence number of the token */
    uint32_t _pad2;
    uint64_t rec_seq;   /* owning record, 0 when no record is open */
} ds4_prof_expert_trace;

/* Called once per generated token before its experts are traced. */
void ds4_profiler_trace_token(uint32_t token_seq);
void ds4_profiler_trace_expert(uint16_t layer, uint16_t expert,
                               uint8_t source, uint8_t src_gpu,
                               uint8_t dest_gpu, uint32_t bytes);
/* Copy the ring in chronological order; returns the number of valid entries. */
int  ds4_profiler_trace_snapshot(ds4_prof_expert_trace *out, int max);
/* Sequence number of the token currently being traced. */
uint32_t ds4_profiler_trace_current_token(void);

/* ---- ordered per-token event log ----------------------------------------
 * Events are timestamped relative to the owning record's start, so one token
 * or prefill chunk has a complete ordered timeline. `rec_seq` says which
 * record owns the event; stage events carry the interned stage id, duration,
 * and the CPU-encode / device-wait split.
 */
enum {
    DS4_PEV_XFER_EXPERT_START = 0,
    DS4_PEV_XFER_EXPERT_END,
    DS4_PEV_CACHE_IN,
    DS4_PEV_COMPUTE_START,
    DS4_PEV_COMPUTE_END,
    DS4_PEV_XFER_ACT_START,
    DS4_PEV_XFER_ACT_END,
    DS4_PEV_SAMPLE,
    DS4_PEV_ROUTE,
    DS4_PEV_STAGE,        /* one named scheduler stage finished */
    DS4_PEV_DETOK,        /* token text produced / pushed to the UI */
    DS4_PEV_STALL,        /* long host stall with no measured step */
    DS4_PEV_COUNT
};

enum {
    /* Sized to cover the record ring, not a guess: a decode token records about
     * 3k events (one per stage boundary plus experts and transfers), and the
     * record ring keeps 256 tokens, so a smaller ring emptied the ordered event
     * list of every record older than the ring while the record itself stayed. */
    DS4_PEV_RING = 1u << 20,
    DS4_PEV_DEV_MAX = 16
};

typedef struct {
    uint64_t seq;       /* global monotonic event sequence */
    uint64_t rec_seq;   /* owning record sequence, 0 if none */
    uint32_t t_us;      /* microseconds since the record started */
    uint32_t dur_ns;    /* event duration; 0 when instantaneous */
    uint16_t kind;      /* DS4_PEV_* */
    uint16_t stage;     /* interned stage id for DS4_PEV_STAGE */
    uint16_t layer;
    uint16_t expert;
    uint8_t  device;
    uint8_t  _pad;
    uint32_t bytes;
    uint32_t token;
} ds4_prof_event;

/* Device state sampled at record start and end for EVERY device in the
 * machine, not only the ones used by inference. */
typedef struct {
    int      device;
    char     name[48];
    uint32_t util_pct;
    uint64_t mem_used;
    uint64_t mem_total;
} ds4_prof_dev_state;

void ds4_profiler_trace_begin(void);
void ds4_profiler_trace_end(void);
void ds4_profiler_trace_event(uint16_t kind, uint16_t layer, uint16_t expert,
                              uint8_t device, uint32_t bytes);
const char *ds4_profiler_event_kind(uint16_t kind);
int  ds4_profiler_event_snapshot(ds4_prof_event *out, int max);
/* True when DS4_PROFILE_TRACE is set or the TUI asked for records. */
int  ds4_profiler_trace_enabled(void);
int  ds4_profiler_dev_state_snapshot(ds4_prof_dev_state *out, int max);

/* ---- per token / per prefill chunk records ------------------------------
 * One record per decoded token or prefilled chunk. Records are bounded and
 * overwrite oldest-first, like the event rings. The TUI's token list and
 * detail page read these; the JSONL writer serialises them verbatim.
 */
typedef enum {
    DS4_PREC_DECODE_TOKEN = 0,
    DS4_PREC_PREFILL_CHUNK = 1,
    DS4_PREC_PREFILL_TOKEN = 2
} ds4_prof_rec_kind;

#define DS4_PREC_KIND_MAX       3

#define DS4_PROF_REC_RING       256
#define DS4_PROF_REC_MAX_STAGES 128
#define DS4_PROF_REC_MAX_DEV    16
#define DS4_PROF_MAX_STAGES     160

typedef struct {
    uint16_t id;        /* interned stage id */
    uint16_t _pad;
    uint64_t total_ns;  /* encode + wait for this stage in this record */
    uint64_t encode_ns; /* CPU time spent emitting the stage's work */
    uint64_t wait_ns;   /* CPU blocked until the device finished it   */
    uint64_t count;     /* boundaries with this id in the record      */
    uint64_t bytes;     /* bytes attributed to the stage, when known  */
} ds4_prof_stage_use;

typedef struct {
    int      device;
    uint32_t util_start;
    uint32_t util_end;
    uint64_t mem_start;
    uint64_t mem_end;
    uint64_t mem_total;
} ds4_prof_rec_dev;

typedef struct {
    uint64_t seq;
    uint8_t  kind;          /* ds4_prof_rec_kind */
    uint8_t  closed;
    uint16_t _pad;
    uint32_t token;         /* sampled token id, 0xFFFFFFFF when none */
    int32_t  pos;           /* position of the first token */
    uint32_t n_tokens;      /* tokens represented by the record */
    char     text[52];      /* decoded text / range preview */
    uint64_t start_ns;
    uint64_t end_ns;
    double   ms;

    uint32_t stage_n;
    ds4_prof_stage_use stages[DS4_PROF_REC_MAX_STAGES];

    uint32_t event_first;
    uint32_t event_n;
    uint32_t expert_first;
    uint32_t expert_n;

    uint32_t dev_n;
    ds4_prof_rec_dev dev[DS4_PROF_REC_MAX_DEV];

    uint64_t ssd_read_bytes;      /* delta over the record */
    uint64_t host_ram_used;       /* bytes used at end      */
    uint64_t host_ram_avail;      /* bytes available at end */
    uint64_t proc_cpu_ns;         /* process CPU delta      */
    uint64_t sys_cpu_busy_ns;     /* machine CPU busy delta */
    uint64_t sys_cpu_total_ns;    /* machine CPU total delta*/
    uint64_t tier_weight_bytes[DS4_PROF_MACHINE_TIERS];
    uint64_t tier_activation_bytes[DS4_PROF_MACHINE_TIERS];
    uint64_t tier_tx_bytes[DS4_PROF_MACHINE_TIERS];
    uint64_t tier_rx_bytes[DS4_PROF_MACHINE_TIERS];
} ds4_prof_record;

typedef struct {
    uint64_t seq;
    uint8_t  kind;
    uint8_t  closed;
    uint16_t _pad;
    uint32_t token;
    int32_t  pos;
    uint32_t n_tokens;
    double   ms;
    char     text[52];
    uint32_t events;
    uint32_t experts;
    uint16_t hot_stage;   /* stage with the largest total in the record */
    uint16_t _pad2;
    double   hot_ms;
    uint8_t  experts_pool;
    uint8_t  experts_arena;
    uint8_t  experts_ssd;
    uint8_t  experts_cpu;
    uint8_t  util_max;    /* max device utilisation at end */
    uint8_t  _pad3[3];
} ds4_prof_record_meta;

/* One stage's work inside a single record kind. A prefill chunk runs the same
 * boundaries as a decode token with completely different durations, so the two
 * are never averaged together: each kind keeps its own sums and its own share
 * denominator. */
typedef struct {
    double   total_ms;  /* sum of all calls of this kind */
    double   encode_ms;
    double   wait_ms;
    uint64_t calls;
    uint64_t bytes;
} ds4_prof_stage_kind;

typedef struct {
    char     key[48];       /* part:name, stable id for the TUI */
    char     label[56];     /* human label, no acronyms */
    char     part[16];
    char     name[40];
    char     category[16];
    char     hint[256];
    ds4_prof_stat total_ms;     /* all kinds mixed, kept for old consumers */
    ds4_prof_stat encode_ms;
    ds4_prof_stat wait_ms;
    uint64_t calls;
    uint64_t bytes;
    ds4_prof_stage_kind kind[DS4_PREC_KIND_MAX];
} ds4_prof_stage_view;

/* The accumulated timing snapshot, for the JSON serializer and tests. */
typedef struct {
    ds4_prof_stat per_token_ms;          /* the total; everything below sums to it */
    ds4_prof_stat host_ms;               /* CPU-driven part of the token           */
    ds4_prof_stat host_read_ms;          /*   of which: expert read/staging wait   */
    ds4_prof_stat device_wait_ms;        /* CPU blocked on the device              */
    ds4_prof_stat device_upload_ms;      /*   of which: expert upload stream       */
    ds4_prof_stat device_readback_ms;    /*   of which: device->host tensor reads  */
    ds4_prof_stat device_sync_ms;        /*   of which: other synchronisation      */
    ds4_prof_stat profiler_ms;           /*   of which: the profiler's own cost    */
    ds4_prof_stat profiler_sync_ms;      /*   of which: stage-boundary device syncs*/
    ds4_prof_stat weight_transfer_ms;    /* reference: bytes at the link peak      */
    ds4_prof_stat activation_transfer_ms;/* reference: bytes at the link peak      */
    ds4_prof_stat tokens_per_s;
    ds4_prof_stat prefill_ms;
    ds4_prof_stat prefill_tokens_per_s;
    uint64_t     tokens;
    uint64_t     prefill_tokens;   /* tokens covered by all counted chunks */
} ds4_profiler_timing;

typedef struct {
    ds4_profiler_timing timing;
    ds4_profiler_stats  stats;
    ds4_profiler_host_stats host;
    int n_stages;
    ds4_prof_stage_view stages[DS4_PROF_MAX_STAGES];
    int n_records;
    ds4_prof_record_meta records[DS4_PROF_REC_RING];
    int dev_n;
    ds4_prof_dev_state dev[DS4_PEV_DEV_MAX];
    uint64_t ssd_read_bytes;
    uint64_t host_ram_used;
    uint64_t host_ram_avail;
    uint64_t proc_cpu_ns;
} ds4_prof_view;

/* Open a record unless one is already open (frontends may wrap a speculative
 * group that calls the engine several times). Returns 1 when this call opened
 * it, 0 when a record was already open. token may be -1 for a chunk. */
int  ds4_profiler_token_begin_ex(ds4_prof_rec_kind kind, int token,
                                 int pos, uint32_t n_tokens);
/* Close the open record; `tokens` is the number of tokens it represents. */
void ds4_profiler_token_end(uint64_t tokens);
int  ds4_profiler_record_enabled(void);
void ds4_profiler_set_record_enabled(int on);
/* Annotate the open record with the token text (first 51 bytes, sanitised). */
void ds4_profiler_record_text(const char *text, size_t len);

/* Full aggregate view for the TUI / JSON. Returns the number of stage rows. */
int  ds4_profiler_view(ds4_prof_view *out);
int  ds4_profiler_record_list(ds4_prof_record_meta *out, int max);
int  ds4_profiler_record_get(uint64_t seq, ds4_prof_record *out);
int  ds4_profiler_record_events(uint64_t seq, ds4_prof_event *out, int max);
int  ds4_profiler_record_experts(uint64_t seq, ds4_prof_expert_trace *out, int max);
const char *ds4_profiler_record_kind_name(int kind);

/* ---- scheduler stage capture -------------------------------------------
 * The DS4 scheduler already brackets named stages. When capture is on those
 * boundaries feed the profiler as well as their optional stderr print. Each
 * stage costs a command-batch end, so capture serialises work: absolute ms
 * are inflated, but the split between CPU launch and device wait is real.
 */
void ds4_profiler_set_stage_capture(int on);
int  ds4_profiler_stage_capture_enabled(void);
void ds4_profiler_note_stage(const char *part, const char *name,
                             uint32_t layer, uint32_t pos, uint32_t n_tokens,
                             uint64_t total_ns, uint64_t encode_ns,
                             uint64_t wait_ns, uint64_t bytes);

/* Prefill is measured per sync call, not per token.  `chunks` is how many
 * prefill chunks the measurement covers: callers timing a single chunk pass 1,
 * callers timing a whole prompt pass the chunk count so per-chunk averages and
 * the stage tables stay honest. */
void ds4_profiler_note_prefill(double ms, uint64_t tokens, uint32_t chunks);

/* Drop all accumulated timing (e.g. between REPL turns). */
void ds4_profiler_reset_timing(void);

void ds4_profiler_timing_snapshot(ds4_profiler_timing *out);

/* ---- detail dump -------------------------------------------------------
 * One record, rendered start to end: its wall time, the share the measured
 * stages explain, every stage with its launch/wait split, every device before
 * and after, every expert use with the tier its bytes came from, and the
 * events in order. It fits a caller-supplied buffer, never overflows it, and
 * marks a cut. `kind` is a DS4_PREC_* value or -1 for every kind; `limit` <= 0
 * means all records.
 */
#define DS4_PROF_DUMP_CAP (1024u * 1024u)   /* room for a full decode token */

size_t ds4_profiler_record_dump(uint64_t seq, char *buf, size_t buflen);
size_t ds4_profiler_records_dump(char *buf, size_t buflen, int kind, int limit);
/* The whole picture in one call: rates, token split, devices, expert arena
 * reservation and fill, per-kind stage tables, then the newest records with
 * their full detail. This is what --profile-dump writes. */
size_t ds4_profiler_report(char *buf, size_t buflen, int kind, int limit);
/* Human byte count: at most three decimals, largest unit that keeps the value
 * below 1024 ("512 B", "6.75 MiB", "18 GiB"). */
void   ds4_profiler_format_bytes(char *out, size_t cap, uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* DS4_PROFILER_H */
