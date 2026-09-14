#ifndef DS4_EXPERT_PLACEMENT_H
#define DS4_EXPERT_PLACEMENT_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Heterogeneous pinned-expert placement model.
 *
 * This is an analytical scheduler, not an execution backend: it takes the
 * observed per-(layer, expert) selection counts and a memory/cost model, and
 * reports where each expert would live and how much of the single-GPU
 * streaming that placement removes.
 *
 * Tiers are ordered coldest to hottest. The solver is greedy because the
 * objective is separable: every expert yields (selection_frequency x avoided
 * access cost) independently, so sorting by frequency and filling the fastest
 * tiers first is optimal for the byte-capacity constraint.
 */

enum {
    DS4_TIER_SSD      = 0, /* only in the GGUF on disk */
    DS4_TIER_RAM      = 1, /* host page cache, still streamed to the hot GPU */
    DS4_TIER_WARM_GPU = 2, /* PCIe 3.0 x4 card (e.g. RTX 5060 Ti), pinned */
    DS4_TIER_HOT_GPU  = 3, /* PCIe 4.0 x16 card (e.g. RTX 3080 Ti), pinned */
    DS4_TIER_COUNT    = 4,
};

typedef struct {
    uint32_t n_layer;
    uint32_t n_expert;
    uint64_t expert_bytes;                 /* bytes per routed expert */
    uint64_t tier_capacity[DS4_TIER_COUNT]; /* usable bytes per tier */
    bool     tier_active[DS4_TIER_COUNT];  /* false disables the tier */

    /* Cost model, microseconds unless noted. Documented so the estimate can
     * be reproduced; the placement/coverage itself does not depend on it. */
    double pcie_bytes_per_us;   /* host RAM -> hot GPU copy bandwidth */
    double ssd_bytes_per_us;    /* SSD -> RAM read bandwidth */
    double exchange_us;         /* per-layer activation exchange to warm GPU */
    double fixed_us;            /* per-token non-streaming cost on the hot GPU */
    double selections_per_token; /* routed expert selections per generated token */
} ds4_placement_config;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    double   frequency; /* selections per token */
    uint8_t  tier;
} ds4_placement_entry;

typedef struct {
    ds4_placement_entry *entries;   /* all experts, sorted by frequency desc */
    uint32_t n_entries;             /* experts with frequency > 0 */
    uint32_t n_expert_total;        /* n_layer * n_expert */

    uint64_t tier_bytes[DS4_TIER_COUNT];
    uint64_t tier_experts[DS4_TIER_COUNT];

    double total_selections;        /* selections per token */
    double vram_selections;         /* selections served from pinned VRAM */
    double selection_coverage;      /* vram_selections / total_selections */
    double vram_byte_coverage;      /* pinned expert bytes / total expert bytes */

    /* Decode model: baseline and placed per-token time and throughput. */
    double baseline_us;
    double placed_us;
    double baseline_decode_tps;
    double placed_decode_tps;

    /* Prefill proxy: all experts are touched in a wide chunk, so the stream
     * scales with bytes, not selection frequency. */
    double prefill_stream_avoided;  /* == vram_byte_coverage */
} ds4_placement_result;

void ds4_placement_config_default(ds4_placement_config *cfg);

/* counts is row-major [n_layer * n_expert]; may contain zeros. */
bool ds4_placement_solve(const ds4_placement_config *cfg,
                         const uint64_t              *counts,
                         ds4_placement_result        *out);

void ds4_placement_result_free(ds4_placement_result *r);

/* Load a profile hotlist in the ds4 expert hotlist text format:
 *   # comments
 *   <layer> <expert> <hits> <weight>
 * Returns a malloc'd counts array [n_layer*n_expert] and the layer/expert
 * dimensions implied by the file's headers, or NULL on error. */
uint64_t *ds4_placement_load_hotlist(const char *path,
                                     uint32_t   *n_layer,
                                     uint32_t   *n_expert,
                                     char       *err,
                                     unsigned    errlen);

#endif
