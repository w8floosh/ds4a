#include "ds4_expert_placement.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ds4_placement_config_default(ds4_placement_config *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layer = 43;
    cfg->n_expert = 256;
    cfg->expert_bytes = 7070000ull; /* ~7.07 MB: IQ2_XXS gate/up + Q2_K down */
    cfg->tier_capacity[DS4_TIER_SSD] = 0;               /* unbounded sink */
    cfg->tier_capacity[DS4_TIER_RAM] = 55ull << 30;     /* host page cache */
    cfg->tier_capacity[DS4_TIER_WARM_GPU] = 15ull << 30; /* RTX 5060 Ti 16 GB */
    cfg->tier_capacity[DS4_TIER_HOT_GPU] = 1ull << 30;   /* 3080 Ti spare */
    cfg->tier_active[DS4_TIER_SSD] = true;
    cfg->tier_active[DS4_TIER_RAM] = true;
    cfg->tier_active[DS4_TIER_WARM_GPU] = true;
    cfg->tier_active[DS4_TIER_HOT_GPU] = true;
    cfg->pcie_bytes_per_us = 19000.0; /* ~19 GB/s: PCIe 4.0 x16 copy */
    cfg->ssd_bytes_per_us = 3500.0;   /* ~3.5 GB/s NVMe */
    cfg->exchange_us = 40.0;          /* per-layer activation exchange to warm GPU */
    cfg->fixed_us = 15000.0;          /* non-streaming per-token cost */
    cfg->selections_per_token = 258.0; /* 43 layers x top-6 */
}

static int entry_cmp(const void *a, const void *b) {
    const ds4_placement_entry *ea = a;
    const ds4_placement_entry *eb = b;
    if (ea->frequency < eb->frequency) return 1;
    if (ea->frequency > eb->frequency) return -1;
    if (ea->layer != eb->layer) return ea->layer < eb->layer ? -1 : 1;
    if (ea->expert != eb->expert) return ea->expert < eb->expert ? -1 : 1;
    return 0;
}

void ds4_placement_result_free(ds4_placement_result *r) {
    if (!r) return;
    free(r->entries);
    memset(r, 0, sizeof(*r));
}

bool ds4_placement_solve(const ds4_placement_config *cfg,
                         const uint64_t              *counts,
                         ds4_placement_result        *out) {
    if (!cfg || !counts || !out || cfg->n_layer == 0 || cfg->n_expert == 0) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    const uint32_t n_total = cfg->n_layer * cfg->n_expert;

    /* Collect used experts and the totals the coverage is measured against. */
    uint32_t used = 0;
    double total = 0.0;
    for (uint32_t i = 0; i < n_total; i++) {
        if (counts[i] != 0) used++;
        total += (double)counts[i];
    }
    out->n_expert_total = n_total;
    out->total_selections = total;

    ds4_placement_entry *entries = NULL;
    if (used != 0) {
        entries = calloc(used, sizeof(*entries));
        if (!entries) return false;
        uint32_t n = 0;
        for (uint32_t il = 0; il < cfg->n_layer; il++) {
            for (uint32_t e = 0; e < cfg->n_expert; e++) {
                const uint64_t c = counts[(size_t)il * cfg->n_expert + e];
                if (c == 0) continue;
                entries[n].layer = il;
                entries[n].expert = e;
                entries[n].frequency = (double)c;
                n++;
            }
        }
        qsort(entries, used, sizeof(*entries), entry_cmp);
    }
    out->entries = entries;
    out->n_entries = used;

    /* Greedy fill from the hottest tier down. Sorting by frequency makes this
     * optimal for a byte capacity: each expert's contribution is independent,
     * so taking the largest frequencies first maximizes covered selections. */
    {
        uint64_t remaining[DS4_TIER_COUNT];
        for (int t = 0; t < DS4_TIER_COUNT; t++) {
            remaining[t] = cfg->tier_active[t] ? cfg->tier_capacity[t] : 0;
        }
        /* The SSD tier is the unbounded catch-all. */
        static const uint8_t order[4] = {
            DS4_TIER_HOT_GPU, DS4_TIER_WARM_GPU, DS4_TIER_RAM, DS4_TIER_SSD
        };
        for (uint32_t i = 0; i < used; i++) {
            uint8_t tier = DS4_TIER_SSD;
            for (int oi = 0; oi < 4; oi++) {
                const uint8_t t = order[oi];
                if (!cfg->tier_active[t]) continue;
                if (t == DS4_TIER_SSD ||
                    remaining[t] >= cfg->expert_bytes) {
                    tier = t;
                    if (t != DS4_TIER_SSD) remaining[t] -= cfg->expert_bytes;
                    break;
                }
            }
            entries[i].tier = tier;
            out->tier_experts[tier]++;
            out->tier_bytes[tier] += cfg->expert_bytes;
            if (tier == DS4_TIER_WARM_GPU || tier == DS4_TIER_HOT_GPU)
                out->vram_selections += entries[i].frequency;
        }
    }

    out->selection_coverage =
        total > 0.0 ? out->vram_selections / total : 0.0;
    {
        const uint64_t total_used_bytes = (uint64_t)used * cfg->expert_bytes;
        const uint64_t pinned_bytes =
            out->tier_bytes[DS4_TIER_WARM_GPU] +
            out->tier_bytes[DS4_TIER_HOT_GPU];
        out->vram_byte_coverage = total_used_bytes
            ? (double)pinned_bytes / (double)total_used_bytes : 0.0;
        out->prefill_stream_avoided = out->vram_byte_coverage;
    }

    /* Decode model. The baseline streams every routed expert from host RAM to
     * the hot GPU; pinned experts on either GPU are not streamed. */
    {
        const double bytes_per_token =
            cfg->selections_per_token * (double)cfg->expert_bytes;
        const double stream_us = bytes_per_token / cfg->pcie_bytes_per_us;
        out->baseline_us = cfg->fixed_us + stream_us;
        const double placed_stream =
            stream_us * (1.0 - out->selection_coverage);

        /* Warm-GPU exchange: for each layer, the chance a token selects at
         * least one warm-pinned expert, given six selections. */
        double exchange_us = 0.0;
        for (uint32_t il = 0; il < cfg->n_layer; il++) {
            double layer_total = 0.0, layer_warm = 0.0;
            for (uint32_t e = 0; e < cfg->n_expert; e++) {
                const double c = (double)counts[(size_t)il * cfg->n_expert + e];
                layer_total += c;
            }
            for (uint32_t i = 0; i < used; i++) {
                if (entries[i].layer == il &&
                    entries[i].tier == DS4_TIER_WARM_GPU)
                    layer_warm += entries[i].frequency;
            }
            if (layer_total > 0.0 && layer_warm > 0.0) {
                const double p = layer_warm / layer_total;
                exchange_us += cfg->exchange_us * (1.0 - pow(1.0 - p, 6.0));
            }
        }
        out->placed_us = cfg->fixed_us + placed_stream + exchange_us;
        out->baseline_decode_tps = out->baseline_us > 0.0
            ? 1.0e6 / out->baseline_us : 0.0;
        out->placed_decode_tps = out->placed_us > 0.0
            ? 1.0e6 / out->placed_us : 0.0;
    }
    return true;
}

uint64_t *ds4_placement_load_hotlist(const char *path,
                                     uint32_t   *n_layer,
                                     uint32_t   *n_expert,
                                     char       *err,
                                     unsigned    errlen) {
    if (err && errlen) err[0] = '\0';
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (err && errlen) snprintf(err, errlen, "cannot open %s", path);
        return NULL;
    }
    uint32_t layers = 0, experts = 0;
    uint64_t *counts = NULL;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') {
            unsigned a = 0, b = 0;
            if (sscanf(line, "# layers %u", &a) == 1 && a != 0) layers = a;
            if (sscanf(line, "# experts %u", &a) == 1 && a != 0) experts = a;
            (void)b;
            continue;
        }
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
        if (!layers || !experts) continue; /* wait for the header */
        if (!counts) {
            counts = calloc((size_t)layers * experts, sizeof(*counts));
            if (!counts) {
                if (err && errlen) snprintf(err, errlen, "out of memory");
                fclose(fp);
                return NULL;
            }
        }
        unsigned il = 0, e = 0;
        unsigned long long hits = 0;
        double weight = 0.0;
        if (sscanf(line, "%u %u %llu %lf", &il, &e, &hits, &weight) >= 3) {
            if (il < layers && e < experts)
                counts[(size_t)il * experts + e] = hits;
        }
    }
    fclose(fp);
    if (!counts || !layers || !experts) {
        if (err && errlen) snprintf(err, errlen,
                                    "hotlist is missing header or rows");
        free(counts);
        return NULL;
    }
    if (n_layer) *n_layer = layers;
    if (n_expert) *n_expert = experts;
    return counts;
}
