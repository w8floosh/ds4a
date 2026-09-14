#include "ds4_expert_placement.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t rng_state = 0x12345678u;
static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static double powerlaw(uint64_t *counts, uint32_t n_layer, uint32_t n_expert,
                       double exponent, uint32_t used_per_layer) {
    double total = 0.0;
    for (uint32_t il = 0; il < n_layer; il++) {
        /* Choose which experts of this layer are used. */
        uint32_t used = used_per_layer < n_expert ? used_per_layer : n_expert;
        uint32_t *picked = calloc(n_expert, sizeof(*picked));
        uint32_t picked_n = 0;
        while (picked_n < used) {
            const uint32_t e = rng_next() % n_expert;
            if (!picked[e]) { picked[e] = 1; picked_n++; }
        }
        /* Assign power-law weights by rank over the picked experts. */
        uint32_t rank = 0;
        for (uint32_t e = 0; e < n_expert; e++) {
            if (!picked[e]) continue;
            const double w = pow((double)(rank + 1), -exponent);
            counts[(size_t)il * n_expert + e] = (uint64_t)(w * 1.0e6) + 1;
            total += (double)counts[(size_t)il * n_expert + e];
            rank++;
        }
        free(picked);
    }
    return total;
}

/* Brute-force reference: top-N by count where N = floor(capacity/expert). */
static double reference_coverage(const uint64_t *counts, uint32_t n_total,
                                 uint32_t n_pin) {
    uint64_t *sorted = malloc((size_t)n_total * sizeof(*sorted));
    if (!sorted) return -1.0;
    memcpy(sorted, counts, (size_t)n_total * sizeof(*sorted));
    /* Simple selection of the n_pin largest (partial insertion). */
    for (uint32_t i = 0; i < n_pin && i < n_total; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n_total; j++)
            if (sorted[j] > sorted[best]) best = j;
        const uint64_t t = sorted[i]; sorted[i] = sorted[best]; sorted[best] = t;
    }
    double pinned = 0.0, total = 0.0;
    for (uint32_t i = 0; i < n_total; i++) total += (double)sorted[i];
    for (uint32_t i = 0; i < n_pin && i < n_total; i++) pinned += (double)sorted[i];
    free(sorted);
    return total > 0.0 ? pinned / total : 0.0;
}

static int test_uniform_capacity(void) {
    enum { L = 1, E = 100 };
    uint64_t counts[L * E];
    for (int i = 0; i < L * E; i++) counts[i] = 1;
    ds4_placement_config cfg;
    ds4_placement_config_default(&cfg);
    cfg.n_layer = L;
    cfg.n_expert = E;
    cfg.tier_capacity[DS4_TIER_SSD] = 0;
    cfg.tier_capacity[DS4_TIER_RAM] = (uint64_t)E * cfg.expert_bytes;
    cfg.tier_capacity[DS4_TIER_WARM_GPU] = (uint64_t)50 * cfg.expert_bytes;
    cfg.tier_capacity[DS4_TIER_HOT_GPU] = 0;
    ds4_placement_result r;
    CHECK(ds4_placement_solve(&cfg, counts, &r));
    CHECK(fabs(r.selection_coverage - 0.5) < 1e-9);
    CHECK(fabs(r.vram_byte_coverage - 0.5) < 1e-9);
    CHECK(r.tier_experts[DS4_TIER_WARM_GPU] == 50);
    ds4_placement_result_free(&r);
    printf("uniform capacity: PASS (coverage=%.4f)\n", 0.5);
    return 1;
}

static int test_greedy_matches_topn(void) {
    enum { L = 8, E = 64 };
    uint64_t counts[L * E];
    for (int i = 0; i < L * E; i++)
        counts[i] = (uint64_t)(rng_next() % 1000u) + 1;
    ds4_placement_config cfg;
    ds4_placement_config_default(&cfg);
    cfg.n_layer = L;
    cfg.n_expert = E;
    cfg.tier_capacity[DS4_TIER_RAM] = (uint64_t)L * E * cfg.expert_bytes;
    cfg.tier_capacity[DS4_TIER_WARM_GPU] = 100ull * cfg.expert_bytes;
    cfg.tier_capacity[DS4_TIER_HOT_GPU] = 20ull * cfg.expert_bytes;
    ds4_placement_result r;
    CHECK(ds4_placement_solve(&cfg, counts, &r));
    const double ref = reference_coverage(counts, L * E, 120);
    CHECK(ref >= 0.0);
    CHECK(fabs(r.selection_coverage - ref) < 1e-9);
    ds4_placement_result_free(&r);
    printf("greedy == top-N reference: PASS (coverage=%.4f)\n", ref);
    return 1;
}

static int test_no_vram(void) {
    enum { L = 4, E = 32 };
    uint64_t counts[L * E];
    for (int i = 0; i < L * E; i++) counts[i] = (uint64_t)(i + 1);
    ds4_placement_config cfg;
    ds4_placement_config_default(&cfg);
    cfg.n_layer = L;
    cfg.n_expert = E;
    cfg.tier_capacity[DS4_TIER_WARM_GPU] = 0;
    cfg.tier_capacity[DS4_TIER_HOT_GPU] = 0;
    cfg.tier_capacity[DS4_TIER_RAM] = (uint64_t)L * E * cfg.expert_bytes;
    ds4_placement_result r;
    CHECK(ds4_placement_solve(&cfg, counts, &r));
    CHECK(r.selection_coverage == 0.0);
    CHECK(r.tier_experts[DS4_TIER_RAM] == L * E);
    CHECK(r.placed_decode_tps == r.baseline_decode_tps);
    ds4_placement_result_free(&r);
    printf("no VRAM tier: PASS\n");
    return 1;
}

static int test_full_vram(void) {
    enum { L = 2, E = 16 };
    uint64_t counts[L * E];
    for (int i = 0; i < L * E; i++) counts[i] = (uint64_t)(i + 1);
    ds4_placement_config cfg;
    ds4_placement_config_default(&cfg);
    cfg.n_layer = L;
    cfg.n_expert = E;
    cfg.tier_capacity[DS4_TIER_WARM_GPU] =
        (uint64_t)L * E * cfg.expert_bytes + 1;
    ds4_placement_result r;
    CHECK(ds4_placement_solve(&cfg, counts, &r));
    CHECK(fabs(r.selection_coverage - 1.0) < 1e-12);
    CHECK(fabs(r.vram_byte_coverage - 1.0) < 1e-12);
    CHECK(r.placed_decode_tps > r.baseline_decode_tps);
    ds4_placement_result_free(&r);
    printf("full VRAM tier: PASS\n");
    return 1;
}

static int test_skew_helps(void) {
    enum { L = 43, E = 256 };
    uint64_t *counts = calloc((size_t)L * E, sizeof(*counts));
    CHECK(counts);
    rng_state = 0xabcdef01u;
    powerlaw(counts, L, E, 0.9, 140);
    ds4_placement_config cfg;
    ds4_placement_config_default(&cfg);
    cfg.n_layer = L;
    cfg.n_expert = E;
    cfg.tier_capacity[DS4_TIER_RAM] = 55ull << 30;
    cfg.tier_capacity[DS4_TIER_WARM_GPU] = 15ull << 30;
    cfg.tier_capacity[DS4_TIER_HOT_GPU] = 1ull << 30;
    ds4_placement_result r;
    CHECK(ds4_placement_solve(&cfg, counts, &r));
    const double byte_share =
        (double)(16ull << 30) /
        ((double)r.n_entries * (double)cfg.expert_bytes);
    const double uniform_expect = byte_share; /* if all frequencies equal */
    CHECK(r.selection_coverage > uniform_expect * 1.5);
    CHECK(r.placed_decode_tps > r.baseline_decode_tps * 1.5);
    printf("skew concentration: PASS (bytes=%.1f%% coverage=%.1f%% tps %.2f -> %.2f)\n",
           byte_share * 100.0, r.selection_coverage * 100.0,
           r.baseline_decode_tps, r.placed_decode_tps);
    ds4_placement_result_free(&r);
    free(counts);
    return 1;
}

static void print_report(const ds4_placement_config *cfg,
                         const ds4_placement_result *r) {
    (void)cfg;
    printf("\n");
    printf("  used experts        : %u of %u (%llu pinned)\n",
           r->n_entries, r->n_expert_total,
           (unsigned long long)(r->tier_experts[DS4_TIER_WARM_GPU] +
                                r->tier_experts[DS4_TIER_HOT_GPU]));
    printf("  total selections    : %.0f\n", r->total_selections);
    printf("  tier bytes          : ssd=%.2f GiB ram=%.2f GiB warm=%.2f GiB hot=%.2f GiB\n",
           r->tier_bytes[DS4_TIER_SSD] / 1073741824.0,
           r->tier_bytes[DS4_TIER_RAM] / 1073741824.0,
           r->tier_bytes[DS4_TIER_WARM_GPU] / 1073741824.0,
           r->tier_bytes[DS4_TIER_HOT_GPU] / 1073741824.0);
    printf("  expert-byte coverage: %.1f%% (prefill stream removed)\n",
           r->prefill_stream_avoided * 100.0);
    printf("  selection coverage  : %.1f%% (decode stream removed)\n",
           r->selection_coverage * 100.0);
    printf("  decode              : %.2f t/s -> %.2f t/s (%.2fx)\n",
           r->baseline_decode_tps, r->placed_decode_tps,
           r->baseline_decode_tps > 0.0
               ? r->placed_decode_tps / r->baseline_decode_tps : 0.0);
}

static int synthetic_report(void) {
    enum { L = 43, E = 256 };
    uint64_t *counts = calloc((size_t)L * E, sizeof(*counts));
    CHECK(counts);
    const double exponents[] = {0.6, 0.8, 1.0};
    const uint64_t caps[] = {8ull << 30, 15ull << 30, 23ull << 30,
                             46ull << 30};
    for (unsigned ei = 0; ei < sizeof(exponents) / sizeof(*exponents); ei++) {
        for (unsigned ci = 0; ci < sizeof(caps) / sizeof(*caps); ci++) {
            memset(counts, 0, (size_t)L * E * sizeof(*counts));
            rng_state = 0x5eed0000u + ei;
            powerlaw(counts, L, E, exponents[ei], 140);
            ds4_placement_config cfg;
            ds4_placement_config_default(&cfg);
            cfg.n_layer = L;
            cfg.n_expert = E;
            cfg.tier_capacity[DS4_TIER_RAM] = 55ull << 30;
            cfg.tier_capacity[DS4_TIER_WARM_GPU] = caps[ci];
            cfg.tier_capacity[DS4_TIER_HOT_GPU] = 1ull << 30;
            ds4_placement_result r;
            CHECK(ds4_placement_solve(&cfg, counts, &r));
            printf("\n[synthetic s=%.1f pinned=%llu GiB] stream avoided decode=%.1f%% prefill=%.1f%%  %.2f -> %.2f t/s",
                   exponents[ei], (unsigned long long)(caps[ci] >> 30),
                   r.selection_coverage * 100.0,
                   r.prefill_stream_avoided * 100.0,
                   r.baseline_decode_tps, r.placed_decode_tps);
            ds4_placement_result_free(&r);
        }
    }
    printf("\n");
    free(counts);
    return 1;
}

static int profile_report(const char *path) {
    uint32_t n_layer = 0, n_expert = 0;
    char err[256];
    uint64_t *counts = ds4_placement_load_hotlist(path, &n_layer, &n_expert,
                                                  err, sizeof(err));
    CHECK(counts);
    printf("loaded hotlist %s: %u layers x %u experts\n",
           path, n_layer, n_expert);
    const uint64_t caps[] = {15ull << 30, 23ull << 30, 46ull << 30};
    for (unsigned ci = 0; ci < sizeof(caps) / sizeof(*caps); ci++) {
        ds4_placement_config cfg;
        ds4_placement_config_default(&cfg);
        cfg.n_layer = n_layer;
        cfg.n_expert = n_expert;
        cfg.tier_capacity[DS4_TIER_RAM] = 55ull << 30;
        cfg.tier_capacity[DS4_TIER_WARM_GPU] = caps[ci];
        cfg.tier_capacity[DS4_TIER_HOT_GPU] = 1ull << 30;
        ds4_placement_result r;
        CHECK(ds4_placement_solve(&cfg, counts, &r));
        printf("\n[real profile pinned=%llu GiB]",
               (unsigned long long)(caps[ci] >> 30));
        print_report(&cfg, &r);
        ds4_placement_result_free(&r);
    }
    free(counts);
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--profile"))
        return profile_report(argv[2]) ? 0 : 1;
    if (argc == 2 && !strcmp(argv[1], "--synthetic"))
        return synthetic_report() ? 0 : 1;
    if (argc != 1) {
        fprintf(stderr,
                "usage: %s [--profile HOTLIST | --synthetic]\n", argv[0]);
        return 2;
    }
    int ok = 1;
    ok = test_uniform_capacity() && ok;
    ok = test_greedy_matches_topn() && ok;
    ok = test_no_vram() && ok;
    ok = test_full_vram() && ok;
    ok = test_skew_helps() && ok;
    printf("\nexpert placement tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
