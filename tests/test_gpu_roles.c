/* Unit tests for the optimizations.md §1 device scoring / role assignment.
 * Compiled with -DDS4_NO_GPU: no CUDA device is needed; probes are synthetic
 * and the §1.4 example from the document is reproduced exactly. */
#include "ds4_gpu_roles.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

/* §1.4 few-shot: the two-device example printed in the document. */
static int test_document_example(void) {
    ds4_gpu_device_probe p[2];
    memset(p, 0, sizeof(p));
    /* dev0: fast decode, small free VRAM. dev1: big free VRAM, weak compute. */
    p[0].device_id = 0;
    snprintf(p[0].name, sizeof(p[0].name), "dev0");
    p[0].pcie_bw_gbps = 1.0;
    p[0].vram_free_bytes = (uint64_t)(2.5 * (1u << 30));
    p[0].vram_bw_gbps = 912.0;
    p[0].fp16_tflops = 34.0;
    p[0].sm_count = 80;
    p[1].device_id = 1;
    snprintf(p[1].name, sizeof(p[1].name), "dev1");
    p[1].pcie_bw_gbps = 1.0;
    p[1].vram_free_bytes = (uint64_t)(11.5 * (1u << 30));
    p[1].vram_bw_gbps = 360.0;
    p[1].fp16_tflops = 13.0;
    p[1].sm_count = 28;

    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    ds4_gpu_roles r;
    CHECK(ds4_gpu_roles_plan(p, 2, &cfg, &r) == 0);
    CHECK(r.primary == 0);
    CHECK(r.cache == 1);
    CHECK(r.single_gpu_mode == 0);
    CHECK(r.expert_sharding == 1);
    CHECK(r.cache_sharding == 1);
    CHECK(r.role[0] == DS4_GPU_ROLE_PRIMARY);
    CHECK(r.role[1] == DS4_GPU_ROLE_CACHE);
    /* Decode in the document: dev0 0.8, dev1 0.2. The vram_bw term in
     * S_decode is the 0.4 weight, so the two must land on opposite sides. */
    CHECK(r.s_decode[0] > r.s_decode[1]);
    CHECK(r.s_capacity[1] > r.s_capacity[0]);
    return 1;
}

static int test_single_gpu_degrades(void) {
    ds4_gpu_device_probe p[1];
    memset(p, 0, sizeof(p));
    p[0].device_id = 3;
    p[0].sm_count = 80;
    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    ds4_gpu_roles r;
    CHECK(ds4_gpu_roles_plan(p, 1, &cfg, &r) == 0);
    CHECK(r.single_gpu_mode == 1);
    CHECK(r.primary == 0);
    CHECK(r.cache == -1);
    CHECK(r.secondary == -1);
    CHECK(r.expert_sharding == 0);
    CHECK(r.pipeline_parallel == 0);
    return 1;
}

static int test_disable_multi_gpu(void) {
    ds4_gpu_device_probe p[2];
    memset(p, 0, sizeof(p));
    p[0].device_id = 0; p[0].vram_free_bytes = 4u << 30; p[0].fp16_tflops = 40;
    p[1].device_id = 1; p[1].vram_free_bytes = 20u << 30; p[1].fp16_tflops = 5;
    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    cfg.disable_multi_gpu = 1;
    ds4_gpu_roles r;
    CHECK(ds4_gpu_roles_plan(p, 2, &cfg, &r) == 0);
    CHECK(r.single_gpu_mode == 1);
    CHECK(r.primary == 0);       /* keeps the caller's first device */
    CHECK(r.cache == -1);
    return 1;
}

static int test_force_primary(void) {
    ds4_gpu_device_probe p[2];
    memset(p, 0, sizeof(p));
    p[0].device_id = 0; p[0].vram_free_bytes = 4u << 30;  p[0].fp16_tflops = 40; p[0].vram_bw_gbps = 900;
    p[1].device_id = 1; p[1].vram_free_bytes = 20u << 30; p[1].fp16_tflops = 5;  p[1].vram_bw_gbps = 300;
    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    cfg.force_primary_gpu = 1;
    ds4_gpu_roles r;
    CHECK(ds4_gpu_roles_plan(p, 2, &cfg, &r) == 0);
    CHECK(r.primary == 1);
    CHECK(r.role[1] == DS4_GPU_ROLE_PRIMARY);
    CHECK(r.cache == 0);         /* capacity winner excluding primary */
    return 1;
}

static int test_secondary_needs_three(void) {
    ds4_gpu_device_probe p[3];
    memset(p, 0, sizeof(p));
    for (int i = 0; i < 3; i++) {
        p[i].device_id = i;
        p[i].vram_free_bytes = (uint64_t)(i + 1) << 30;
        p[i].fp16_tflops = 10.0 * (i + 1);
        p[i].vram_bw_gbps = 100.0 * (i + 1);
        p[i].pcie_bw_gbps = 1.0 + i;
        p[i].sm_count = 10 * (i + 1);
    }
    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    ds4_gpu_roles r;
    CHECK(ds4_gpu_roles_plan(p, 3, &cfg, &r) == 0);
    CHECK(r.pipeline_parallel == 1);
    CHECK(r.secondary >= 0);
    CHECK(r.secondary != r.primary && r.secondary != r.cache);
    /* The prefill winner among the two unassigned devices is the higher-SM
     * one as long as it was not already taken for decode/capacity. */
    CHECK(r.role[r.secondary] == DS4_GPU_ROLE_SECONDARY);
    return 1;
}

static int test_chunk_table(void) {
    const uint64_t gb = 1ull << 30;
    CHECK(ds4_gpu_roles_pick_chunk(0) == 512u);
    CHECK(ds4_gpu_roles_pick_chunk(gb - 1) == 512u);
    CHECK(ds4_gpu_roles_pick_chunk(gb) == 2048u);
    CHECK(ds4_gpu_roles_pick_chunk(2 * gb) == 4096u);
    CHECK(ds4_gpu_roles_pick_chunk(4 * gb) == 8192u);
    return 1;
}

static int test_chunk_override(void) {
    ds4_gpu_device_probe p[1];
    memset(p, 0, sizeof(p));
    p[0].vram_free_bytes = 8u << 30;
    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    cfg.chunk_size = 2048;       /* §11.3 --chunk-size wins over the table */
    ds4_gpu_roles r;
    CHECK(ds4_gpu_roles_plan(p, 1, &cfg, &r) == 0);
    CHECK(r.chunk_size == 2048u);
    return 1;
}

static int test_determinism(void) {
    ds4_gpu_device_probe p[2];
    memset(p, 0, sizeof(p));
    p[0].device_id = 0; p[0].vram_free_bytes = 3u << 30; p[0].fp16_tflops = 20; p[0].vram_bw_gbps = 500;
    p[1].device_id = 1; p[1].vram_free_bytes = 3u << 30; p[1].fp16_tflops = 20; p[1].vram_bw_gbps = 500;
    ds4_gpu_roles_config cfg;
    ds4_gpu_roles_config_default(&cfg);
    ds4_gpu_roles a, b;
    CHECK(ds4_gpu_roles_plan(p, 2, &cfg, &a) == 0);
    CHECK(ds4_gpu_roles_plan(p, 2, &cfg, &b) == 0);
    CHECK(a.primary == b.primary && a.cache == b.cache &&
          a.secondary == b.secondary);
    CHECK(a.primary == 0);       /* ties break to the lowest index */
    return 1;
}

static int test_kv_location_parse(void) {
    int v = 0;
    CHECK(ds4_gpu_roles_parse_kv_location("vram", &v) == 0 && v == DS4_KV_LOC_VRAM);
    CHECK(ds4_gpu_roles_parse_kv_location("ram", &v) == 0 && v == DS4_KV_LOC_RAM);
    CHECK(ds4_gpu_roles_parse_kv_location("auto", &v) == 0 && v == DS4_KV_LOC_AUTO);
    CHECK(ds4_gpu_roles_parse_kv_location("gpu:1", &v) == 0 && v == 1);
    CHECK(ds4_gpu_roles_parse_kv_location("disk", &v) != 0);
    CHECK(ds4_gpu_roles_parse_kv_location("gpu:", &v) != 0);
    return 1;
}

int main(void) {
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"document_example", test_document_example},
        {"single_gpu_degrades", test_single_gpu_degrades},
        {"disable_multi_gpu", test_disable_multi_gpu},
        {"force_primary", test_force_primary},
        {"secondary_needs_three", test_secondary_needs_three},
        {"chunk_table", test_chunk_table},
        {"chunk_override", test_chunk_override},
        {"determinism", test_determinism},
        {"kv_location_parse", test_kv_location_parse},
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (tests[i].fn()) {
            printf("ok   %s\n", tests[i].name);
        } else {
            printf("FAIL %s\n", tests[i].name);
            failed = 1;
        }
    }
    return failed;
}
