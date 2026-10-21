/* Runtime check for the CUDA device probe + role apply path. Runs on any CUDA
 * host; on a single-device machine it must resolve to single_gpu_mode with the
 * caller's first device as primary. Exits 0 with SKIP when no device is up. */
#include "ds4_gpu_roles.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    cfg.vram_bytes[0] = 1ull << 30;

    ds4_gpu_roles_config rc;
    ds4_gpu_roles_config_default(&rc);

    char err[256] = {0};
    if (ds4_gpu_roles_apply(&cfg, &rc, err, sizeof(err)) != 0) {
        printf("SKIP: no usable CUDA device (%s)\n", err[0] ? err : "probe failed");
        return 0;
    }
    const ds4_gpu_roles *r = ds4_gpu_roles_last();
    if (!r) {
        printf("FAIL: apply returned success but no plan was published\n");
        return 1;
    }
    printf("plan: n=%d primary=%d cache=%d secondary=%d single=%d chunk=%u\n",
           r->n, r->primary, r->cache, r->secondary,
           r->single_gpu_mode, r->chunk_size);
    if (r->n != 1 || r->single_gpu_mode != 1 || r->primary != 0 || r->cache != -1) {
        printf("FAIL: single-device plan is not the graceful degrade\n");
        return 1;
    }
    printf("ok test_gpu_roles_cuda\n");
    return 0;
}
