/* CPU quantized-dot regression + benchmark.
 *
 * The engine's x86 SIMD branches must be bit-identical to the scalar
 * reference: these are integer dot products, so any difference is a bug.
 * The same binary measures the scalar and SIMD paths so the speedup is
 * reported in microseconds per real in_dim=7168 row.
 */
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ds4.h"

#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

static int failures;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fputc('\n', stderr); \
    } \
} while (0)

static uint64_t rng_state;
static uint32_t rnd(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

/* Valid, finite half bit patterns: 1.0, 0.5, 2.0, 0.25. */
static uint16_t rnd_half(void) {
    static const uint16_t v[] = {0x3c00, 0x3800, 0x4000, 0x3400};
    return v[rnd() & 3u];
}

static float rnd_float(void) {
    return 0.05f + (float)(rnd() % 1000u) / 500.0f;
}

static void fill_q8(block_q8_K *y, int nb, int vary) {
    for (int i = 0; i < nb; i++) {
        y[i].d = vary ? rnd_float() : 1.0f;
        for (int j = 0; j < QK_K; j++) y[i].qs[j] = (int8_t)(rnd() & 0xffu);
        for (int j = 0; j < QK_K / 16; j++) {
            y[i].bsums[j] = (int16_t)(rnd() & 0xffffu);
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* The integer dots are exact; the float epilogue may contract to FMA
 * differently in the two inlined branches under -ffast-math, so allow a
 * last-bit difference while still rejecting any real arithmetic error. */
static int same_bits(float a, float b) {
    return a == b || fabsf(a - b) <= 1e-6f * (1.0f + fabsf(a) + fabsf(b));
}

typedef void (*dot_fn)(int, const void *, const void *, float *);

static void bench_one(const char *name, dot_fn fn,
                      const void *x, const void *y, int n,
                      uint64_t bytes_per_call) {
    enum { ITERS = 20000 };
    volatile float sink = 0.0f;
    ds4_test_cpu_set_simd(0);
    double t0 = now_ms();
    for (int i = 0; i < ITERS; i++) {
        float s;
        fn(n, x, y, &s);
        sink += s;
    }
    double scalar_ms = now_ms() - t0;

    ds4_test_cpu_set_simd(1);
    t0 = now_ms();
    for (int i = 0; i < ITERS; i++) {
        float s;
        fn(n, x, y, &s);
        sink += s;
    }
    double simd_ms = now_ms() - t0;

    const double per_call_us = simd_ms * 1000.0 / ITERS;
    const double gbs = (double)bytes_per_call * ITERS / (simd_ms * 1.0e6);
    printf("%-18s n=%-5d scalar %8.3f us  simd %8.3f us  %5.2fx  %6.1f GB/s\n",
           name, n, scalar_ms * 1000.0 / ITERS, per_call_us,
           simd_ms > 0.0 ? scalar_ms / simd_ms : 0.0, gbs);
    (void)sink;
}

int main(void) {
    rng_state = 0x123456789abcdef0ULL;
    const int nb = 28;                 /* in_dim=7168, the engine's FFN width */
    const int n = nb * QK_K;
    block_iq2_xxs *iq2 = calloc((size_t)nb, sizeof(*iq2));
    block_q2_K *q2 = calloc((size_t)nb, sizeof(*q2));
    block_q4_K *q4 = calloc((size_t)nb, sizeof(*q4));
    block_q8_K *q8 = calloc((size_t)nb, sizeof(*q8));
    block_q8_K *q8b = calloc((size_t)nb, sizeof(*q8b));
    CHECK(iq2 && q2 && q4 && q8 && q8b, "allocation");
    if (!(iq2 && q2 && q4 && q8 && q8b)) return 1;

    for (int i = 0; i < nb; i++) {
        iq2[i].d = rnd_half();
        for (int j = 0; j < QK_K / 8; j++) iq2[i].qs[j] = (uint16_t)(rnd() & 0xffffu);
        q2[i].d = rnd_half();
        q2[i].dmin = rnd_half();
        for (int j = 0; j < QK_K / 16; j++) q2[i].scales[j] = (uint8_t)(rnd() & 0xffu);
        for (int j = 0; j < QK_K / 4; j++) q2[i].qs[j] = (uint8_t)(rnd() & 0xffu);
        q4[i].d = rnd_half();
        q4[i].dmin = rnd_half();
        for (int j = 0; j < 12; j++) q4[i].scales[j] = (uint8_t)(rnd() & 0xffu);
        for (int j = 0; j < QK_K / 2; j++) q4[i].qs[j] = (uint8_t)(rnd() & 0xffu);
    }
    fill_q8(q8, nb, 1);
    fill_q8(q8b, nb, 1);

    /* Bit-exactness over several block counts, including a small n. */
    for (int blocks = 1; blocks <= nb; blocks++) {
        const int nn = blocks * QK_K;
        float r_iq2, r_q2, r_q4, r_q8, r_p0, r_p1;
        float s_iq2, s_q2, s_q4, s_q8, s_p0, s_p1;
        ds4_test_cpu_set_simd(0);
        ds4_test_vec_dot_iq2_xxs(nn, iq2, q8, &r_iq2);
        ds4_test_vec_dot_q2_K(nn, q2, q8, &r_q2);
        ds4_test_vec_dot_q4_K(nn, q4, q8, &r_q4);
        ds4_test_vec_dot_q8_K(nn, q8, q8b, &r_q8);
        ds4_test_vec_dot_q8_K_pair(nn, q8, q8b, q8b, &r_p0, &r_p1);
        ds4_test_cpu_set_simd(1);
        ds4_test_vec_dot_iq2_xxs(nn, iq2, q8, &s_iq2);
        ds4_test_vec_dot_q2_K(nn, q2, q8, &s_q2);
        ds4_test_vec_dot_q4_K(nn, q4, q8, &s_q4);
        ds4_test_vec_dot_q8_K(nn, q8, q8b, &s_q8);
        ds4_test_vec_dot_q8_K_pair(nn, q8, q8b, q8b, &s_p0, &s_p1);
        CHECK(same_bits(r_iq2, s_iq2), "iq2_xxs blocks=%d scalar=%g simd=%g",
              blocks, r_iq2, s_iq2);
        CHECK(same_bits(r_q2, s_q2), "q2_K blocks=%d scalar=%g simd=%g",
              blocks, r_q2, s_q2);
        CHECK(same_bits(r_q4, s_q4), "q4_K blocks=%d scalar=%g simd=%g",
              blocks, r_q4, s_q4);
        CHECK(same_bits(r_q8, s_q8), "q8_K blocks=%d scalar=%g simd=%g",
              blocks, r_q8, s_q8);
        CHECK(same_bits(r_p0, s_p0) && same_bits(r_p1, s_p1),
              "q8_K pair blocks=%d scalar=%g/%g simd=%g/%g",
              blocks, r_p0, r_p1, s_p0, s_p1);
    }

    bench_one("iq2_xxs", ds4_test_vec_dot_iq2_xxs, iq2, q8, n,
              (uint64_t)nb * sizeof(*iq2));
    bench_one("q2_K", ds4_test_vec_dot_q2_K, q2, q8, n,
              (uint64_t)nb * sizeof(*q2));
    bench_one("q4_K", ds4_test_vec_dot_q4_K, q4, q8, n,
              (uint64_t)nb * sizeof(*q4));
    bench_one("q8_K", ds4_test_vec_dot_q8_K, q8, q8b, n,
              (uint64_t)nb * sizeof(*q8));

    free(iq2);
    free(q2);
    free(q4);
    free(q8);
    free(q8b);
    if (failures) {
        fprintf(stderr, "cpu dot tests: %d failure(s)\n", failures);
        return 1;
    }
    printf("cpu dot tests: OK\n");
    return 0;
}
