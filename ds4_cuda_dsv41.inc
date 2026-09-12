/*
 * DeepSeek V4.1 Flash CUDA primitives.
 *
 * Included at the end of ds4_cuda.cu so the V4.1 graph in ds4.c can run on
 * CUDA without touching the existing CUDA implementation. The graph
 * orchestration is backend-neutral; this file only provides the primitives
 * the graph calls, ported from metal/dsv41.metal and the Metal wrappers in
 * ds4_metal.m.
 *
 * Scope notes:
 *   - The packed tensor-core indexer path is a Metal optimization. On CUDA
 *     ds4_gpu_dsv41_tensor_ops_available() returns 0, so the graph uses the
 *     F32 GLM indexer fallback and never calls the packed entry points.
 *   - V4.1 tensor parallelism is not wired for CUDA (the session leaves
 *     tp_world = 1); the TP entry points are best-effort shims.
 *   - Kernels are written for Ampere (sm_80+) but avoid architecture-specific
 *     intrinsics so the same source builds for newer targets.
 */

/* ds4_cuda.cu deliberately does not include ds4_gpu.h (it mirrors the shared
 * types locally to avoid Metal/C conflicts). Repeat the two small V4.1 enums
 * here; C linkage makes them ABI-compatible with the declarations in
 * ds4_gpu.h that ds4.c compiles against. */
typedef enum {
    DS4_V41_BF16 = 0,
    DS4_V41_FP8_E8M0 = 1,
    DS4_V41_FP4_E8M0 = 2,
    DS4_V41_FP4_E4M3 = 3,
} ds4_v41_activation_format;
enum { DS4_V41_CARRY_BF16, DS4_V41_CARRY_MASK, DS4_V41_CARRY_F32 };

#define DSV41_MAX_ROPE_FREQ 32u

__device__ __forceinline__ float dsv41_bf16(float x) {
    unsigned bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ __forceinline__ float dsv41_pow2_ceil(float x) {
    const unsigned bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) +
                           ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__device__ __forceinline__ float dsv41_e4m3_value(int i) {
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

/* Round-to-nearest E4M3FN value, matching the Metal dequant table search. */
__device__ float dsv41_e4m3_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv41_e4m3_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv41_e4m3_value(best));
        const float next_diff = fabsf(ax - dsv41_e4m3_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0))
            best = best + 1;
    }
    return sign * dsv41_e4m3_value(best);
}

__device__ float dsv41_e2m1_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0;
    float best_diff = fabsf(ax - values[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - values[i]);
        if (diff < best_diff ||
            (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * values[best];
}

__device__ __forceinline__ float dsv41_warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float dsv41_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

/* -------------------------------------------------------------------------
 * Activation quantization (BF16 / FP8-E8M0 / FP4-E8M0 / FP4-E4M3).
 * ---------------------------------------------------------------------- */

__global__ void dsv41_bf16_linear_kernel(float *x, unsigned long long count) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += stride)
        x[i] = dsv41_bf16(x[i]);
}

__global__ void dsv41_quantize_kernel(float *x, unsigned width, unsigned rows,
                                      int mode) {
    const unsigned block = mode == 3 ? 16u : 32u;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned column = blockIdx.x * block + lane;
    const bool valid = lane < block && column < width;
    const size_t index = (size_t)blockIdx.y * width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = dsv41_warp_max(fabsf(value));
    float result = value;
    if (mode == 1) {
        const float scale = dsv41_pow2_ceil(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysignf(dsv41_e4m3_dequant(fabsf(value) / scale), value) * scale;
    } else if (mode == 2 || mode == 3) {
        const float scale = mode == 3
            ? dsv41_e4m3_dequant(fmaxf(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(fmaxf(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysignf(dsv41_e2m1_dequant(fabsf(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width,
                                      uint32_t rows,
                                      ds4_v41_activation_format format) {
    if (!x || !width || !rows || (int)format < 0 ||
        (int)format > (int)DS4_V41_FP4_E4M3 ||
        x->bytes < (uint64_t)width * rows * sizeof(float)) return 0;
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    if (format != DS4_V41_BF16 && (width % block)) return 0;
    if (format == DS4_V41_BF16) {
        const uint64_t count = (uint64_t)width * rows;
        const unsigned threads = 256u;
        uint64_t blocks = (count + threads - 1u) / threads;
        if (blocks > 0x7fffffffULL) return 0;
        if (blocks == 0) return 1;
        dsv41_bf16_linear_kernel<<<(unsigned)blocks, threads>>>(
                (float *)x->ptr, (unsigned long long)count);
    } else {
        dim3 grid((width + block - 1u) / block, rows);
        dsv41_quantize_kernel<<<grid, 32>>>((float *)x->ptr, width, rows,
                                            (int)format);
    }
    return cuda_ok(cudaGetLastError(), "dsv41 quantize launch");
}

/* -------------------------------------------------------------------------
 * Unit-magnitude RoPE with the released V4.1 frequencies.
 * ---------------------------------------------------------------------- */

struct dsv41_rope_args {
    unsigned width, heads, rows, start, stride, inverse;
    float frequencies[DSV41_MAX_ROPE_FREQ];
};

__global__ void dsv41_rope_kernel(float *x, struct dsv41_rope_args a) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned head = blockIdx.x;
    const unsigned row = blockIdx.y;
    const double theta = (double)(a.start + row * a.stride) *
                         (double)a.frequencies[lane];
    const float c = (float)cos(theta);
    const float s = a.inverse ? -(float)sin(theta) : (float)sin(theta);
    const size_t i = ((size_t)row * a.heads + head) * a.width +
                     a.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

static void dsv41_rope_frequencies(int compressed, float out[DSV41_MAX_ROPE_FREQ]) {
    const double base = compressed ? 160000.0 : 10000.0;
    const double low = floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) /
                             (2.0 * log(base)));
    const double high = ceil(64.0 * log(65536.0 / (2.0 * M_PI)) /
                             (2.0 * log(base)));
    for (int i = 0; i < 32; i++) {
        const double denominator = pow(base, (double)i / 32.0);
        double f = 1.0 / denominator;
        if (compressed) {
            const double ramp = fmin(1.0, fmax(0.0, ((double)i - low) /
                                                      (high - low)));
            const double smooth = 1.0 - ramp;
            const double interpolate = (f / 16.0) * (1.0 - smooth);
            const double extrapolate = f * smooth;
            f = interpolate + extrapolate;
        }
        out[i] = (float)f;
    }
}

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width,
                                         uint32_t heads, uint32_t rows,
                                         uint32_t start, uint32_t stride,
                                         bool compressed, bool inverse) {
    if (!x || width < 64u || !heads || !rows || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT64_MAX / width ||
        x->bytes < (uint64_t)width * heads * rows * sizeof(float)) return 0;
    struct dsv41_rope_args args;
    memset(&args, 0, sizeof(args));
    args.width = width;
    args.heads = heads;
    args.rows = rows;
    args.start = start;
    args.stride = stride;
    args.inverse = inverse ? 1u : 0u;
    dsv41_rope_frequencies(compressed ? 1 : 0, args.frequencies);
    dim3 grid(heads, rows);
    dsv41_rope_kernel<<<grid, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "dsv41 rope launch");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width,
                                  uint32_t heads, uint32_t rows, uint32_t start,
                                  bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1u,
                                     compressed, inverse);
}

/* -------------------------------------------------------------------------
 * Engram residual gate.
 * ---------------------------------------------------------------------- */

struct dsv41_engram_args {
    unsigned width, rows, masked;
    float eps;
};

__global__ void dsv41_engram_add_kernel(float *residual, const float *kv,
                                        const float *q_weight,
                                        const float *k_weight,
                                        const unsigned char *mask,
                                        struct dsv41_engram_args a) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gx = blockIdx.x;
    const unsigned gy = blockIdx.y;
    if (a.masked && !mask[gx]) return;
    const size_t offset = ((size_t)gx * 4u + gy) * a.width;
    const size_t key_offset = ((size_t)gx * 5u + gy) * a.width;
    const size_t value_offset = ((size_t)gx * 5u + 4u) * a.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (unsigned i = lane; i < a.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const unsigned wi = gy * a.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = dsv41_warp_sum(h2);
    k2 = dsv41_warp_sum(k2);
    dot = dsv41_warp_sum(dot) * rsqrtf(h2 / (float)a.width + a.eps) *
          rsqrtf(k2 / (float)a.width + a.eps) * rsqrtf((float)a.width);
    const float gate = 1.0f /
        (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
    for (unsigned i = lane; i < a.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual,
                                        const ds4_gpu_tensor *kv,
                                        const ds4_gpu_tensor *q_weight,
                                        const ds4_gpu_tensor *k_weight,
                                        const ds4_gpu_tensor *mask,
                                        uint32_t width, uint32_t rows,
                                        float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!residual || !kv || !q_weight || !k_weight || !width || !rows ||
        !isfinite(eps) || eps <= 0.0f || count > UINT64_MAX / 5u ||
        residual->bytes < count * 4u * sizeof(float) ||
        kv->bytes < count * 5u * sizeof(float) ||
        q_weight->bytes < (uint64_t)width * 4u * sizeof(float) ||
        k_weight->bytes < (uint64_t)width * 4u * sizeof(float) ||
        (mask && mask->bytes < rows)) return 0;
    struct dsv41_engram_args args = {width, rows, mask ? 1u : 0u, eps};
    dim3 grid(rows, 4);
    dsv41_engram_add_kernel<<<grid, 32>>>(
            (float *)residual->ptr, (const float *)kv->ptr,
            (const float *)q_weight->ptr, (const float *)k_weight->ptr,
            mask ? (const unsigned char *)mask->ptr : NULL, args);
    return cuda_ok(cudaGetLastError(), "dsv41 engram add launch");
}

/* -------------------------------------------------------------------------
 * Compressed KV pair pooling.
 * ---------------------------------------------------------------------- */

struct dsv41_pool_args {
    unsigned width, pairs, tail;
};

__global__ void dsv41_pool2_kernel(float *out, const float *kv,
                                   const float *scores, const float *prev_kv,
                                   const float *prev_scores,
                                   struct dsv41_pool_args a) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (x >= a.width || pair >= a.pairs) return;
    const long idx = (long)pair * 2 - (long)a.tail;
    const size_t b = (size_t)(idx + 1) * a.width + x;
    const float ka = idx < 0 ? prev_kv[x] : kv[(size_t)idx * a.width + x];
    const float sa = idx < 0 ? prev_scores[x] : scores[(size_t)idx * a.width + x];
    const float sb = scores[b];
    const float peak = fmaxf(sa, sb);
    const float ea = expf(sa - peak), eb = expf(sb - peak);
    out[(size_t)pair * a.width + x] =
        dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out,
                                   const ds4_gpu_tensor *kv,
                                   const ds4_gpu_tensor *scores,
                                   ds4_gpu_tensor *previous_kv,
                                   ds4_gpu_tensor *previous_scores,
                                   uint32_t width, uint32_t rows,
                                   uint32_t start) {
    const uint64_t count = (uint64_t)width * rows;
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    if (!kv || !scores || !previous_kv || !previous_scores ||
        !width || !rows || rows > UINT32_MAX - start ||
        (pairs && !out) ||
        kv->bytes < count * sizeof(float) ||
        scores->bytes < count * sizeof(float) ||
        previous_kv->bytes < (uint64_t)width * sizeof(float) ||
        previous_scores->bytes < (uint64_t)width * sizeof(float) ||
        (pairs && out->bytes < (uint64_t)width * pairs * sizeof(float))) return 0;
    if (pairs) {
        struct dsv41_pool_args args = {width, pairs, start & 1u};
        dim3 grid((width + 255u) / 256u, pairs);
        dsv41_pool2_kernel<<<grid, 256>>>(
                (float *)out->ptr, (const float *)kv->ptr,
                (const float *)scores->ptr, (const float *)previous_kv->ptr,
                (const float *)previous_scores->ptr, args);
        if (!cuda_ok(cudaGetLastError(), "dsv41 pool2 launch")) return 0;
    }
    /* Match the row-at-a-time path: retain the last even-position input. */
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    if (last_even >= start) {
        const uint64_t bytes = (uint64_t)width * sizeof(float);
        const uint64_t offset = (uint64_t)(last_even - start) * bytes;
        if (!ds4_gpu_tensor_copy(previous_kv, 0, kv, offset, bytes) ||
            !ds4_gpu_tensor_copy(previous_scores, 0, scores, offset, bytes))
            return 0;
    }
    return 1;
}

/* -------------------------------------------------------------------------
 * Causal candidate block maxima / filtering.
 * ---------------------------------------------------------------------- */

__global__ void dsv41_candidate_blocks_kernel(const float *scores, float *blocks,
                                              unsigned width, unsigned rows,
                                              unsigned start, unsigned ratio) {
    const unsigned count = (width + 7u) / 8u;
    const unsigned bx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y;
    if (bx >= count || row >= rows) return;
    const unsigned visible = min(width, (start + row + 1u) / ratio);
    float best = -INFINITY;
    for (unsigned i = bx * 8u; i < min(visible, (bx + 1u) * 8u); i++)
        best = fmaxf(best, scores[(size_t)row * width + i]);
    if (visible && bx == (visible - 1u) / 8u) best = INFINITY;
    blocks[(size_t)row * count + bx] = best;
}

__global__ void dsv41_candidate_filter_kernel(const float *scores, float *out,
                                              const float *block_mask,
                                              unsigned width, unsigned rows,
                                              unsigned start, unsigned ratio) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y;
    if (x >= width || row >= rows) return;
    const size_t offset = (size_t)row * width + x;
    const unsigned blocks = (width + 7u) / 8u;
    const unsigned visible = min(width, (start + row + 1u) / ratio);
    out[offset] = (x < visible &&
                   block_mask[(size_t)row * blocks + x / 8u] == 0.0f)
        ? scores[offset] : -INFINITY;
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks,
                                              const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows,
                                              uint32_t start, uint32_t ratio) {
    if (!blocks || !scores || !width || width > UINT32_MAX - 7u || !rows ||
        !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t count = (width + 7u) / 8u;
    if (scores->bytes < (uint64_t)width * rows * sizeof(float) ||
        blocks->bytes < (uint64_t)count * rows * sizeof(float)) return 0;
    dim3 grid((count + 255u) / 256u, rows);
    dsv41_candidate_blocks_kernel<<<grid, 256>>>(
            (const float *)scores->ptr, (float *)blocks->ptr, width, rows,
            start, ratio);
    return cuda_ok(cudaGetLastError(), "dsv41 candidate blocks launch");
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores,
                                              const ds4_gpu_tensor *block_mask,
                                              uint32_t width, uint32_t rows,
                                              uint32_t start, uint32_t ratio) {
    if (!scores || !block_mask || !width || width > UINT32_MAX - 7u || !rows ||
        !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    if (scores->bytes < (uint64_t)width * rows * sizeof(float) ||
        block_mask->bytes < (uint64_t)blocks * rows * sizeof(float)) return 0;
    dim3 grid((width + 255u) / 256u, rows);
    dsv41_candidate_filter_kernel<<<grid, 256>>>(
            (const float *)scores->ptr, (float *)scores->ptr,
            (const float *)block_mask->ptr, width, rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "dsv41 candidate filter launch");
}

/* -------------------------------------------------------------------------
 * Compact prefill carry (lossless BF16 / mask storage).
 * ---------------------------------------------------------------------- */

__global__ void dsv41_carry_copy_kernel(unsigned *packed, float *plain,
                                        unsigned width, unsigned rows,
                                        unsigned words, unsigned format,
                                        unsigned pack) {
    const unsigned tid = threadIdx.x & 127u;
    const unsigned lane = tid & 31u;
    const unsigned col = blockIdx.x * 128u + tid;
    const size_t row = blockIdx.y;
    if (format == 0u) {
        if (col >= width) return;
        unsigned short *p = (unsigned short *)(packed + row * words);
        if (pack) p[col] = (unsigned short)(__float_as_uint(plain[row * width + col]) >> 16);
        else plain[row * width + col] = __uint_as_float((unsigned)p[col] << 16);
    } else {
        const unsigned word = col / 32u;
        if (pack) {
            const bool allowed = col < width && plain[row * width + col] == 0.0f;
            unsigned bits = allowed ? (1u << lane) : 0u;
            for (int off = 16; off > 0; off >>= 1)
                bits |= __shfl_xor_sync(0xffffffffu, bits, off);
            if (lane == 0 && word < words) packed[row * words + word] = bits;
        } else if (col < width) {
            const unsigned bits = packed[row * words + word];
            plain[row * width + col] =
                (bits & (1u << lane)) ? 0.0f : -INFINITY;
        }
    }
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed,
                                        uint32_t row_offset,
                                        ds4_gpu_tensor *plain, uint32_t width,
                                        uint32_t rows, uint32_t format,
                                        bool pack) {
    if (!packed || !plain || !width || !rows || rows > UINT32_MAX - row_offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16
        ? (uint32_t)(((uint64_t)width + 1u) / 2u)
        : (uint32_t)(((uint64_t)width + 31u) / 32u);
    if (packed->bytes < (uint64_t)(row_offset + rows) * words * sizeof(uint32_t) ||
        plain->bytes < (uint64_t)rows * width * sizeof(float)) return 0;
    dim3 grid((width + 127u) / 128u, rows);
    dsv41_carry_copy_kernel<<<grid, 128>>>(
            (unsigned *)packed->ptr + (size_t)row_offset * words,
            (float *)plain->ptr, width, rows, words, format, pack ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), "dsv41 carry copy launch");
}

/* -------------------------------------------------------------------------
 * Sparse KV gather.
 * ---------------------------------------------------------------------- */

__global__ void dsv41_gather_kv_kernel(float *out, const float *source,
                                       const int *ids, unsigned source_rows,
                                       unsigned selected_rows) {
    const unsigned r = blockIdx.x;
    if (r >= selected_rows) return;
    int id = ids[r];
    if (id < 0 || (unsigned)id >= source_rows) id = 0;
    const float *src = source + (size_t)id * 512u;
    float *dst = out + (size_t)r * 512u;
    for (unsigned j = threadIdx.x; j < 512u; j += blockDim.x)
        dst[j] = src[j];
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out,
                                       const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids,
                                       uint32_t source_rows,
                                       uint32_t selected_rows) {
    if (!out || !source || !ids || !source_rows || !selected_rows ||
        selected_rows > 512u || selected_rows > source_rows ||
        source->bytes < (uint64_t)source_rows * 512u * sizeof(float) ||
        out->bytes < (uint64_t)selected_rows * 512u * sizeof(float) ||
        ids->bytes < (uint64_t)selected_rows * sizeof(int32_t)) return 0;
    dsv41_gather_kv_kernel<<<selected_rows, 256>>>(
            (float *)out->ptr, (const float *)source->ptr,
            (const int *)ids->ptr, source_rows, selected_rows);
    return cuda_ok(cudaGetLastError(), "dsv41 gather kv launch");
}

/* -------------------------------------------------------------------------
 * Indexer and projection delegation.
 * ---------------------------------------------------------------------- */

extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows,
                                                       uint32_t rows) {
    /* The packed tensor-core indexer is Metal-only. The graph always allocates
     * this scratch, so keep a small non-zero placeholder. */
    (void)source_rows; (void)rows;
    return 4096u;
}

extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) {
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed,
                                          const ds4_gpu_tensor *q,
                                          const ds4_gpu_tensor *keys,
                                          uint32_t source_rows, uint32_t rows) {
    (void)packed; (void)q; (void)keys; (void)source_rows; (void)rows;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_packed(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
        const ds4_gpu_tensor *packed, uint32_t source_rows, uint32_t rows,
        uint32_t start, uint32_t ratio, uint32_t packed_rows, uint32_t offset) {
    (void)scores; (void)q; (void)weights; (void)keys; (void)packed;
    (void)source_rows; (void)rows; (void)start; (void)ratio;
    (void)packed_rows; (void)offset;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                                  const ds4_gpu_tensor *q,
                                                  const ds4_gpu_tensor *weights,
                                                  const ds4_gpu_tensor *keys,
                                                  uint32_t source_rows,
                                                  uint32_t rows, uint32_t start,
                                                  uint32_t ratio) {
    /* V4.1 quantizes indexer Q/K to FP4 with a per-block scale, so values can
     * span a huge dynamic range. The WMMA GLM path stages Q/K as fp16 and
     * overflows there, so run the F32 reduction kernel directly. */
    const uint32_t n_head = 32u, head_dim = 128u;
    const float scale = 1.0f / 64.0f;
    if ((ratio != 1u && ratio != 2u) || !scores || !q || !weights || !keys ||
        !source_rows || !rows || start > UINT32_MAX - rows ||
        (uint64_t)(start + rows) / ratio > source_rows ||
        q->bytes < (uint64_t)rows * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)rows * n_head * sizeof(float) ||
        keys->bytes < (uint64_t)source_rows * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)rows * source_rows * sizeof(float)) return 0;
    dim3 grid(source_rows, rows, 1);
    glm_indexer_scores_f32_kernel<float><<<grid, 128>>>(
            (float *)scores->ptr, (const float *)q->ptr,
            (const float *)weights->ptr, (const float *)keys->ptr,
            source_rows, rows, start, n_head, head_dim, ratio, scale, true);
    return cuda_ok(cudaGetLastError(), "dsv41 indexer scores f32 launch");
}

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected,
                                                const ds4_gpu_tensor *scores,
                                                uint32_t width, uint32_t rows,
                                                uint32_t start, uint32_t ratio) {
    /* Each row has its own causal width, and the Metal causal argsort matches
     * a plain top-k over exactly the visible prefix. Reproduce that by running
     * the existing top-k per row over a view of the visible columns. Future
     * entries may hold attractive scores, so the bound cannot be skipped. */
    if ((ratio != 1u && ratio != 2u) || !selected || !scores || !rows ||
        rows > UINT32_MAX - start || width > INT32_MAX || rows > INT32_MAX ||
        (start + rows) / ratio > width || (start + 1u) / ratio < 1024u ||
        selected->bytes < (uint64_t)rows * 512u * sizeof(uint32_t) ||
        scores->bytes < (uint64_t)rows * width * sizeof(float)) return 0;
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t visible = (start + t + 1u) / ratio;
        if (visible < 512u) return 0;
        ds4_gpu_tensor sv = *scores;
        sv.ptr = (char *)scores->ptr + (size_t)t * width * sizeof(float);
        sv.bytes = (uint64_t)visible * sizeof(float);
        sv.owner = 0;
        ds4_gpu_tensor ov = *selected;
        ov.ptr = (char *)selected->ptr + (size_t)t * 512u * sizeof(uint32_t);
        ov.bytes = 512u * sizeof(uint32_t);
        ov.owner = 0;
        if (!ds4_gpu_indexer_topk_tensor(&ov, &sv, visible, 1u, 512u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_projection_rows(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t width, uint32_t outputs, uint32_t rows,
        const ds4_gpu_tensor *in) {
    /* V4.1 requires the batched projection to be bit-exact with the
     * row-at-a-time matvec (the batched cuBLAS path is not). Run the exact
     * n_tok=1 path per row; decode uses rows=1 so this only costs launches
     * during prefill. */
    if (!out || !in || !width || !outputs || !rows || rows > 8192u ||
        !model_map ||
        in->bytes < (uint64_t)rows * width * sizeof(float) ||
        out->bytes < (uint64_t)rows * outputs * sizeof(float)) return 0;
    for (uint32_t t = 0; t < rows; t++) {
        ds4_gpu_tensor iv = *in;
        iv.ptr = (char *)in->ptr + (size_t)t * width * sizeof(float);
        iv.bytes = (uint64_t)width * sizeof(float);
        iv.owner = 0;
        ds4_gpu_tensor ov = *out;
        ov.ptr = (char *)out->ptr + (size_t)t * outputs * sizeof(float);
        ov.bytes = (uint64_t)outputs * sizeof(float);
        ov.owner = 0;
        if (!ds4_gpu_matmul_f16_tensor(&ov, model_map, model_size,
                                       weight_offset, width, outputs, &iv, 1))
            return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    /* Two Q8 projections with a BF16 rounding of the low-rank activation
     * between them. The shared CUDA batch op does not round, so compose the
     * exact row kernel, the V4.1 quantizer, and the output matmul here. */
    const uint64_t group_dim = 4096u, rank = 1024u, out_dim = 5120u;
    const uint32_t n_groups = 8u;
    if (!out || !low || !heads || !model_map || !n_tokens) return 0;
    if (!ds4_gpu_attention_output_low_q8_rows_exact_tensor(
            low, model_map, model_size, out_a_offset, group_dim, rank,
            n_groups, 0u, n_groups, heads, n_tokens)) return 0;
    if (!ds4_gpu_dsv41_quantize(low, (uint32_t)(n_groups * rank), n_tokens,
                                DS4_V41_BF16)) return 0;
    /* Row-at-a-time to stay bit-exact with the reference; the batched Q8 path
     * accumulates differently. Decode uses n_tokens=1. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor lv = *low;
        lv.ptr = (char *)low->ptr + (size_t)t * (n_groups * rank) * sizeof(float);
        lv.bytes = (uint64_t)(n_groups * rank) * sizeof(float);
        lv.owner = 0;
        ds4_gpu_tensor ov = *out;
        ov.ptr = (char *)out->ptr + (size_t)t * out_dim * sizeof(float);
        ov.bytes = out_dim * sizeof(float);
        ov.owner = 0;
        if (!ds4_gpu_matmul_q8_0_tensor(&ov, model_map, model_size,
                                        out_b_offset, n_groups * rank, out_dim,
                                        &lv, 1)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank) {
    /* V4.1 tensor parallelism is not wired for CUDA: the rank-sharded output
     * projection needs a column offset into the Q8 B matrix that the shared
     * attention-output op does not expose. The session keeps tp_world = 1. */
    (void)out; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)heads; (void)n_tokens;
    (void)tp_rank;
    return 0;
}

extern "C" int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scale_scratch,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x,
        uint32_t n_rows, float eps) {
    if (!out || !scale_scratch || !model_map || !x || !in_dim || !out_dim ||
        !n_rows) return 0;
    return ds4_gpu_rms_norm_plain_rows_tensor(scale_scratch, x, in_dim, n_rows,
                                              eps) != 0 &&
           ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset,
                                     in_dim, out_dim, scale_scratch, n_rows) != 0;
}

/* -------------------------------------------------------------------------
 * Streaming / TP shims.
 * ---------------------------------------------------------------------- */

extern "C" int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table, const int32_t *expert_ids,
        const uint32_t *expert_priorities, uint32_t n_experts) {
    return ds4_gpu_stream_expert_cache_seed_experts(table, expert_ids,
                                                    expert_priorities,
                                                    n_experts);
}

extern "C" int ds4_gpu_tp_failed(void) {
    return 0;
}
