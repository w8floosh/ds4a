/*
 * ds4-requant: re-encode selected tensors of an existing DS4 GGUF without the
 * original Hugging Face safetensors.
 *
 * deepseek4-quantize regenerates every tensor from the HF source. This tool is
 * for the common case where the weights already exist in a GGUF and only a few
 * tensor families should change precision (for example token_embd.weight
 * F16 -> Q8_0 to free VRAM at long context). All other tensors are copied
 * verbatim, so no HF download is needed.
 *
 * Supported source types for a re-encoded tensor: F32, F16, Q8_0. Supported
 * targets are the quants.c types that do not need an importance matrix
 * (q8_0, q8_K, q4_K, q2_K). iq2_xxs needs an imatrix, which this tool does
 * not accept, and is rejected rather than quantized with a NULL imatrix.
 *
 * usage:
 *   ds4-requant --in SRC.gguf --out DST.gguf --set token_embd.weight=q8_0
 *   ds4-requant --in SRC.gguf --list
 *   ds4-requant --in SRC.gguf --out DST.gguf --set NAME=q8_0 --dry-run
 */

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quants.h"

/* GGUF stores every integer little-endian; these helpers read/write host
 * words directly. Fail the build rather than silently corrupt on a big-endian
 * host. */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "ds4-requant assumes a little-endian host (GGUF is little-endian)"
#endif

typedef struct {
    char    *name;
    uint32_t nd;
    uint64_t ne[4];
    uint32_t type;
    uint64_t src_off;
    uint64_t dst_off;
    size_t   src_size;
    size_t   dst_size;
    uint32_t dst_type;
} tensor_rec;

typedef struct {
    tensor_rec *t;
    uint32_t    n;
} set_list;

static void die(const char *msg) {
    fprintf(stderr, "ds4-requant: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4-requant: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

static uint32_t rd_u32(FILE *fp) {
    uint32_t v;
    if (fread(&v, 4, 1, fp) != 1) die("short read (u32)");
    return v;
}
static uint64_t rd_u64(FILE *fp) {
    uint64_t v;
    if (fread(&v, 8, 1, fp) != 1) die("short read (u64)");
    return v;
}
static char *rd_str(FILE *fp, uint64_t *len_out) {
    const uint64_t n = rd_u64(fp);
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) die("out of memory (string)");
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short read (string)");
    s[n] = '\0';
    if (len_out) *len_out = n;
    return s;
}

/* Skip one GGUF metadata value of the given type. */
static void skip_val(FILE *fp, uint32_t type) {
    switch (type) {
    case 0: case 1: case 7: { uint8_t x; if (fread(&x,1,1,fp)!=1) die("short read"); break; }
    case 2: case 3: { uint16_t x; if (fread(&x,2,1,fp)!=1) die("short read"); break; }
    case 4: case 5: case 6: { uint32_t x; if (fread(&x,4,1,fp)!=1) die("short read"); break; }
    case 8: { uint64_t n = rd_u64(fp); if (fseeko(fp,(off_t)n,SEEK_CUR)!=0) die("short read (str)"); break; }
    case 9: {
        const uint32_t et = rd_u32(fp);
        const uint64_t n = rd_u64(fp);
        for (uint64_t i = 0; i < n; i++) skip_val(fp, et);
        break;
    }
    case 10: case 11: case 12: { uint64_t x; if (fread(&x,8,1,fp)!=1) die("short read"); break; }
    default: die("unknown GGUF metadata type");
    }
}

static uint32_t type_from_name(const char *name) {
    for (uint32_t t = 0; t < DS4Q_TYPE_COUNT; t++) {
        const char *n = ds4q_type_name((ds4q_type)t);
        if (n && strcmp(n, name) == 0) return t;
    }
    return DS4Q_TYPE_COUNT;
}

static void tensor_dims(uint64_t ne[4], uint32_t nd, uint64_t *elems) {
    uint64_t e = 1;
    for (uint32_t i = 0; i < nd; i++) { ne[i] = ne[i] ? ne[i] : 1; e *= ne[i]; }
    for (uint32_t i = nd; i < 4; i++) ne[i] = 1;
    *elems = e;
}

static size_t row_bytes_for(uint32_t type, uint64_t ne0) {
    return ds4q_row_size((ds4q_type)type, (int64_t)ne0);
}

/* ---------------- tensor transforms ---------------- */

/* Dequantize one row-batch (nrows rows of ncols elements) of type `type` into
 * a float buffer already sized nrows*ncols. Only F32/F16/Q8_0 are supported,
 * which is everything the DS4 non-routed families use. */
static void dequant_rows(uint32_t type, const uint8_t *src, float *dst,
                         uint64_t nrows, uint64_t ncols) {
    const uint64_t n = nrows * ncols;
    if (type == 0) { /* F32 */
        memcpy(dst, src, (size_t)n * 4);
        return;
    }
    if (type == 1) { /* F16 */
        const uint16_t *h = (const uint16_t *)src;
        for (uint64_t i = 0; i < n; i++) dst[i] = ds4q_f16_to_f32(h[i]);
        return;
    }
    if (type == 8) { /* Q8_0: 2-byte half scale + 32 int8 per block */
        const uint64_t nb = ncols / 32u;
        const uint8_t *p = src;
        for (uint64_t r = 0; r < nrows; r++) {
            for (uint64_t b = 0; b < nb; b++, p += 34) {
                const float d = ds4q_f16_to_f32(*(const uint16_t *)p);
                const int8_t *q = (const int8_t *)(p + 2);
                for (uint32_t i = 0; i < 32; i++) {
                    dst[r * ncols + b * 32u + i] = d * (float)q[i];
                }
            }
        }
        return;
    }
    die("unsupported source tensor type for re-encoding");
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL;
    int list_only = 0, dry_run = 0;
    tensor_rec *recs = NULL;
    uint32_t n_recs = 0;
    char **set_name = NULL; uint32_t *set_type = NULL; uint32_t n_set = 0, cap_set = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--list")) list_only = 1;
        else if (!strcmp(argv[i], "--dry-run")) dry_run = 1;
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            const char *kv = argv[++i];
            const char *eq = strrchr(kv, '=');
            if (!eq) die("--set expects NAME=TYPE");
            char name[512];
            const size_t nl = (size_t)(eq - kv);
            if (nl >= sizeof(name)) die("tensor name too long");
            memcpy(name, kv, nl); name[nl] = '\0';
            const uint32_t t = type_from_name(eq + 1);
            if (t >= DS4Q_TYPE_COUNT) die("unknown target type");
            if (n_set == cap_set) {
                cap_set = cap_set ? cap_set * 2 : 8;
                set_name = realloc(set_name, cap_set * sizeof(char *));
                set_type = realloc(set_type, cap_set * sizeof(uint32_t));
                if (!set_name || !set_type) die("out of memory");
            }
            set_name[n_set] = strdup(name);
            set_type[n_set] = t;
            n_set++;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("usage: %s --in SRC.gguf --out DST.gguf --set NAME=TYPE [--set ...] [--dry-run]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (!in_path) die("missing --in");

    FILE *fp = fopen(in_path, "rb");
    if (!fp) die_errno("open", in_path);
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "GGUF", 4) != 0) die("not a GGUF file");
    const uint32_t version = rd_u32(fp);
    if (version != 3) die("only GGUF v3 is supported");
    const uint64_t n_tensors = rd_u64(fp);
    const uint64_t n_kv = rd_u64(fp);
    if (n_tensors > 100000) die("implausible tensor count");

    const off_t kv_start = ftello(fp);
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rd_str(fp, NULL);
        const uint32_t type = rd_u32(fp);
        if (!strcmp(key, "general.alignment") && type == 4) {
            uint32_t a; if (fread(&a, 4, 1, fp) != 1) die("short read"); alignment = a ? a : 32;
            free(key);
            continue;
        }
        free(key);
        skip_val(fp, type);
    }
    const off_t kv_end = ftello(fp);
    if (alignment == 0) alignment = 32;

    recs = calloc((size_t)n_tensors, sizeof(*recs));
    if (!recs) die("out of memory (tensors)");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_rec *r = &recs[i];
        r->name = rd_str(fp, NULL);
        r->nd = rd_u32(fp);
        if (r->nd == 0 || r->nd > 4) die("bad tensor rank");
        for (uint32_t d = 0; d < r->nd; d++) r->ne[d] = rd_u64(fp);
        r->type = rd_u32(fp);
        r->src_off = rd_u64(fp);
        uint64_t elems = 0;
        uint64_t ne[4];
        memcpy(ne, r->ne, sizeof(ne));
        tensor_dims(ne, r->nd, &elems);
        memcpy(r->ne, ne, sizeof(ne));
        const size_t rb = row_bytes_for(r->type, r->ne[0]);
        if (rb == 0) die("unsupported source tensor type in file");
        r->src_size = rb * (elems / r->ne[0]);
        r->dst_type = r->type;
        r->dst_size = r->src_size;
    }
    n_recs = (uint32_t)n_tensors;
    const off_t infos_end = ftello(fp);
    const off_t src_data = (off_t)ds4q_pad((size_t)infos_end, (size_t)alignment);

    if (list_only) {
        for (uint32_t i = 0; i < n_recs; i++) {
            printf("%-52s %-8s %10.2f MiB\n", recs[i].name,
                   ds4q_type_name((ds4q_type)recs[i].type),
                   (double)recs[i].src_size / 1048576.0);
        }
        return 0;
    }
    if (!out_path) die("missing --out");

    /* Apply the requested type changes and recompute sizes/offsets. */
    for (uint32_t s = 0; s < n_set; s++) {
        int found = 0;
        for (uint32_t i = 0; i < n_recs; i++) {
            if (strcmp(recs[i].name, set_name[s]) != 0) continue;
            found = 1;
            if (!ds4q_can_quantize((ds4q_type)set_type[s])) {
                fprintf(stderr, "ds4-requant: target type %s is not implemented by quants.c\n",
                        ds4q_type_name((ds4q_type)set_type[s]));
                return 1;
            }
            if (ds4q_requires_imatrix((ds4q_type)set_type[s])) {
                fprintf(stderr,
                        "ds4-requant: target type %s needs an importance matrix, "
                        "which this tool does not accept\n",
                        ds4q_type_name((ds4q_type)set_type[s]));
                return 1;
            }
            recs[i].dst_type = set_type[s];
        }
        if (!found) {
            fprintf(stderr, "ds4-requant: tensor not found: %s\n", set_name[s]);
            return 1;
        }
    }

    for (uint32_t i = 0; i < n_recs; i++) {
        tensor_rec *r = &recs[i];
        uint64_t elems = r->ne[0] * r->ne[1] * r->ne[2] * r->ne[3];
        const size_t rb = row_bytes_for(r->dst_type, r->ne[0]);
        if (rb == 0) die("unsupported target tensor type/shape");
        r->dst_size = rb * (elems / r->ne[0]);
    }

    /* Serialize tensor infos and assign output offsets. */
    size_t names_len = 0;
    for (uint32_t i = 0; i < n_recs; i++) names_len += strlen(recs[i].name) + 8 + 4 + 8 * recs[i].nd + 4 + 8;
    uint8_t *infos = malloc(names_len + 64);
    if (!infos) die("out of memory (infos)");
    size_t ip = 0;
    #define PUT(p, n) do { memcpy(infos + ip, (p), (n)); ip += (n); } while (0)
    uint64_t off = 0;
    for (uint32_t i = 0; i < n_recs; i++) {
        tensor_rec *r = &recs[i];
        r->dst_off = off;
        off += ds4q_pad(r->dst_size, (size_t)alignment);
        const uint64_t nl = strlen(r->name);
        PUT(&nl, 8);
        PUT(r->name, nl);
        PUT(&r->nd, 4);
        for (uint32_t d = 0; d < r->nd; d++) PUT(&r->ne[d], 8);
        PUT(&r->dst_type, 4);
        PUT(&r->dst_off, 8);
    }
    const uint64_t data_total = off;

    fprintf(stderr, "ds4-requant: %u tensors, %u re-encoded, data %.2f GiB\n",
            n_recs, n_set, (double)data_total / 1073741824.0);
    for (uint32_t s = 0; s < n_set; s++) {
        fprintf(stderr, "  %s -> %s\n", set_name[s], ds4q_type_name((ds4q_type)set_type[s]));
    }
    if (dry_run) return 0;

    /* Validate every source type we will dequantize before creating the output,
     * so an unsupported tensor fails cleanly instead of leaving a truncated
     * file behind (dequant_rows would otherwise die mid-write). */
    for (uint32_t i = 0; i < n_recs; i++) {
        const tensor_rec *r = &recs[i];
        if (r->dst_type == r->type) continue;
        if (r->type != 0 && r->type != 1 && r->type != 8) {
            fprintf(stderr,
                    "ds4-requant: cannot re-encode %s from source type %s "
                    "(only F32/F16/Q8_0 sources are supported)\n",
                    r->name, ds4q_type_name((ds4q_type)r->type));
            return 1;
        }
    }

    FILE *out = fopen(out_path, "wb");
    if (!out) die_errno("create", out_path);

    /* header: magic, version, n_tensors, n_kv */
    fwrite("GGUF", 1, 4, out);
    fwrite(&version, 4, 1, out);
    fwrite(&n_tensors, 8, 1, out);
    fwrite(&n_kv, 8, 1, out);

    /* copy raw KV block verbatim */
    const size_t kv_len = (size_t)(kv_end - kv_start);
    uint8_t *kv_raw = malloc(kv_len);
    if (!kv_raw) die("out of memory (kv)");
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die_errno("seek", in_path);
    if (fread(kv_raw, 1, kv_len, fp) != kv_len) die("short read (kv block)");
    fwrite(kv_raw, 1, kv_len, out);

    fwrite(infos, 1, ip, out);

    const off_t out_infos_end = (off_t)(4 + 4 + 8 + 8 + kv_len + ip);
    const off_t out_data = (off_t)ds4q_pad((size_t)out_infos_end, (size_t)alignment);
    for (off_t p = out_infos_end; p < out_data; p++) fputc(0, out);

    /* data: copy or re-encode, padding each tensor to alignment */
    const size_t CHUNK = 8u * 1024u * 1024u;
    uint8_t *rbuf = malloc(CHUNK);
    if (!rbuf) die("out of memory (rbuf)");
    for (uint32_t i = 0; i < n_recs; i++) {
        tensor_rec *r = &recs[i];
        if (fseeko(fp, src_data + (off_t)r->src_off, SEEK_SET) != 0) die_errno("seek", in_path);
        if (r->dst_type == r->type) {
            size_t left = r->src_size;
            while (left) {
                const size_t n = left < CHUNK ? left : CHUNK;
                if (fread(rbuf, 1, n, fp) != n) die("short read (tensor)");
                if (fwrite(rbuf, 1, n, out) != n) die_errno("write tensor", out_path);
                left -= n;
            }
        } else {
            /* Re-encode row batches. */
            const uint64_t ncols = r->ne[0];
            const uint64_t total_rows = (r->ne[1] * r->ne[2] * r->ne[3]);
            const uint64_t src_row = row_bytes_for(r->type, ncols);
            const uint64_t dst_row = row_bytes_for(r->dst_type, ncols);
            uint64_t batch = 4096;
            if (batch > total_rows) batch = total_rows;
            float *fbuf = malloc((size_t)(batch * ncols) * sizeof(float));
            uint8_t *sbuf = malloc((size_t)(batch * src_row));
            uint8_t *dbuf = malloc((size_t)(batch * dst_row));
            if (!fbuf || !sbuf || !dbuf) die("out of memory (batch)");
            for (uint64_t r0 = 0; r0 < total_rows; r0 += batch) {
                uint64_t nr = total_rows - r0 < batch ? total_rows - r0 : batch;
                if (fread(sbuf, 1, (size_t)(nr * src_row), fp) != nr * src_row) die("short read (tensor)");
                dequant_rows(r->type, sbuf, fbuf, nr, ncols);
                const size_t wr = ds4q_quantize_chunk((ds4q_type)r->dst_type, fbuf, dbuf,
                                                      0, (int64_t)nr, (int64_t)ncols, NULL);
                if (wr != (size_t)(nr * dst_row)) die("quantized size mismatch");
                if (fwrite(dbuf, 1, wr, out) != wr) die_errno("write tensor", out_path);
            }
            free(fbuf); free(sbuf); free(dbuf);
        }
        /* pad to alignment */
        size_t padded = ds4q_pad(r->dst_size, (size_t)alignment);
        for (size_t p = r->dst_size; p < padded; p++) fputc(0, out);
    }

    free(rbuf);
    free(kv_raw);
    free(infos);
    if (fclose(out) != 0) die_errno("close", out_path);
    fclose(fp);
    fprintf(stderr, "ds4-requant: wrote %s\n", out_path);
    return 0;
}
