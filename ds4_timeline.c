/* ds4_timeline.c — interval lanes for the profiler's token timeline. */
#include "ds4_timeline.h"

#include <stdlib.h>
#include <string.h>

void ds4_tl_init(ds4_tl *tl, uint32_t wall_us) {
    if (!tl) return;
    for (int i = 0; i < DS4_TL_NLANES; i++) tl->lane[i].n = 0;
    tl->wall_us = wall_us;
    tl->v0_us = 0;
    tl->v_us = wall_us;
}

void ds4_tl_set_view(ds4_tl *tl, uint32_t v0_us, uint32_t v_us) {
    if (!tl) return;
    const uint32_t wall = tl->wall_us ? tl->wall_us : 1u;
    if (v_us < 1u) v_us = wall;
    if (v_us > wall) v_us = wall;
    if (v0_us > wall - v_us) v0_us = wall - v_us;
    tl->v0_us = v0_us;
    tl->v_us = v_us;
}

int ds4_tl_push(ds4_tl *tl, int lane, uint32_t t0_us, uint32_t t1_us) {
    if (!tl || lane < 0 || lane >= DS4_TL_NLANES) return 0;
    ds4_tl_lane *l = &tl->lane[lane];
    if (l->n >= DS4_TL_MAX) return 0;
    if (t1_us <= t0_us) t1_us = t0_us + 1u;      /* a mark still paints */
    if (tl->wall_us) {
        if (t0_us >= tl->wall_us) return 0;
        if (t1_us > tl->wall_us) t1_us = tl->wall_us;
    }
    l->span[l->n].t0_us = t0_us;
    l->span[l->n].t1_us = t1_us;
    l->n++;
    return 1;
}

static int span_cmp(const void *a, const void *b) {
    const ds4_tl_span *x = a, *y = b;
    if (x->t0_us != y->t0_us) return x->t0_us < y->t0_us ? -1 : 1;
    if (x->t1_us != y->t1_us) return x->t1_us < y->t1_us ? -1 : 1;
    return 0;
}

void ds4_tl_normalize(ds4_tl *tl) {
    if (!tl) return;
    for (int i = 0; i < DS4_TL_NLANES; i++) {
        ds4_tl_lane *l = &tl->lane[i];
        if (l->n <= 0) continue;
        qsort(l->span, (size_t)l->n, sizeof(l->span[0]), span_cmp);
        /* Merge in place: `out` never passes the read cursor. */
        int out = 0;
        for (int r = 1; r < l->n; r++) {
            if (l->span[r].t0_us <= l->span[out].t1_us) {
                if (l->span[r].t1_us > l->span[out].t1_us)
                    l->span[out].t1_us = l->span[r].t1_us;
            } else {
                l->span[++out] = l->span[r];
            }
        }
        l->n = out + 1;
    }
}

void ds4_tl_carve(ds4_tl *tl, int dst, int src) {
    if (!tl || dst == src || dst < 0 || src < 0) return;
    if (dst >= DS4_TL_NLANES || src >= DS4_TL_NLANES) return;
    ds4_tl_lane *d = &tl->lane[dst];
    const ds4_tl_lane *s = &tl->lane[src];
    if (d->n <= 0 || s->n <= 0) return;
    ds4_tl_span out[DS4_TL_MAX];
    int n = 0;
    for (int i = 0; i < d->n && n < DS4_TL_MAX; i++) {
        uint32_t cur = d->span[i].t0_us;
        const uint32_t end = d->span[i].t1_us;
        for (int j = 0; j < s->n; j++) {
            if (s->span[j].t0_us < cur) continue;
            if (s->span[j].t0_us >= end) break;          /* spans are sorted */
            if (s->span[j].t1_us > end) continue;        /* only containment */
            if (s->span[j].t0_us > cur && n < DS4_TL_MAX) {
                out[n].t0_us = cur;
                out[n].t1_us = s->span[j].t0_us;
                n++;
            }
            if (s->span[j].t1_us > cur) cur = s->span[j].t1_us;
        }
        if (cur < end && n < DS4_TL_MAX) {
            out[n].t0_us = cur;
            out[n].t1_us = end;
            n++;
        }
    }
    memcpy(d->span, out, (size_t)n * sizeof(out[0]));
    d->n = n;
}

int ds4_tl_count(const ds4_tl *tl, int lane) {
    if (!tl || lane < 0 || lane >= DS4_TL_NLANES) return 0;
    return tl->lane[lane].n;
}

double ds4_tl_lane_us(const ds4_tl *tl, int lane) {
    if (!tl || lane < 0 || lane >= DS4_TL_NLANES) return 0.0;
    const ds4_tl_lane *l = &tl->lane[lane];
    double sum = 0.0;
    for (int i = 0; i < l->n; i++) sum += (double)(l->span[i].t1_us - l->span[i].t0_us);
    return sum;
}

double ds4_tl_overlap_us(const ds4_tl *tl, int a, int b) {
    if (!tl || a < 0 || b < 0 || a >= DS4_TL_NLANES || b >= DS4_TL_NLANES) return 0.0;
    const ds4_tl_lane *la = &tl->lane[a], *lb = &tl->lane[b];
    if (a == b) return ds4_tl_lane_us(tl, a);
    int i = 0, j = 0;
    double sum = 0.0;
    while (i < la->n && j < lb->n) {
        const uint32_t lo = la->span[i].t0_us > lb->span[j].t0_us
                          ? la->span[i].t0_us : lb->span[j].t0_us;
        const uint32_t hi = la->span[i].t1_us < lb->span[j].t1_us
                          ? la->span[i].t1_us : lb->span[j].t1_us;
        if (hi > lo) sum += (double)(hi - lo);
        if (la->span[i].t1_us < lb->span[j].t1_us) i++;
        else j++;
    }
    return sum;
}

/* k-way merge over the lanes, advancing the lane whose span starts first, and
 * accumulate the union without materialising it. */
double ds4_tl_union_us(const ds4_tl *tl) {
    if (!tl) return 0.0;
    int idx[DS4_TL_NLANES];
    for (int i = 0; i < DS4_TL_NLANES; i++) idx[i] = 0;
    uint32_t cur0 = 0, cur1 = 0;
    int have = 0;
    double sum = 0.0;
    for (;;) {
        int best = -1;
        uint32_t best_t0 = 0;
        for (int i = 0; i < DS4_TL_NLANES; i++) {
            if (idx[i] >= tl->lane[i].n) continue;
            const uint32_t t0 = tl->lane[i].span[idx[i]].t0_us;
            if (best < 0 || t0 < best_t0) { best = i; best_t0 = t0; }
        }
        if (best < 0) break;
        const ds4_tl_span s = tl->lane[best].span[idx[best]++];
        if (!have) { cur0 = s.t0_us; cur1 = s.t1_us; have = 1; continue; }
        if (s.t0_us <= cur1) {
            if (s.t1_us > cur1) cur1 = s.t1_us;
        } else {
            sum += (double)(cur1 - cur0);
            cur0 = s.t0_us;
            cur1 = s.t1_us;
        }
    }
    if (have) sum += (double)(cur1 - cur0);
    return sum;
}

int ds4_tl_paint(const ds4_tl *tl, int lane, int cols, unsigned char *level) {
    if (!level || cols <= 0) return 0;
    memset(level, 0, (size_t)cols);
    if (!tl || lane < 0 || lane >= DS4_TL_NLANES) return 0;
    const ds4_tl_lane *l = &tl->lane[lane];
    if (l->n <= 0) return 0;
    const uint32_t wall = tl->wall_us ? tl->wall_us : l->span[l->n - 1].t1_us;
    if (!wall) return 0;
    const uint32_t v0 = tl->v0_us;
    const uint32_t v = tl->v_us ? tl->v_us : wall;
    /* Coverage per column: the share of the column's own time the lane was
     * busy, in 1/64ths. Spans in a lane are merged first, so their coverage
     * never overlaps and the sum below cannot count the same instant twice. */
    uint32_t cov[512];
    if (cols > 512) cols = 512;
    memset(cov, 0, sizeof(cov[0]) * (size_t)cols);
    const double colw = (double)v / (double)cols;
    for (int i = 0; i < l->n; i++) {
        const double a = (double)l->span[i].t0_us - (double)v0;
        const double b = (double)l->span[i].t1_us - (double)v0;
        if (b <= 0.0 || a >= (double)v) continue;
        int c0 = (int)((a > 0.0 ? a : 0.0) / colw);
        int c1 = (int)((b < (double)v ? b : (double)v) / colw);
        if (c0 < 0) c0 = 0;
        if (c1 >= cols) c1 = cols - 1;
        for (int c = c0; c <= c1; c++) {
            const double cs = (double)c * colw;
            const double ce = cs + colw;
            const double lo = a > cs ? a : cs;
            const double hi = b < ce ? b : ce;
            if (hi <= lo) continue;
            uint32_t add = (uint32_t)((hi - lo) * 64.0 / colw + 0.5);
            cov[c] += add ? add : 1u;
            if (cov[c] > 64u) cov[c] = 64u;
        }
    }
    int painted = 0;
    for (int c = 0; c < cols; c++) {
        if (!cov[c]) continue;
        level[c] = (unsigned char)(cov[c] >= 48u ? 3 : cov[c] >= 16u ? 2 : 1);
        painted++;
    }
    return painted;
}
