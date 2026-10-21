/* ds4_timeline.h — interval lanes behind the profiler's token timeline.
 *
 * A record's events are points and spans on one axis (microseconds from the
 * record start). This module turns them into per-lane intervals, merges them,
 * and reports how much of the record each lane covered and how much the lanes
 * overlapped. It is pure arithmetic: no terminal, no profiler globals, so the
 * math is unit-tested on its own.
 *
 * Lanes are the questions a reader asks of a token: what the CPU did, what each
 * device did, what was in flight, and what the host waited for. A span lands in
 * exactly one lane; two lanes covering the same instant is the overlap the pane
 * exists to show.
 */
#ifndef DS4_TIMELINE_H
#define DS4_TIMELINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One record is ~3k events, and a stage event splits into a launch and a body
 * span, so a decode token lands under 8k spans. Prefill chunks are coarser. */
#define DS4_TL_MAX 8192

enum {
    DS4_TL_HOST = 0,   /* CPU encoding a stage (the launch column)          */
    DS4_TL_DEV0,       /* primary device busy (host-observed sync window)   */
    DS4_TL_DEV1,       /* cache/tier device busy                            */
    DS4_TL_XFER,       /* bytes in flight: expert uploads, peer copies, I/O  */
    DS4_TL_WAIT,       /* host blocked: stream sync, upload wait, readback  */
    DS4_TL_NLANES
};

typedef struct {
    uint32_t t0_us, t1_us;
} ds4_tl_span;

typedef struct {
    ds4_tl_span span[DS4_TL_MAX];
    int         n;
} ds4_tl_lane;

typedef struct {
    ds4_tl_lane lane[DS4_TL_NLANES];
    uint32_t   wall_us;      /* the record's own wall, the axis' right edge */
    /* The drawn window. The whole record at 40 columns cannot show a 20 us
     * launch, so the overlay zooms: the view is what paint() maps to columns. */
    uint32_t   v0_us, v_us;
} ds4_tl;

/* Empty every lane. `wall_us` 0 means "unknown": spans are then not clipped. */
void ds4_tl_init(ds4_tl *tl, uint32_t wall_us);

/* Add a span. Times are clipped to the wall and t1 is forced past t0, so a
 * zero-length mark still paints one column. Returns 0 when the lane is full or
 * the span falls entirely outside the wall. */
int  ds4_tl_push(ds4_tl *tl, int lane, uint32_t t0_us, uint32_t t1_us);

/* Sort every lane and merge overlapping spans, so a lane holds a union and its
 * total is the time that lane was busy, not the sum of its calls. Called once
 * after the last push. */
void ds4_tl_normalize(ds4_tl *tl);

/* Remove from `dst` every part that `src` also covers and that lies inside a
 * dst span, because the two are the same physical time seen twice: a stage's
 * CPU window contains the read that blocked it, a device window contains the
 * stream sync that ended it. Spans that merely overlap — a transfer running
 * while the device computes — are left alone, since those are two resources
 * busy at once, which is the overlap the pane exists to show. */
void ds4_tl_carve(ds4_tl *tl, int dst, int src);

int    ds4_tl_count(const ds4_tl *tl, int lane);
double ds4_tl_lane_us(const ds4_tl *tl, int lane);

/* Time covered by both lanes at once (0 when either is empty). */
double ds4_tl_overlap_us(const ds4_tl *tl, int a, int b);

/* Time covered by at least one lane: the union of all measured activity. The
 * record wall minus this is time the profiler cannot attribute. */
double ds4_tl_union_us(const ds4_tl *tl);

/* Zoom the drawn window to [v0_us, v0_us + v_us). The window is clamped to the
 * record: `v_us` 0 or a window past the wall falls back to the whole record. */
void ds4_tl_set_view(ds4_tl *tl, uint32_t v0_us, uint32_t v_us);

/* Fill `level` with one level per column over the drawn window: 0 for a
 * column the lane does not touch, 1 to 3 by how much of it the lane covers.
 * A token has thousands of 2 us launches, so a level of presence alone would
 * claim a duty cycle the lane does not have. Returns the number of columns with
 * a level above 0; adjacent spans that meet inside one column count once. The
 * module stays text-free: the caller picks the glyph for each level. */
int  ds4_tl_paint(const ds4_tl *tl, int lane, int cols, unsigned char *level);

#ifdef __cplusplus
}
#endif

#endif /* DS4_TIMELINE_H */
