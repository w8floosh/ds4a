/* tests/test_timeline.c — the token timeline's arithmetic.
 *
 * The lanes are the only place where a profiler record's events are turned
 * into time intervals, so the clipping, merging, overlap and column mapping are
 * checked here rather than by reading a screen: a wrong overlap number would
 * look exactly like a slow token. */
#include "ds4_timeline.h"

#include <stdio.h>
#include <string.h>

static int fails;

static void check(int cond, const char *what) {
    if (!cond) {
        printf("FAIL: %s\n", what);
        fails++;
    }
}

static int near(double a, double b) {
    const double d = a > b ? a - b : b - a;
    return d < 0.51;
}

int main(void) {
    /* Push clips to the record wall, keeps a zero-length mark visible, and
     * refuses a span that starts past the wall. */
    {
        ds4_tl tl;
        ds4_tl_init(&tl, 1000);
        check(ds4_tl_push(&tl, DS4_TL_HOST, 0, 100) == 1, "span inside wall");
        check(ds4_tl_push(&tl, DS4_TL_HOST, 900, 1200) == 1, "span over the wall");
        check(ds4_tl_push(&tl, DS4_TL_HOST, 2000, 2100) == 0,
              "span past the wall dropped");
        check(ds4_tl_push(&tl, DS4_TL_HOST, 500, 500) == 1, "zero-length mark kept");
        ds4_tl_normalize(&tl);
        check(ds4_tl_count(&tl, DS4_TL_HOST) == 3, "three spans survived");
        check(near(ds4_tl_lane_us(&tl, DS4_TL_HOST), 100 + 100 + 1),
              "clipped to the wall, mark is one microsecond");
    }

    /* Overlapping spans merge, so a lane reports busy time, not call time. */
    {
        ds4_tl tl;
        ds4_tl_init(&tl, 100);
        ds4_tl_push(&tl, DS4_TL_XFER, 10, 20);
        ds4_tl_push(&tl, DS4_TL_XFER, 15, 30);
        ds4_tl_push(&tl, DS4_TL_XFER, 30, 40);
        ds4_tl_push(&tl, DS4_TL_XFER, 50, 55);
        ds4_tl_normalize(&tl);
        check(ds4_tl_count(&tl, DS4_TL_XFER) == 2, "two running stretches");
        check(near(ds4_tl_lane_us(&tl, DS4_TL_XFER), 35), "touching spans merged");
    }

    /* Overlap and union: the numbers the pane prints as "xfer/gpu0 30.0". */
    {
        ds4_tl tl;
        ds4_tl_init(&tl, 1000);
        ds4_tl_push(&tl, DS4_TL_XFER, 0, 10);
        ds4_tl_push(&tl, DS4_TL_XFER, 20, 30);
        ds4_tl_push(&tl, DS4_TL_DEV0, 5, 15);
        ds4_tl_push(&tl, DS4_TL_DEV0, 22, 28);
        ds4_tl_normalize(&tl);
        check(near(ds4_tl_overlap_us(&tl, DS4_TL_XFER, DS4_TL_DEV0), 11),
              "overlap counts both shared stretches");
        check(near(ds4_tl_overlap_us(&tl, DS4_TL_DEV0, DS4_TL_XFER), 11),
              "overlap is symmetric");
        check(near(ds4_tl_union_us(&tl), 25), "union does not double count");
        check(near(ds4_tl_overlap_us(&tl, DS4_TL_WAIT, DS4_TL_DEV0), 0),
              "empty lane overlaps nothing");
        check(near(ds4_tl_overlap_us(&tl, DS4_TL_DEV0, DS4_TL_DEV0), 16),
              "a lane against itself is its total");
    }

    /* Columns: duty decides the glyph, so 3000 short launches do not paint a
     * solid bar, and adjacent spans meeting inside one column paint it once. */
    {
        ds4_tl tl;
        ds4_tl_init(&tl, 1000);
        ds4_tl_push(&tl, DS4_TL_HOST, 0, 5);            /* 0.5% of the token */
        ds4_tl_push(&tl, DS4_TL_HOST, 500, 1000);       /* half of it        */
        ds4_tl_normalize(&tl);
        unsigned char out[64];
        int painted = ds4_tl_paint(&tl, DS4_TL_HOST, 20, out);
        check(painted > 0, "painted something");
        check(out[0] == 1, "a sliver is the light level");
        check(out[19] == 3, "a full column is the heavy level");
        check(out[5] == 0, "an untouched column has no level");

        ds4_tl back;
        ds4_tl_init(&back, 1000);
        ds4_tl_push(&back, DS4_TL_WAIT, 0, 500);
        ds4_tl_push(&back, DS4_TL_WAIT, 500, 1000);
        ds4_tl_normalize(&back);
        ds4_tl_paint(&back, DS4_TL_WAIT, 10, out);
        int full = 0;
        for (int i = 0; i < 10; i++) if (out[i] == 3) full++;
        check(full == 10, "a lane busy end to end is solid, no seam");
        check(ds4_tl_paint(&back, DS4_TL_HOST, 10, out) == 0,
              "empty lane paints nothing");
    }

    /* Zoom: the window decides both the axis and the columns, and is clamped
     * to the record so a pan can never scroll past its edge. */
    {
        ds4_tl tl;
        ds4_tl_init(&tl, 1000);
        ds4_tl_push(&tl, DS4_TL_DEV0, 0, 200);
        ds4_tl_push(&tl, DS4_TL_DEV0, 800, 1000);
        ds4_tl_normalize(&tl);
        unsigned char out[64];
        check(ds4_tl_paint(&tl, DS4_TL_DEV0, 10, out) == 4,
              "whole token shows both stretches");
        check(out[0] == 3 && out[9] == 3 && out[5] == 0,
              "both ends busy, the middle idle");
        ds4_tl_set_view(&tl, 700, 300);
        ds4_tl_paint(&tl, DS4_TL_DEV0, 10, out);
        check(out[9] == 3, "zoomed window shows the end stretch only");
        check(out[0] == 0, "zoomed window drops what is left of it");
        ds4_tl_set_view(&tl, 5000, 100);
        check(tl.v0_us == 900 && tl.v_us == 100, "pan is clamped to the record");
        ds4_tl_set_view(&tl, 0, 0);
        check(tl.v0_us == 0 && tl.v_us == 1000, "a zero window means the record");
    }

    /* Carving: a nested span is the same second seen twice, so it is removed
     * from the outer lane; a span that only straddles the edge is not. */
    {
        ds4_tl tl;
        ds4_tl_init(&tl, 200);
        ds4_tl_push(&tl, DS4_TL_HOST, 0, 100);
        ds4_tl_push(&tl, DS4_TL_HOST, 120, 200);
        ds4_tl_push(&tl, DS4_TL_XFER, 10, 20);
        ds4_tl_push(&tl, DS4_TL_XFER, 30, 40);
        ds4_tl_push(&tl, DS4_TL_XFER, 110, 130);   /* straddles 120 */
        ds4_tl_normalize(&tl);
        check(near(ds4_tl_lane_us(&tl, DS4_TL_HOST), 180),
              "host total before carving");
        ds4_tl_carve(&tl, DS4_TL_HOST, DS4_TL_XFER);
        check(ds4_tl_count(&tl, DS4_TL_HOST) == 4, "carved into four pieces");
        check(near(ds4_tl_lane_us(&tl, DS4_TL_HOST), 160),
              "the two nested reads came off the host lane");
        check(near(ds4_tl_lane_us(&tl, DS4_TL_XFER), 40),
              "the transfer lane keeps its own time");
        check(near(ds4_tl_overlap_us(&tl, DS4_TL_HOST, DS4_TL_XFER), 10),
              "only the straddling read still overlaps");
        ds4_tl_normalize(&tl);
        check(near(ds4_tl_lane_us(&tl, DS4_TL_HOST), 160),
              "carved pieces merge back without inventing time");
    }

    if (fails) {
        printf("test_timeline: %d failure(s)\n", fails);
        return 1;
    }
    printf("test_timeline: ok\n");
    return 0;
}
