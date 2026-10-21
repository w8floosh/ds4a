/* tests/test_tui.c — render the three-pane UI with the synthetic feed.
 *
 * Needs a terminal; run it through script(1) or a pty harness:
 *   DS4_TUI_FAKE=1 script -qec ./tests/test_tui /dev/null
 * Exits 0 when the UI can be created and drawn, or when stdin is not a tty
 * (the engine then falls back to the line REPL). */
#include "ds4_tui.h"
#include "ds4_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    ds4_tui *t = ds4_tui_create();
    if (!t) {
        printf("test_tui: no terminal, UI disabled\n");
        return 0;
    }
    if (getenv("DS4_TUI_FAKE_BP")) {
        ds4_tui_set_breakpoints(t, 1);
        ds4_tui_set_busy(t, 1, "synthetic");
        ds4_tui_breakpoint(t, "synthetic token");
    }
    ds4_tui_set_busy(t, 1, "synthetic");
    int iters = 60;
    if (getenv("DS4_TUI_FAKE_ITERS")) {
        int v = atoi(getenv("DS4_TUI_FAKE_ITERS"));
        if (v > 0) iters = v;
    }
    const int stream = getenv("DS4_TUI_FAKE_STREAM") != NULL;
    for (int i = 0; i < iters; i++) {
        usleep(50000);
        if (stream) {
            /* Decode-shaped traffic: a few characters per token at ~20 tok/s. */
            ds4_tui_chat_append(t, "streamed text ", strlen("streamed text "), 0);
            if ((i % 10) == 9) ds4_tui_chat_newline(t);
        }
        ds4_tui_pump(t);
        if (getenv("DS4_TUI_FAKE_EXPIRE") && i == 30) {
            for (int k = 0; k < 300; k++) {
                if (!ds4_profiler_token_begin_ex(DS4_PREC_DECODE_TOKEN, 200 + k, k, 1))
                    break;
                ds4_profiler_note_stage("decode", "routed_moe", 1, k, 1,
                                        1000000, 100000, 900000, 0);
                ds4_profiler_token_end(1);
            }
        }
    }
    ds4_tui_redraw(t);
    ds4_tui_set_busy(t, 0, NULL);
    ds4_tui_pump(t);
    ds4_tui_destroy(t);
    printf("test_tui: ok\n");
    return 0;
}
