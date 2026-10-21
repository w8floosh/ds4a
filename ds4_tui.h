/* ds4_tui.h — four-pane terminal UI: chat, live profiler, console and help.
 *
 * The UI is a renderer over ds4_profiler snapshots plus a chat transcript
 * buffer. It owns the terminal (raw mode, alternate screen) and captures
 * stderr into the console pane so engine warnings can never be mistaken for
 * model output; the help pane next to it explains whatever the profiler shows.
 *
 * Generation stays synchronous: the frontend calls ds4_tui_pump() from token
 * and prefill callbacks. pump() reads keys non-blocking, drains stderr and
 * redraws at a bounded rate. Only the UI thread touches these buffers.
 */
#ifndef DS4_TUI_H
#define DS4_TUI_H

#include <stddef.h>
#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_tui ds4_tui;

typedef enum {
    DS4_TUI_CHAT = 0,
    DS4_TUI_PROFILER,
    DS4_TUI_MESSAGES,
    DS4_TUI_HELP,
    DS4_TUI_PANE_COUNT
} ds4_tui_focus;

/* Create the UI only when both stdin and stdout are terminals. Returns NULL
 * when it cannot or when DS4_TUI=0. stderr is redirected into the console
 * pane until destroy. */
ds4_tui *ds4_tui_create(void);
void     ds4_tui_destroy(ds4_tui *t);

int      ds4_tui_active(const ds4_tui *t);

/* Chat transcript. role: 0 user, 1 assistant, 2 system/notice. `think` dims
 * assistant text produced inside a reasoning block. */
void ds4_tui_chat_role(ds4_tui *t, int role, const char *label);
void ds4_tui_chat_append(ds4_tui *t, const char *text, size_t len, int think);
void ds4_tui_chat_newline(ds4_tui *t);

/* Console pane: engine stderr and UI notices. */
void ds4_tui_log(ds4_tui *t, int level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;
void ds4_tui_status(ds4_tui *t, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* Mark generation busy/idle. While busy the chat input is disabled and the
 * status line explains how to stop. */
void ds4_tui_set_busy(ds4_tui *t, int busy, const char *what);

/* Process keys, drain stderr, redraw if needed. Safe to call from callbacks. */
void ds4_tui_pump(ds4_tui *t);
void ds4_tui_redraw(ds4_tui *t);

/* Read one line with editing/history. Returns a malloc'd line, or NULL on
 * EOF/Ctrl-D. Focus starts on the chat pane. Pumping continues while waiting,
 * so the profiler stays live and scrollable. */
char *ds4_tui_readline(ds4_tui *t, const char *prompt);

/* Bubble the terminal's SIGINT to the UI so a pending line is cleared and the
 * status reflects cancellation. */
void ds4_tui_set_interrupt(ds4_tui *t,
                           int (*requested)(void *), void (*clear)(void *),
                           void *ud);

/* Enable records/stage capture (default on) and optionally a synthetic feed
 * for smoke tests. DS4_TUI_FAKE=1 in the environment enables the feed. */
void ds4_tui_set_granular(ds4_tui *t, int on);
int  ds4_tui_granular(const ds4_tui *t);

/* Breakpoint mode: when enabled, ds4_tui_breakpoint() freezes the engine at
 * the current token/chunk boundary and keeps the UI live so the developer can
 * inspect records, scroll and open detail pages. Space continues, b disables
 * and continues, Ctrl-C keeps the interrupt pending so generation stops. */
void ds4_tui_set_breakpoints(ds4_tui *t, int on);
int  ds4_tui_breakpoints(const ds4_tui *t);
void ds4_tui_breakpoint(ds4_tui *t, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* DS4_TUI_H */
