/* ds4_tui.c — see ds4_tui.h.
 *
 * Rendering model: each pane owns a tui_doc (bytes + one style byte per byte).
 * Appends never fail the renderer; a frame re-wraps the doc to the pane width
 * and draws the visible window. That keeps chat, messages and profiler
 * independently scrollable without shared state.
 */
#include "ds4_tui.h"
#include "ds4_profiler.h"

#include <errno.h>
#include <fcntl.h>
#include <locale.h>
#include <poll.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>

#define TUI_HIST 256

static double tui_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Style values; bit 0x80 marks a selected line (reverse video), 0x40 marks a
 * row that must keep its columns: such a row is never wrapped, it is clipped
 * at the pane edge and can be scrolled sideways by the row cursor. */
enum {
    ST_NORMAL = 0, ST_DIM, ST_BOLD, ST_CYAN, ST_GREEN, ST_YELLOW,
    ST_RED, ST_BLUE, ST_MAGENTA, ST_GRAY, ST_TITLE, ST_USER, ST_ASSISTANT
};

#define ST_NOWRAP 0x40

static const char *k_sgr[] = {
    "\x1b[0m", "\x1b[90m", "\x1b[1m", "\x1b[36m", "\x1b[32m", "\x1b[33m",
    "\x1b[31m", "\x1b[34m", "\x1b[35m", "\x1b[37m", "\x1b[1;36m",
    "\x1b[1;36m", "\x1b[1;32m",
};

typedef struct {
    size_t off, len;
} vis_line;

typedef struct {
    char *s;
    size_t len, cap;
    unsigned char *st;
    size_t scap;
    /* Wrap cache: a frame redraws all visible panes, but the doc only grows,
     * so the line index is valid until the byte count or the width changes.
     * Without it every 100 ms frame re-scans the whole chat transcript. */
    int wrap_w;
    size_t wrap_len;
    vis_line *wrap;
    int wrap_n, wrap_cap, wrap_stale;
} tui_doc;

typedef enum {
    K_NONE = 0, K_CHAR, K_ENTER, K_BACKSPACE, K_DELETE, K_LEFT, K_RIGHT,
    K_UP, K_DOWN, K_PGUP, K_PGDN, K_HOME, K_END, K_TAB, K_STAB, K_ESC,
    K_CTRL_A, K_CTRL_B, K_CTRL_C, K_CTRL_D, K_CTRL_E, K_CTRL_K, K_CTRL_L,
    K_CTRL_U, K_CTRL_W, K_CTRL_N, K_CTRL_P, K_CTRL_Y, K_ALT_1, K_ALT_2,
    K_ALT_3, K_ALT_4, K_PASTE
} tui_key;

typedef enum {
    PV_OVERVIEW = 0, PV_TOKENS, PV_DETAIL, PV_DEVICES, PV_EXPERTS
} prof_view;

/* Cell grid + per-row diff. Drawing into a grid and writing only changed
 * rows is what keeps the bottom pane and the profiler from flickering: the
 * terminal never sees a full-screen clear during normal updates. */
#define CS_BOLD 0x80
#define CS_REV  0x40
#define CS_MASK 0x3f

typedef struct {
    uint32_t cp;
    uint8_t  style;
    uint8_t  cont;      /* second cell of a double-width glyph */
} screen_cell;

typedef struct {
    int rows, cols;
    screen_cell *cells;
    screen_cell *prev;      /* cells as last painted on the terminal */
    int prev_rows, prev_cols;
    int force_clear;
    int sync;               /* DS4_TUI_SYNC=1: wrap frames in ?2026 updates */
} screen;

struct ds4_tui {
    int active;
    int rows, cols;
    struct termios orig;
    int term_saved;
    int orig_flags;
    int saved_stderr;
    int pipe_r, pipe_w;

    tui_doc chat, msg, help, prof;
    int prof_scroll;
    int chat_top, msg_top, help_top;
    int chat_follow, msg_follow;
    /* Row cursor per pane (index into the wrapped visual lines, -1 = none).
     * A value wider than the pane is only readable if it can be scrolled, so
     * the cursor row alone is drawn with a horizontal offset that advances by
     * itself; every other row stays left aligned. */
    int cur_chat, cur_msg, cur_prof, cur_help;
    int hoff;                 /* columns shifted inside the cursor row      */
    double hoff_t;            /* last advance of hoff                       */
    size_t sel_off;           /* byte offset of the selected record's row   */
    int sel_off_valid;
    int slide;                /* cursor row is wider than its pane          */
    int reveal_cur;           /* a key moved the cursor: scroll it into view */
    int chat_cols, msg_cols, help_cols;  /* pane content widths, per frame */
    /* Tokens view filter: decode only by default, because a decode token is
     * what a latency question is about and prefill chunks drown it. */
    int kind_filter;
    screen scr;
    int prof_rows, prof_cols;

    ds4_tui_focus focus;
    int readline;
    int submitted, eof;
    char *input;
    size_t input_len, input_cap, input_pos;
    char *history[TUI_HIST];
    int hist_n, hist_pos;
    char prompt[64];
    /* Bracketed paste: the terminal frames a paste with ESC[200~ .. ESC[201~,
     * so a pasted newline is data, not a keypress. The frame can arrive over
     * several reads, hence the buffer lives in the TUI. */
    int pasting;
    char *paste;
    size_t paste_len, paste_cap;
    int busy;
    char status[160];
    int dirty;
    double last_draw, last_refresh;
    double frame_iv;          /* minimum time between frames (DS4_TUI_FPS) */
    double last_size_check, resize_at;

    /* profiler state */
    ds4_prof_view *view;
    prof_view pv;
    int sel;                  /* index into view->records, newest last      */
    int have_sel;
    uint64_t sel_seq;
    uint64_t detail_seq;
    ds4_prof_record detail;
    int detail_ready;
    ds4_prof_event *detail_events;
    int detail_event_n, detail_event_cap;
    ds4_prof_expert_trace *detail_experts;
    int detail_expert_n, detail_expert_cap;
    int granular;
    int breakpoints;
    int paused;
    int records_enabled;
    ds4_prof_record_meta list[DS4_PROF_REC_RING];
    int list_n;
    int list_valid;
    int list_frozen;
    int page;
    uint64_t list_newest_seq;
    int new_count;

    /* live history (nvtop-style) */
    double h_tps[TUI_HIST], h_ms[TUI_HIST];
    double h_ssd[TUI_HIST], h_ram[TUI_HIST];
    uint8_t h_util[DS4_PROF_REC_MAX_DEV][TUI_HIST];
    int h_n, h_head;
    uint64_t last_ssd;
    double last_sample;
    int dev_n;

    int (*irq)(void *);
    void (*irq_clear)(void *);
    void *irq_ud;
};

/* ------------------------------------------------------------------ docs -- */

static void doc_reserve(tui_doc *d, size_t extra) {
    if (d->len + extra + 1 <= d->cap) return;
    size_t cap = d->cap ? d->cap : 4096;
    while (cap < d->len + extra + 1) cap *= 2;
    char *s = realloc(d->s, cap);
    unsigned char *st = realloc(d->st, cap);
    if (!s || !st) { free(s); free(st); return; }
    d->s = s; d->st = st; d->cap = cap;
}

static void doc_putn(tui_doc *d, const char *s, size_t n, unsigned char style) {
    if (!s || n == 0) return;
    doc_reserve(d, n);
    if (d->len + n + 1 > d->cap) return;
    memcpy(d->s + d->len, s, n);
    memset(d->st + d->len, style, n);
    d->len += n;
    d->s[d->len] = '\0';
}

static void doc_puts(tui_doc *d, const char *s, unsigned char style) {
    doc_putn(d, s, strlen(s), style);
}

static void doc_nl(tui_doc *d) {
    doc_putn(d, "\n", 1, ST_NORMAL);
}

static void doc_clear(tui_doc *d) {
    d->len = 0;
    if (d->s && d->cap) d->s[0] = '\0';
    d->wrap_stale = 1;      /* rebuilt text may keep the same length */
}

static void doc_printf(tui_doc *d, unsigned char style, const char *fmt, ...) {
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    doc_putn(d, tmp, (size_t)n, style);
}

static void doc_line(tui_doc *d, unsigned char style, const char *fmt, ...) {
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    if ((size_t)n >= sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    doc_putn(d, tmp, (size_t)n, style);
    doc_nl(d);
}

/* A line whose columns matter (a table row): never wrapped, clipped instead,
 * so numbers stay under their header while the row scrolls sideways. */
static void doc_line_wide(tui_doc *d, unsigned char style, const char *fmt, ...) {
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    if ((size_t)n >= sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    const size_t start = d->len;
    doc_putn(d, tmp, (size_t)n, style);
    for (size_t i = start; i < d->len; i++) d->st[i] |= ST_NOWRAP;
    doc_nl(d);
}

/* ----------------------------------------------------------------- width -- */

/* Locale-independent UTF-8 decoding. mbrtowc/wcwidth treat our box-drawing
 * bytes as invalid under LANG=C and turn every border into three glyphs,
 * which corrupts row widths and leaves ghosts when scrolling. */
static size_t utf8_decode(const char *s, size_t n, uint32_t *cp) {
    const unsigned char *p = (const unsigned char *)s;
    if (n == 0) { *cp = 0; return 0; }
    if (p[0] < 0x80) { *cp = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0 && n >= 2 && (p[1] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
        return 2;
    }
    if ((p[0] & 0xF0) == 0xE0 && n >= 3 &&
        (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(p[0] & 0x0F) << 12) |
              ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
        return 3;
    }
    if ((p[0] & 0xF8) == 0xF0 && n >= 4 &&
        (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 &&
        (p[3] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(p[0] & 0x07) << 18) |
              ((uint32_t)(p[1] & 0x3F) << 12) |
              ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
        return 4;
    }
    *cp = p[0];
    return 1;
}

static int cp_wcwidth(uint32_t cp) {
    if (cp == 0 || cp < 0x20) return 0;
    if (cp < 0x7F) return 1;
    /* Box drawing, block elements, common punctuation. */
    if (cp >= 0x2500 && cp <= 0x25FF) return 1;
    /* Combining marks. */
    if ((cp >= 0x0300 && cp <= 0x036F) ||
        (cp >= 0x1AB0 && cp <= 0x1AFF) ||
        (cp >= 0x20D0 && cp <= 0x20FF) ||
        (cp >= 0xFE20 && cp <= 0xFE2F)) return 0;
    /* Wide East Asian / emoji ranges. */
    if ((cp >= 0x1100 && cp <= 0x115F) ||
        (cp >= 0x2E80 && cp <= 0xA4CF) ||
        (cp >= 0xAC00 && cp <= 0xD7A3) ||
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0xFE30 && cp <= 0xFE6F) ||
        (cp >= 0xFF00 && cp <= 0xFF60) ||
        (cp >= 0xFFE0 && cp <= 0xFFE6) ||
        (cp >= 0x1F300 && cp <= 0x1FAFF)) return 2;
    return 1;
}

static int cp_width(const char *s, size_t n, size_t *used) {
    uint32_t cp = 0;
    size_t r = utf8_decode(s, n, &cp);
    if (r == 0) { *used = 1; return 1; }
    *used = r;
    int w = cp_wcwidth(cp);
    return w < 1 ? 1 : w;
}

/* Split a doc into visual lines at width `w`, preferring a word boundary in the
 * second half of the line so prose does not break mid-word. The result is
 * cached on the doc: text is append-only except after doc_clear(), and a resize
 * is the only thing that changes the width, so a streaming chat does not
 * re-wrap per frame. */
static const vis_line *doc_wrap(tui_doc *d, int w, int *out_n) {
    if (w < 1) w = 1;
    if (!d->wrap_stale && d->wrap && d->wrap_w == w && d->wrap_len == d->len) {
        *out_n = d->wrap_n;
        return d->wrap;
    }
    int cap = d->wrap_cap > 0 ? d->wrap_cap : 64;
    if (cap < 64) cap = 64;
    vis_line *v = realloc(d->wrap, (size_t)cap * sizeof(*v));
    if (!v) { *out_n = 0; return NULL; }
    d->wrap = v;
    d->wrap_cap = cap;
    int n = 0;
    size_t i = 0, start = 0;
    int col = 0;
    size_t brk = 0;       /* byte after the last space on this visual line */
    int brk_col = 0;      /* its column, so a wrapped tail keeps its width  */
    while (i < d->len) {
        unsigned char c = (unsigned char)d->s[i];
        if (c != '\n' && start == i && d->st && (d->st[i] & ST_NOWRAP)) {
            /* Column-critical row: one visual line, however wide it is. */
            while (i < d->len && d->s[i] != '\n') i++;
            if (n == d->wrap_cap) {
                d->wrap_cap *= 2;
                v = realloc(d->wrap, (size_t)d->wrap_cap * sizeof(*v));
                if (!v) { *out_n = 0; return NULL; }
                d->wrap = v;
            }
            v[n].off = start; v[n].len = i - start; n++;
            if (i < d->len) i++;
            start = i; col = 0; brk = 0;
            continue;
        }
        if (c == '\n') {
            if (n == d->wrap_cap) {
                d->wrap_cap *= 2;
                v = realloc(d->wrap, (size_t)d->wrap_cap * sizeof(*v));
                if (!v) { *out_n = 0; return NULL; }
                d->wrap = v;
            }
            v[n].off = start; v[n].len = i - start; n++;
            i++; start = i; col = 0; brk = 0;
            continue;
        }
        if (c == ' ') {
            brk = i + 1; brk_col = col + 1;
            col++; i++;
            continue;
        }
        if (c == '\t') {
            int adv = 4 - (col % 4);
            if (col + adv > w) {
                if (n == d->wrap_cap) {
                    d->wrap_cap *= 2;
                    v = realloc(d->wrap, (size_t)d->wrap_cap * sizeof(*v));
                    if (!v) { *out_n = 0; return NULL; }
                    d->wrap = v;
                }
                v[n].off = start; v[n].len = i - start; n++;
                start = i; col = 0; brk = 0;
                continue;
            }
            col += adv; i++;
            continue;
        }
        size_t used = 1;
        int cw = cp_width(d->s + i, d->len - i, &used);
        if (col + cw > w && col > 0) {
            size_t end = i;
            int tail = 0;
            /* Break at the last space when it is not so early that the line
             * would look ragged; the tail keeps its counted width so a word
             * longer than the pane still splits at the edge. */
            if (brk > start && brk_col * 2 >= w) {
                end = brk - 1;            /* the space is dropped */
                tail = col - brk_col;
            }
            if (n == d->wrap_cap) {
                d->wrap_cap *= 2;
                v = realloc(d->wrap, (size_t)d->wrap_cap * sizeof(*v));
                if (!v) { *out_n = 0; return NULL; }
                d->wrap = v;
            }
            v[n].off = start; v[n].len = end - start; n++;
            start = tail ? brk : i; col = tail; brk = 0;
            continue;
        }
        if (cw > w) cw = w;
        col += cw;
        i += used;
    }
    if (start < d->len || n == 0) {
        if (n == d->wrap_cap) {
            d->wrap_cap++;
            v = realloc(d->wrap, (size_t)d->wrap_cap * sizeof(*v));
            if (!v) { *out_n = 0; return NULL; }
            d->wrap = v;
        }
        v[n].off = start; v[n].len = d->len - start; n++;
    }
    d->wrap_n = n;
    d->wrap_w = w;
    d->wrap_len = d->len;
    d->wrap_stale = 0;
    *out_n = n;
    return d->wrap;
}

/* ------------------------------------------------------------------ term -- */

static void term_size(ds4_tui *t) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 &&
        ws.ws_row > 0 && ws.ws_col > 0) {
        t->rows = ws.ws_row;
        t->cols = ws.ws_col;
    } else {
        t->rows = 24;
        t->cols = 80;
    }
}

static int term_init(ds4_tui *t) {
    if (tcgetattr(STDIN_FILENO, &t->orig) != 0) return 0;
    struct termios raw = t->orig;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO | IEXTEN);
    raw.c_lflag |= ISIG;          /* Ctrl-C still reaches the CLI handler */
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) return 0;
    t->term_saved = 1;
    t->orig_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, t->orig_flags | O_NONBLOCK);
    /* The first frame clears; clearing twice at startup makes slow terminals
     * show a blank flash right before the UI appears. */
    fputs("\x1b[?1049h\x1b[?7l\x1b[?25l\x1b[?2004h\x1b[0m", stdout);
    fflush(stdout);
    return 1;
}

static void term_done(ds4_tui *t) {
    if (t->pipe_w >= 0) { close(t->pipe_w); t->pipe_w = -1; }
    if (t->saved_stderr >= 0) {
        dup2(t->saved_stderr, STDERR_FILENO);
        close(t->saved_stderr);
        t->saved_stderr = -1;
    }
    if (t->term_saved) {
        fcntl(STDIN_FILENO, F_SETFL, t->orig_flags);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &t->orig);
        t->term_saved = 0;
    }
    fputs("\x1b[0m\x1b[?2004l\x1b[?7h\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
}

/* ---------------------------------------------------------------- output -- */

/* Small string builder, used for bars/sparklines and the final row image. */
typedef struct {
    char *s;
    size_t len, cap;
} sbuf;

static void sb_reserve(sbuf *b, size_t n) {
    if (b->len + n + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 8192;
    while (cap < b->len + n + 1) cap *= 2;
    char *s = realloc(b->s, cap);
    if (!s) return;
    b->s = s; b->cap = cap;
}

static void sb_add(sbuf *b, const char *s, size_t n) {
    sb_reserve(b, n);
    if (b->len + n + 1 > b->cap) return;
    memcpy(b->s + b->len, s, n);
    b->len += n;
    b->s[b->len] = '\0';
}

static void sb_puts(sbuf *b, const char *s) { sb_add(b, s, strlen(s)); }
static void sb_printf(sbuf *b, const char *fmt, ...) {
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n > 0) sb_add(b, tmp, (size_t)n < sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
}

static void utf8_encode(uint32_t cp, char out[5]) {
    if (cp < 0x80) { out[0] = (char)cp; out[1] = 0; }
    else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = 0;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = 0;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        out[4] = 0;
    }
}

static void screen_drop_prev(screen *s) {
    free(s->prev);
    s->prev = NULL;
    s->prev_rows = s->prev_cols = 0;
}

static void screen_resize(screen *s, int rows, int cols) {
    if (rows == s->rows && cols == s->cols && s->cells) return;
    free(s->cells);
    screen_drop_prev(s);
    s->rows = rows;
    s->cols = cols;
    s->cells = calloc((size_t)rows * (size_t)cols, sizeof(*s->cells));
    s->prev = calloc((size_t)rows * (size_t)cols, sizeof(*s->prev));
    if (s->prev)
        memset(s->prev, 0xFF, (size_t)rows * (size_t)cols * sizeof(*s->prev));
    s->prev_rows = rows;
    s->prev_cols = cols;
    s->force_clear = 1;
}

static void screen_clear(screen *s) {
    if (!s->cells) return;
    for (int i = 0; i < s->rows * s->cols; i++) {
        s->cells[i].cp = ' ';
        s->cells[i].style = ST_NORMAL;
        s->cells[i].cont = 0;
    }
}


static void screen_put(screen *s, int row, int col, uint32_t cp,
                       uint8_t style, int width) {
    if (row < 0 || row >= s->rows || col < 0 || col >= s->cols) return;
    if (width < 1) width = 1;
    if (col + width > s->cols) return;
    screen_cell *c = &s->cells[row * s->cols + col];
    c->cp = cp;
    c->style = style;
    c->cont = 0;
    for (int i = 1; i < width; i++) {
        screen_cell *n = &s->cells[row * s->cols + col + i];
        n->cp = 0;
        n->style = style;
        n->cont = 1;
    }
}

/* Decode UTF-8 and place it. Tabs expand to 4 columns. */
static void screen_puts(screen *s, int row, int col, uint8_t style,
                        const char *text, size_t len) {
    size_t i = 0;
    while (i < len && col < s->cols) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\t') {
            int adv = 4 - (col % 4);
            for (int k = 0; k < adv && col < s->cols; k++, col++)
                screen_put(s, row, col, ' ', style, 1);
            i++;
            continue;
        }
        if (c < 0x20) { i++; continue; }
        uint32_t wc = 0;
        size_t used = utf8_decode(text + i, len - i, &wc);
        if (used == 0) { i++; continue; }
        int cw = cp_wcwidth(wc);
        if (cw < 1) cw = 1;
        screen_put(s, row, col, wc, style, cw);
        col += cw;
        i += used;
    }
}

static void screen_printf(screen *s, int row, int col, uint8_t style,
                          const char *fmt, ...) {
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    screen_puts(s, row, col, style, tmp, (size_t)n);
}

/* Paint the cells that differ from what the terminal already shows. Rows are
 * diffed cell by cell and each row is written from its first to its last
 * changed column, so a streaming token costs a few bytes at the end of one
 * line instead of a full-width row rewrite. That is what keeps a software
 * rendered terminal responsive. */
static void screen_flush(screen *s, int cur_row, int cur_col, int show_cursor) {
    if (!s->cells || s->rows <= 0 || s->cols <= 0) return;
    sbuf out;
    memset(&out, 0, sizeof(out));
    if (s->sync) sb_puts(&out, "\x1b[?2026h");   /* begin synchronized update */
    int full = s->force_clear || !s->prev ||
               s->prev_rows != s->rows || s->prev_cols != s->cols;
    if (full) {
        if (!s->prev || s->prev_rows != s->rows || s->prev_cols != s->cols) {
            screen_drop_prev(s);
            s->prev = calloc((size_t)s->rows * (size_t)s->cols, sizeof(*s->prev));
            s->prev_rows = s->rows;
            s->prev_cols = s->cols;
        }
        if (s->prev)
            memset(s->prev, 0xFF, (size_t)s->rows * (size_t)s->cols * sizeof(*s->prev));
        /* Home the cursor and clear once: a resize leaves the terminal with
         * reflowed rows that no diff can reconcile. */
        sb_puts(&out, "\x1b[0m\x1b[2J\x1b[H");
        s->force_clear = 0;
    }
    if (!s->prev) { free(out.s); return; }
    for (int r = 0; r < s->rows; r++) {
        const screen_cell *cur = s->cells + (size_t)r * s->cols;
        screen_cell *old = s->prev + (size_t)r * s->cols;
        int first = -1, last = -1;
        for (int c = 0; c < s->cols; c++) {
            if (cur[c].cp != old[c].cp || cur[c].style != old[c].style ||
                cur[c].cont != old[c].cont) {
                if (first < 0) first = c;
                last = c;
            }
        }
        if (first < 0) continue;
        while (first > 0 && cur[first].cont) first--;
        sb_printf(&out, "\x1b[%d;%dH", r + 1, first + 1);
        int prevkey = -1;
        for (int c = first; c <= last; c++) {
            const screen_cell *cell = &cur[c];
            if (cell->cont) continue;
            int base = cell->style & CS_MASK;
            if (base >= (int)(sizeof(k_sgr) / sizeof(k_sgr[0]))) base = 0;
            int bold = (cell->style & CS_BOLD) != 0;
            int rev = (cell->style & CS_REV) != 0;
            int key = base | (bold ? 0x100 : 0) | (rev ? 0x200 : 0);
            if (key != prevkey) {
                sb_puts(&out, "\x1b[0m");
                if (bold) sb_puts(&out, "\x1b[1m");
                if (rev) sb_puts(&out, "\x1b[7m");
                if (base) sb_puts(&out, k_sgr[base]);
                prevkey = key;
            }
            char enc[5];
            utf8_encode(cell->cp ? cell->cp : ' ', enc);
            sb_puts(&out, enc);
        }
        memcpy(old, cur, (size_t)s->cols * sizeof(*old));
    }
    if (show_cursor && cur_row > 0 && cur_col > 0)
        sb_printf(&out, "\x1b[%d;%dH\x1b[?25h", cur_row, cur_col);
    else
        sb_puts(&out, "\x1b[?25l");
    if (s->sync) sb_puts(&out, "\x1b[?2026l");
    size_t off = 0;
    while (off < out.len) {
        ssize_t n = write(STDOUT_FILENO, out.s + off, out.len - off);
        if (n <= 0) break;
        off += (size_t)n;
    }
    free(out.s);
}

/* -------------------------------------------------------------- drawing -- */

/* Visual line that renders the given byte offset (the last line starting at or
 * before it). Used to point the cursor at a row the builder knows by offset. */
static int doc_visual_line(tui_doc *d, int w, size_t off) {
    int n = 0;
    const vis_line *lines = doc_wrap(d, w, &n);
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (lines[i].off > off) break;
        best = i;
    }
    return best;
}

/* Display width of one visual line, tabs expanded like the renderer does. */
static int vis_line_width(const tui_doc *d, const vis_line *l) {
    const size_t end = l->off + l->len;
    int col = 0;
    size_t i = l->off;
    while (i < end) {
        if (d->s[i] == '\t') { col += 4 - (col % 4); i++; continue; }
        size_t used = 1;
        int cw = cp_width(d->s + i, end - i, &used);
        if (cw < 1) cw = 1;
        col += cw;
        i += used;
    }
    return col;
}

/* One pane's scroll position plus its row cursor. The cursor line is moved by
 * the arrow keys and the view then follows it once; scrolling keys move the
 * view alone and drag the cursor to the nearest visible row, so the two never
 * fight (a cursor pinned to the pane edge used to block scrolling entirely).
 * When the cursor row is wider than the pane it slides sideways by itself:
 * that is what makes a clipped table row readable without wrapping it
 * (wrapping loses the column alignment). */
typedef struct {
    int *top;
    int *follow;
    int *cur;         /* visual line index, -1 for none                   */
    int *hoff;        /* columns shifted inside the cursor row            */
    double *hoff_t;   /* when hoff last advanced, throttled by the caller  */
    int *slide;       /* set when the cursor row needs sliding             */
    double now;
    int reveal;       /* this frame came from a cursor key                 */
} pane_view;

#define HOFF_STEP 0.07        /* seconds per column while sliding         */

static void draw_doc(screen *s, tui_doc *d, int x, int y, int w, int h,
                     pane_view *pv) {
    if (w <= 0 || h <= 0) return;
    int n = 0;
    const vis_line *lines = doc_wrap(d, w, &n);
    const int max_first = n > h ? n - h : 0;
    int first = (pv->follow && *pv->follow) ? max_first : *pv->top;
    if (first < 0) first = 0;
    if (first > max_first) first = max_first;

    /* A cursor key scrolls the view to the cursor; any other scroll (PgUp,
     * PgDn, wheel) owns the viewport and the cursor rides along on the nearest
     * visible row. Without that split a cursor resting on the top or bottom
     * row kept snapping the pane back and scrolling stalled at the edges. */
    int cur = pv->cur ? *pv->cur : -1;
    if (n > 0 && cur >= n) cur = n - 1;
    if (cur < 0) cur = -1;
    if (cur >= 0 && pv->reveal) {
        if (cur < first) first = cur;
        if (cur >= first + h) first = cur - h + 1;
        if (first > max_first) first = max_first;
        if (first < 0) first = 0;
    } else if (cur >= 0) {
        if (cur < first) cur = first;
        else if (cur >= first + h) cur = first + h - 1;
        if (cur > n - 1) cur = n - 1;
    }
    if (pv->cur) *pv->cur = cur;
    if (pv->follow) *pv->follow = (first >= max_first);
    *pv->top = first;

    /* Slide only the cursor row, and only when it does not fit. */
    const int cur_seen = cur >= first && cur < first + h;
    int hoff = 0;
    int hmax = 0;
    if (cur_seen && cur < n && pv->hoff) {
        const int width = vis_line_width(d, &lines[cur]);
        hmax = width > w ? width - w : 0;
        hoff = *pv->hoff;
        if (hoff > hmax) hoff = 0;
        if (hmax > 0 && pv->hoff_t && pv->now - *pv->hoff_t >= HOFF_STEP) {
            hoff = hoff >= hmax ? 0 : hoff + 1;
            *pv->hoff_t = pv->now;
        }
        *pv->hoff = hoff;
    } else if (pv->hoff) {
        *pv->hoff = 0;
    }
    if (pv->slide) *pv->slide = hmax > 0;

    for (int row = 0; row < h; row++) {
        const int li = first + row;
        if (li < 0 || li >= n) break;
        const int is_cur = cur_seen && (li == cur);
        const int skip = is_cur ? hoff : 0;
        size_t i = lines[li].off, end = lines[li].off + lines[li].len;
        int col = 0, drawn = 0;
        while (i < end) {
            unsigned char st = d->st ? d->st[i] : 0;
            int sel = (st & 0x80) != 0 || is_cur;
            int base = st & 0x3f;
            if (base >= (int)(sizeof(k_sgr) / sizeof(k_sgr[0]))) base = 0;
            uint8_t style = (uint8_t)(base | (sel ? CS_REV : 0));
            if ((unsigned char)d->s[i] == '\t') {
                int adv = 4 - (col % 4);
                for (int k = 0; k < adv; k++, col++) {
                    if (col < skip) continue;
                    if (col - skip >= w) break;
                    screen_put(s, y + row, x + col - skip, ' ', style, 1);
                    drawn++;
                }
                i++;
                continue;
            }
            uint32_t wc = 0;
            size_t used = utf8_decode(d->s + i, end - i, &wc);
            if (used == 0) { i++; continue; }
            int cw = cp_wcwidth(wc);
            if (cw < 1) cw = 1;
            /* A shifted row still emits every glyph; the ones left of the
             * window are dropped instead of being clipped mid-glyph. */
            if (col < skip) { col += cw; i += used; continue; }
            if (col - skip + cw > w) break;
            screen_put(s, y + row, x + col - skip, wc, style, cw);
            col += cw;
            drawn += cw;
            i += used;
        }
        /* Reverse video spanning the whole row reads as a cursor, not as a
         * highlighted word that happens to be under it. */
        if (is_cur)
            for (; drawn < w; drawn++)
                screen_put(s, y + row, x + drawn, ' ', (uint8_t)(ST_NORMAL | CS_REV), 1);
    }
}

static uint8_t box_style(int focused) {
    return (uint8_t)(focused ? (ST_CYAN | CS_BOLD) : ST_GRAY);
}

static void draw_box_top(screen *s, int x, int y, int w, const char *title,
                         int focused) {
    uint8_t st = box_style(focused);
    if (w < 2) return;
    screen_put(s, y, x, 0x250C, st, 1);                 /* ┌ */
    for (int i = 1; i < w - 1; i++)
        screen_put(s, y, x + i, 0x2500, st, 1);         /* ─ */
    screen_put(s, y, x + w - 1, 0x2510, st, 1);         /* ┐ */
    if (title && title[0])
        screen_puts(s, y, x + 2, ST_CYAN | CS_BOLD, title, strlen(title));
}

static void draw_box_side(screen *s, int x, int y, int h) {
    for (int row = 0; row < h; row++)
        screen_put(s, y + row, x, 0x2502, ST_GRAY, 1);  /* │ */
}

static void draw_box_bottom(screen *s, int x, int y, int w, int focused) {
    uint8_t st = box_style(focused);
    if (w < 2) return;
    screen_put(s, y, x, 0x2514, st, 1);                 /* └ */
    for (int i = 1; i < w - 1; i++)
        screen_put(s, y, x + i, 0x2500, st, 1);         /* ─ */
    screen_put(s, y, x + w - 1, 0x2518, st, 1);         /* ┘ */
}

static void screen_fit(screen *s, int row, int col, uint8_t style,
                       const char *text, int w) {
    if (!text || w <= 0) return;
    size_t i = 0, n = strlen(text);
    int used_cols = 0;
    while (i < n) {
        size_t used = 1;
        int cw = cp_width(text + i, n - i, &used);
        if (used_cols + cw > w) {
            if (used_cols + 1 <= w) screen_put(s, row, col + used_cols, '~', style, 1);
            return;
        }
        screen_puts(s, row, col + used_cols, style, text + i, used);
        used_cols += cw;
        i += used;
    }
}

/* ------------------------------------------------------------- profiler -- */

static const char *view_name(prof_view v) {
    switch (v) {
    case PV_TOKENS: return "tokens";
    case PV_DETAIL: return "token detail";
    case PV_DEVICES: return "devices";
    case PV_EXPERTS: return "experts";
    default: return "overview";
    }
}

static const char *kind_short(uint8_t k) {
    switch (k) {
    case DS4_PREC_PREFILL_CHUNK: return "prefill";
    case DS4_PREC_PREFILL_TOKEN: return "prefill-t";
    default: return "decode";
    }
}

static char kind_char(uint8_t k) {
    switch (k) {
    case DS4_PREC_PREFILL_CHUNK: return 'P';
    case DS4_PREC_PREFILL_TOKEN: return 'p';
    default: return 'D';
    }
}

static void bar(sbuf *b, double v, double max, int width, const char *color) {
    (void)color;
    int n = max > 0 ? (int)((v / max) * width + 0.5) : 0;
    if (n < 0) n = 0;
    if (n > width) n = width;
    sb_puts(b, "[");
    for (int i = 0; i < width; i++) sb_puts(b, i < n ? "#" : "-");
    sb_puts(b, "]");
}

/* 8-level sparkline. Zero values render as a baseline dot. */
static void spark(sbuf *b, const double *v, int n, double max, const char *color) {
    static const char *blocks[] = {" ", ".", ":", "-", "=", "+", "*", "#"};
    (void)color;
    if (max <= 0) max = 1;
    for (int i = 0; i < n; i++) {
        double x = v[i];
        if (x < 0) x = 0;
        int lv = (int)(x / max * 7.0 + 0.5);
        if (lv < 0) lv = 0;
        if (lv > 7) lv = 7;
        sb_puts(b, blocks[lv]);
    }
}

static void hist_push(ds4_tui *t, const ds4_prof_view *v) {
    const int i = t->h_head % TUI_HIST;
    const double tps = ds4_prof_avg(&v->timing.tokens_per_s);
    t->h_tps[i] = tps;
    t->h_ms[i] = ds4_prof_avg(&v->timing.per_token_ms);
    double ssd = 0;
    const double now = tui_now();
    const double dt = now - t->last_sample;
    if (t->last_sample > 0 && dt > 1e-6 &&
        v->ssd_read_bytes >= t->last_ssd) {
        ssd = (double)(v->ssd_read_bytes - t->last_ssd) / dt;
    }
    t->h_ssd[i] = ssd;
    t->h_ram[i] = (double)v->host_ram_used;
    t->last_ssd = v->ssd_read_bytes;
    t->last_sample = now;
    for (int d = 0; d < DS4_PROF_REC_MAX_DEV; d++) {
        t->h_util[d][i] = d < v->dev_n ? (uint8_t)v->dev[d].util_pct : 0;
    }
    t->dev_n = v->dev_n;
    t->h_head = (t->h_head + 1) % TUI_HIST;
    if (t->h_n < TUI_HIST) t->h_n++;
}

static void hist_series(const ds4_tui *t, const double *src, double *out, int maxn) {
    int n = t->h_n < maxn ? t->h_n : maxn;
    const int start = (t->h_head - n + TUI_HIST) % TUI_HIST;
    for (int i = 0; i < n; i++) out[i] = src[(start + i) % TUI_HIST];
}

static double series_max(const double *v, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) if (v[i] > m) m = v[i];
    return m > 0 ? m : 1;
}

static void hist_series_u8(const ds4_tui *t, const uint8_t *src, double *out, int maxn) {
    int n = t->h_n < maxn ? t->h_n : maxn;
    const int start = (t->h_head - n + TUI_HIST) % TUI_HIST;
    for (int i = 0; i < n; i++) out[i] = src[(start + i) % TUI_HIST];
}

/* Which record kinds the records view shows. A decode token and a prefill
 * chunk are different measurements, so the list is filtered by kind instead of
 * interleaving them; decode is the default because that is the latency a
 * reader is usually chasing. */
static int rec_kind_match(const ds4_tui *t, int kind) {
    switch (t->kind_filter) {
    case 0: return kind == DS4_PREC_DECODE_TOKEN;
    case 1: return kind == DS4_PREC_PREFILL_CHUNK || kind == DS4_PREC_PREFILL_TOKEN;
    default: return 1;
    }
}

static const char *kind_filter_name(const ds4_tui *t) {
    switch (t->kind_filter) {
    case 0: return "decode";
    case 1: return "prefill";
    default: return "all";
    }
}

/* The list is a snapshot. While generation is running it deliberately does
 * not slide under the reader; new records are counted and appear when the
 * user presses u, or automatically when generation stops. */
static void list_refresh(ds4_tui *t, int force) {
    if (!t->view) return;
    const int n = t->view->n_records;
    if (n <= 0) { t->list_n = 0; t->list_valid = 1; t->new_count = 0; return; }
    const uint64_t newest = t->view->records[n - 1].seq;
    if (!force && t->busy && t->list_frozen && t->list_n > 0) {
        int fresh = 0;
        for (int i = n - 1; i >= 0 && t->view->records[i].seq > t->list_newest_seq; i--)
            if (rec_kind_match(t, (int)t->view->records[i].kind)) fresh++;
        t->new_count = fresh;
        return;
    }
    int kept = 0;
    for (int i = 0; i < n && kept < DS4_PROF_REC_RING; i++) {
        if (!rec_kind_match(t, (int)t->view->records[i].kind)) continue;
        t->list[kept++] = t->view->records[i];
    }
    t->list_n = kept;
    t->list_newest_seq = newest;
    t->new_count = 0;
    t->list_valid = 1;
    if (!t->busy) t->list_frozen = 0;
    if (t->sel >= t->list_n) t->sel = t->list_n - 1;
    if (t->sel < 0) t->sel = 0;
    if (t->have_sel) {
        int found = 0;
        for (int i = 0; i < t->list_n; i++)
            if (t->list[i].seq == t->sel_seq) { found = 1; t->sel = i; break; }
        if (!found) t->have_sel = 0;
    }
    if (!t->have_sel && t->list_n > 0) {
        t->sel = t->list_n - 1;
        t->sel_seq = t->list[t->sel].seq;
        t->have_sel = 1;
    }
}

static const ds4_prof_record_meta *sel_meta(ds4_tui *t) {
    if (t->list_n <= 0) return NULL;
    if (t->have_sel) {
        for (int i = 0; i < t->list_n; i++)
            if (t->list[i].seq == t->sel_seq) { t->sel = i; return &t->list[i]; }
    }
    if (t->sel < 0) t->sel = 0;
    if (t->sel >= t->list_n) t->sel = t->list_n - 1;
    t->sel_seq = t->list[t->sel].seq;
    t->have_sel = 1;
    return &t->list[t->sel];
}

static int token_page_size(const ds4_tui *t) {
    int ps = t->prof_rows - (t->prof_rows < 14 ? 4 : 10);
    return ps < 2 ? 2 : ps;
}

static void detail_open(ds4_tui *t, uint64_t seq) {
    t->detail_ready = 0;
    if (!ds4_profiler_record_get(seq, &t->detail)) return;
    if (t->detail.event_n) {
        if ((int)t->detail.event_n > t->detail_event_cap) {
            ds4_prof_event *p = realloc(t->detail_events,
                                        (size_t)t->detail.event_n * sizeof(*p));
            if (p) { t->detail_events = p; t->detail_event_cap = (int)t->detail.event_n; }
        }
        t->detail_event_n = t->detail_events
            ? ds4_profiler_record_events(seq, t->detail_events, t->detail_event_cap) : 0;
    } else {
        t->detail_event_n = 0;
    }
    if (t->detail.expert_n) {
        if ((int)t->detail.expert_n > t->detail_expert_cap) {
            ds4_prof_expert_trace *p = realloc(t->detail_experts,
                                               (size_t)t->detail.expert_n * sizeof(*p));
            if (p) { t->detail_experts = p; t->detail_expert_cap = (int)t->detail.expert_n; }
        }
        t->detail_expert_n = t->detail_experts
            ? ds4_profiler_record_experts(seq, t->detail_experts, t->detail_expert_cap) : 0;
    } else {
        t->detail_expert_n = 0;
    }
    t->detail_seq = seq;
    t->detail_ready = 1;
    t->pv = PV_DETAIL;
    t->prof_scroll = 0;
    t->dirty = 1;
}

static const ds4_prof_record_meta *latest_kind(const ds4_tui *t, int kind) {
    if (!t->view) return NULL;
    for (int i = t->view->n_records - 1; i >= 0; i--) {
        if ((int)t->view->records[i].kind == kind) return &t->view->records[i];
    }
    return NULL;
}

static void build_overview(ds4_tui *t) {
    tui_doc *d = &t->prof;
    doc_clear(d);
    ds4_prof_view *v = t->view;
    const ds4_profiler_timing *tm = &v->timing;
    doc_line(d, ST_TITLE, " D %.2f t/s  %.2f ms  P %.2f t/s",
             ds4_prof_avg(&tm->tokens_per_s), ds4_prof_avg(&tm->per_token_ms),
             ds4_prof_avg(&tm->prefill_tokens_per_s));
    doc_line(d, ST_GRAY, " capture %s  n=%llu", t->granular ? "on" : "off",
             (unsigned long long)tm->tokens);
    const ds4_prof_record_meta *ld = latest_kind(t, DS4_PREC_DECODE_TOKEN);
    const ds4_prof_record_meta *lp = latest_kind(t, DS4_PREC_PREFILL_CHUNK);
    if (!lp) lp = latest_kind(t, DS4_PREC_PREFILL_TOKEN);
    if (ld) {
        if (t->prof_cols >= 58)
            doc_line(d, ST_NORMAL,
                     " latest D #%llu %.2f ms %.0f t/s pos %d tok %u",
                     (unsigned long long)ld->seq, ld->ms,
                     ld->ms > 0 ? 1000.0 / ld->ms : 0, ld->pos,
                     ld->token == UINT32_MAX ? 0 : ld->token);
        else
            doc_line(d, ST_NORMAL, " latest D #%llu %.2f ms",
                     (unsigned long long)ld->seq, ld->ms);
    } else {
        doc_line(d, ST_GRAY, " latest D none yet");
    }
    if (lp) {
        const double per = lp->n_tokens ? lp->ms / (double)lp->n_tokens : lp->ms;
        doc_line(d, ST_NORMAL,
                 " latest P #%llu %u tok %.2f ms %.2f ms/t %.2f t/s",
                 (unsigned long long)lp->seq, lp->n_tokens, lp->ms, per,
                 per > 0 ? 1000.0 / per : 0);
    }
    doc_line(d, ST_NORMAL, "");

    /* Averages: what one token costs on average, and how much of that the
     * named stages explain. Kept deliberately separate from the per-stage
     * rows so the two never get read as the same number. */
    doc_line(d, ST_TITLE, " Averages over all counted tokens");
    {
        double dsum = 0, psum = 0;
        uint64_t dbytes = 0, pbytes = 0;
        for (int i = 0; i < v->n_stages; i++) {
            dsum += v->stages[i].kind[DS4_PREC_DECODE_TOKEN].total_ms;
            psum += v->stages[i].kind[DS4_PREC_PREFILL_CHUNK].total_ms;
            dbytes += v->stages[i].kind[DS4_PREC_DECODE_TOKEN].bytes;
            pbytes += v->stages[i].kind[DS4_PREC_PREFILL_CHUNK].bytes;
        }
        const double dper = tm->tokens ? dsum / (double)tm->tokens : 0;
        const double pper = tm->prefill_tokens ? psum / (double)tm->prefill_tokens : 0;
        const double wall = ds4_prof_avg(&tm->per_token_ms);
        const double pwall = tm->prefill_ms.n
            ? tm->prefill_ms.sum / (double)tm->prefill_ms.n : 0;
        char bdt[24], bpt[24];
        ds4_profiler_format_bytes(bdt, sizeof(bdt),
            tm->tokens ? dbytes / tm->tokens : 0);
        ds4_profiler_format_bytes(bpt, sizeof(bpt),
            tm->prefill_tokens ? pbytes / tm->prefill_tokens : 0);
        /* Named work is per-device wall clock and decode overlaps the two GPUs,
         * so it can exceed the token wall; a negative residual would read as a
         * bug, so say what it is instead. */
        if (dper <= wall)
            doc_line(d, ST_NORMAL,
                     "  decode   wall %.2f ms/token  named %.2f ms (%4.1f%%)  residual %.2f ms",
                     wall, dper, wall > 0 ? 100.0 * dper / wall : 0,
                     wall - dper);
        else
            doc_line(d, ST_NORMAL,
                     "  decode   wall %.2f ms/token  named %.2f ms (%4.1f%%, stages overlap)",
                     wall, dper, wall > 0 ? 100.0 * dper / wall : 0);
        doc_line(d, ST_NORMAL, "           %s/token moved, %llu tokens counted",
                 bdt, (unsigned long long)tm->tokens);
        if (tm->prefill_tokens)
            doc_line(d, ST_NORMAL, "  prefill  wall %.2f ms/chunk  named %.2f ms/chunk (%.2f ms/token)",
                     pwall, psum / (double)(tm->prefill_ms.n ? tm->prefill_ms.n : 1), pper);
        if (tm->prefill_tokens)
            doc_line(d, ST_NORMAL, "           %s/token moved, %llu tokens in %llu chunks",
                     bpt, (unsigned long long)tm->prefill_tokens,
                     (unsigned long long)tm->prefill_ms.n);
    }
    doc_line(d, ST_NORMAL, "");

    doc_line(d, ST_TITLE, " Cost per token by stage");
    const int wide = t->prof_cols >= 86;
    const char *cats[] = {"attention", "routed", "shared", "router", "indexer",
                          "norm and mixing", "output", "other"};
    /* Decode and prefill run the same stage boundaries with wildly different
     * durations, so each is tabulated on its own kind and never averaged
     * across kinds: that was what made two rows share a label with two
     * different values. */
    for (int pass = 0; pass < 2; pass++) {
        const int kind = pass == 0 ? DS4_PREC_DECODE_TOKEN : DS4_PREC_PREFILL_CHUNK;
        const char *kname = pass == 0 ? "decode token" : "prefill chunk";
        const double units = pass == 0 ? (double)tm->tokens
                                       : (double)tm->prefill_tokens;
        double ksum = 0;
        for (int i = 0; i < v->n_stages; i++) ksum += v->stages[i].kind[kind].total_ms;
        if (ksum <= 0) continue;
        doc_line(d, ST_CYAN, " %s  (%.3f ms each, n=%.0f)", kname,
                 units > 0 ? ksum / units : 0, units);
        if (wide)
            doc_line(d, ST_GRAY, "  stage                         total    launch     wait   calls/tok   share");
        else
            doc_line(d, ST_GRAY, "  stage             total launch  wait shr");
        int shown = 0;
        for (size_t ci = 0; ci < sizeof(cats) / sizeof(cats[0]); ci++) {
            int any = 0;
            for (int i = 0; i < v->n_stages; i++)
                if (!strcmp(v->stages[i].category, cats[ci]) &&
                    v->stages[i].kind[kind].total_ms > 0)
                    any = 1;
            if (!any) continue;
            doc_line(d, ST_CYAN, "  %s", cats[ci]);
            for (int i = 0; i < v->n_stages; i++) {
                const ds4_prof_stage_view *s = &v->stages[i];
                const ds4_prof_stage_kind *k = &s->kind[kind];
                if (strcmp(s->category, cats[ci]) || k->total_ms <= 0) continue;
                const double total = units > 0 ? k->total_ms / units : 0;
                const double enc = units > 0 ? k->encode_ms / units : 0;
                const double wait = units > 0 ? k->wait_ms / units : 0;
                const double share = 100.0 * k->total_ms / ksum;
                if (wide)
                    doc_line_wide(d, ST_NORMAL,
                                  "  %-28.28s %7.3f  %7.3f  %7.3f  %8.2f  %5.1f%%",
                                  s->label, total, enc, wait,
                                  units > 0 ? (double)k->calls / units : 0, share);
                else {
                    const int lw = t->prof_cols >= 44 ? 20 : 12;
                    doc_line_wide(d, ST_NORMAL, "  %-*.*s %5.2f %5.2f %5.2f %4.1f%%",
                                  lw, lw, s->label, total, enc, wait, share);
                }
                shown++;
            }
        }
        doc_line(d, ST_NORMAL, "  %-28.28s %7.3f  %35s %5.1f%%", "total",
                 units > 0 ? ksum / units : 0, "", 100.0);
        if (!shown)
            doc_line(d, ST_YELLOW,
                     "  no stage rows captured (capture is %s)",
                     t->granular ? "on" : "off");
    }
    doc_line(d, ST_NORMAL, "");

    doc_line(d, ST_TITLE, " Transfers and waits (overlap stages)");
    int sub = 0;
    for (int i = 0; i < v->n_stages; i++) {
        const ds4_prof_stage_view *s = &v->stages[i];
        if (strcmp(s->category, "transfer") && strcmp(s->category, "wait")) continue;
        const ds4_prof_stage_kind *k = &s->kind[DS4_PREC_DECODE_TOKEN];
        const double units = tm->tokens ? (double)tm->tokens : 0;
        if (k->total_ms <= 0 && !k->bytes && !s->total_ms.n && !s->bytes) continue;
        const double ms = units > 0 ? k->total_ms / units : 0;
        char per[24], rate[24];
        ds4_profiler_format_bytes(per, sizeof(per), units > 0 ? k->bytes / (uint64_t)units : 0);
        ds4_profiler_format_bytes(rate, sizeof(rate),
            k->total_ms > 0 ? (uint64_t)((double)k->bytes / (k->total_ms / 1e3)) : 0);
        doc_line_wide(d, ST_NORMAL, "  %-30.30s %8.3f ms/tok  %10s/tok  %s/s",
                      s->label, ms, per, rate);
        sub++;
    }
    if (!sub) doc_line(d, ST_GRAY, "  none measured yet");
    doc_line(d, ST_NORMAL, "");

    doc_line(d, ST_TITLE, " Expert residency");
    const ds4_profiler_stats *st = &v->stats;
    const ds4_profiler_host_stats *hs = &v->host;
    uint64_t gpu_exp = 0;
    for (int i = 0; i < st->n_devices; i++) gpu_exp += st->dev[i].expert_count;
    const uint64_t ram_exp = st->host_arena_capacity_bytes ? st->host_arena_experts
                                                           : st->host_experts_resident;
    const uint64_t total = hs->total_experts;
    const uint64_t ssd = total > gpu_exp + ram_exp ? total - gpu_exp - ram_exp : 0;
    doc_line(d, ST_NORMAL, "  %llu experts: %llu GPU, %llu RAM, %llu SSD",
             (unsigned long long)total, (unsigned long long)gpu_exp,
             (unsigned long long)ram_exp, (unsigned long long)ssd);
    if (st->host_arena_capacity_bytes) {
        char ba[24], bc[24];
        ds4_profiler_format_bytes(ba, sizeof(ba), st->host_arena_used_bytes);
        ds4_profiler_format_bytes(bc, sizeof(bc), st->host_arena_capacity_bytes);
        doc_line(d, ST_NORMAL, "  pinned host arena %s/%s  hits %llu  misses %llu  evictions %llu",
                 ba, bc,
                 (unsigned long long)st->host_arena_hits,
                 (unsigned long long)st->host_arena_misses,
                 (unsigned long long)st->host_arena_evictions);
    }
    doc_line(d, ST_NORMAL, "");
    doc_line(d, ST_TITLE, " Devices and storage (live)");
    for (int i = 0; i < v->dev_n; i++) {
        const ds4_prof_dev_state *dev = &v->dev[i];
        const char *name = dev->name;
        if (!strncmp(name, "NVIDIA GeForce ", 15)) name += 15;
        const double mem = dev->mem_total
            ? 100.0 * (double)dev->mem_used / (double)dev->mem_total : 0;
        char bu[24], bt[24];
        ds4_profiler_format_bytes(bu, sizeof(bu), dev->mem_used);
        ds4_profiler_format_bytes(bt, sizeof(bt), dev->mem_total);
        doc_line(d, ST_NORMAL, "  GPU%d %-18.18s %3u%%  %s/%s %3.0f%%",
                 dev->device, name, dev->util_pct, bu, bt, mem);
    }
    double series[64];
    int n = t->h_n < 24 ? t->h_n : 24;
    hist_series(t, t->h_tps, series, n);
    doc_printf(d, ST_GRAY, "  decode tok/s  ");
    {
        sbuf sp;
        memset(&sp, 0, sizeof(sp));
        spark(&sp, series, n, series_max(series, n), "\x1b[32m");
        doc_puts(d, sp.s ? sp.s : "", ST_GRAY);
        free(sp.s);
    }
    doc_printf(d, ST_GRAY, " %5.0f", n ? series[n - 1] : 0.0);
    doc_nl(d);
    hist_series(t, t->h_ms, series, n);
    doc_printf(d, ST_GRAY, "  token ms      ");
    {
        sbuf sp;
        memset(&sp, 0, sizeof(sp));
        spark(&sp, series, n, series_max(series, n), "\x1b[31m");
        doc_puts(d, sp.s ? sp.s : "", ST_GRAY);
        free(sp.s);
    }
    doc_printf(d, ST_GRAY, " %5.0f", n ? series[n - 1] : 0.0);
    doc_nl(d);
    hist_series(t, t->h_ssd, series, n);
    doc_printf(d, ST_GRAY, "  SSD read B/s  ");
    {
        sbuf sp;
        memset(&sp, 0, sizeof(sp));
        spark(&sp, series, n, series_max(series, n), "\x1b[33m");
        doc_puts(d, sp.s ? sp.s : "", ST_GRAY);
        free(sp.s);
    }
    doc_printf(d, ST_GRAY, " %5.0f", n ? series[n - 1] : 0.0);
    doc_nl(d);
    for (int dv = 0; dv < v->dev_n; dv++) {
        hist_series_u8(t, t->h_util[dv], series, n);
        doc_printf(d, ST_GRAY, "  GPU%d util     ", v->dev[dv].device);
        sbuf sp;
        memset(&sp, 0, sizeof(sp));
        spark(&sp, series, n, 100, "\x1b[36m");
        doc_puts(d, sp.s ? sp.s : "", ST_GRAY);
        free(sp.s);
        doc_printf(d, ST_GRAY, " %3.0f%%", n ? series[n - 1] : 0.0);
        doc_nl(d);
    }
}

static void build_tokens(ds4_tui *t) {
    tui_doc *d = &t->prof;
    doc_clear(d);
    ds4_prof_view *v = t->view;
    const int tiny = t->prof_rows < 14;
    if (!tiny) {
        doc_line(d, ST_TITLE, " Latest decode / prefill (always live)");
        const ds4_prof_record_meta *ld = latest_kind(t, DS4_PREC_DECODE_TOKEN);
        const ds4_prof_record_meta *lp = latest_kind(t, DS4_PREC_PREFILL_CHUNK);
        if (!lp) lp = latest_kind(t, DS4_PREC_PREFILL_TOKEN);
        if (ld) {
            const char *hot = ld->hot_stage < (uint16_t)v->n_stages
                ? v->stages[ld->hot_stage].label : "-";
            if (t->prof_cols >= 62)
                doc_line(d, ST_NORMAL,
                         "  D #%llu %.2f ms %.2f t/s pos %d tok %u hot %s",
                         (unsigned long long)ld->seq, ld->ms,
                         ld->ms > 0 ? 1000.0 / ld->ms : 0, ld->pos,
                         ld->token == UINT32_MAX ? 0 : ld->token, hot);
            else
                doc_line(d, ST_NORMAL, "  D #%llu %.2f ms",
                         (unsigned long long)ld->seq, ld->ms);
        } else doc_line(d, ST_GRAY, "  decode  no token yet");
        if (lp) {
            const double per = lp->n_tokens ? lp->ms / (double)lp->n_tokens : lp->ms;
            doc_line(d, ST_NORMAL,
                     "  P #%llu %u tok %.2f ms %.2f ms/t %.2f t/s",
                     (unsigned long long)lp->seq, lp->n_tokens, lp->ms, per,
                     per > 0 ? 1000.0 / per : 0);
        } else doc_line(d, ST_GRAY, "  prefill no chunk yet");
        doc_line(d, ST_NORMAL, "");
    }

    const int ps = token_page_size(t);
    int pages = t->list_n > 0 ? (t->list_n + ps - 1) / ps : 1;
    if (t->page >= pages) t->page = pages - 1;
    if (t->page < 0) t->page = 0;
    if (t->new_count)
        doc_line(d, ST_YELLOW, " Records [%s]  page %d/%d  +%d new - u to load",
                 kind_filter_name(t), t->page + 1, pages, t->new_count);
    else
        doc_line(d, ST_TITLE, " Records [%s]  page %d/%d", kind_filter_name(t),
                 t->page + 1, pages);
    if (t->list_n <= 0) {
        if (t->kind_filter != 2)
            doc_line(d, ST_YELLOW, "  no %s records yet", kind_filter_name(t));
        else
            doc_line(d, ST_YELLOW, "  no records yet");
        return;
    }
    const int wide = t->prof_cols >= 84;
    if (wide)
        doc_line_wide(d, ST_GRAY,
                      "  #    seq     k   pos        ms   ms/tok  hot stage  "
                      "pool/arena/ssd/cpu  util  text");
    else
        doc_line_wide(d, ST_GRAY,
                      "  #    seq   k  pos     ms ms/tok hot stage text");

    int end = t->list_n - 1 - t->page * ps;
    if (end < 0) end = t->list_n - 1;
    int begin = end - ps + 1;
    if (begin < 0) begin = 0;
    /* Keep the selected record on the visible page. */
    if (t->have_sel && t->sel >= 0 && t->sel < t->list_n) {
        if (t->sel < begin || t->sel > end) {
            const int target_page = (t->list_n - 1 - t->sel) / ps;
            t->page = target_page;
            if (t->page >= pages) t->page = pages - 1;
            end = t->list_n - 1 - t->page * ps;
            if (end < 0) end = t->list_n - 1;
            begin = end - ps + 1;
            if (begin < 0) begin = 0;
        }
    }
    /* Resolve the selection once before rows are marked, so the highlighted
     * row is the record the help card and the detail page describe. */
    (void)sel_meta(t);
    t->sel_off_valid = 0;
    for (int i = end; i >= begin; i--) {
        const ds4_prof_record_meta *r = &t->list[i];
        const int selected = t->have_sel ? (r->seq == t->sel_seq) : 0;
        const char *stage = "";
        if (r->hot_stage < (uint16_t)v->n_stages)
            stage = v->stages[r->hot_stage].label;
        const double per = r->n_tokens ? r->ms / (double)r->n_tokens : r->ms;
        const unsigned char st = (unsigned char)(ST_NORMAL | (selected ? 0x80 : 0));
        if (selected) { t->sel_off = d->len; t->sel_off_valid = 1; }
        if (wide)
            doc_line_wide(d, st, "%s%4d #%llu %c pos %4d %8.2f ms %7.3f ms/t  %s  "
                          "pool %u arena %u ssd %u cpu %u  util %3u%%  %s",
                          selected ? ">" : " ", i + 1,
                          (unsigned long long)r->seq, kind_char(r->kind), r->pos,
                          r->ms, per, stage,
                          r->experts_pool, r->experts_arena, r->experts_ssd,
                          r->experts_cpu, r->util_max, r->text);
        else
            doc_line_wide(d, st, "%s%4d %5llu %c %4d %5.2f %5.2f %s %s",
                          selected ? ">" : " ", i + 1,
                          (unsigned long long)r->seq, kind_char(r->kind), r->pos,
                          r->ms, per, stage, r->text);
    }
    if (t->new_count)
        doc_line(d, ST_YELLOW, "  %d newer record(s) hidden to keep this page readable; u to load",
                 t->new_count);
}

static void build_detail(ds4_tui *t) {
    tui_doc *d = &t->prof;
    doc_clear(d);
    if (!t->detail_ready) {
        doc_line(d, ST_YELLOW, " no record open (Enter on a record in view 2)");
        return;
    }
    ds4_prof_record r = t->detail;
    doc_line(d, ST_TITLE, " %s #%llu pos %d tok %u %.2f/%.2fms",
             ds4_profiler_record_kind_name(r.kind), (unsigned long long)r.seq,
             r.pos, r.n_tokens, r.ms, r.n_tokens ? r.ms / r.n_tokens : r.ms);
    doc_line(d, ST_GRAY, " text: %s", r.text[0] ? r.text : "(none)");
    {
        char bssd[24], bram[24];
        ds4_profiler_format_bytes(bssd, sizeof(bssd), r.ssd_read_bytes);
        ds4_profiler_format_bytes(bram, sizeof(bram), r.host_ram_used);
        doc_line(d, ST_GRAY, " cpu %.2fms sys %.0f%% ssd %s ram %s",
                 (double)r.proc_cpu_ns / 1e6,
                 r.sys_cpu_total_ns ? 100.0 * (double)r.sys_cpu_busy_ns / (double)r.sys_cpu_total_ns : 0,
                 bssd, bram);
    }

    doc_line(d, ST_TITLE, " Cost components (one row per operation)");
    const int dwide = t->prof_cols >= 96;
    if (dwide)
        doc_line(d, ST_GRAY, "  operation                     total    launch     wait    calls      bytes     rate");
    else
        doc_line(d, ST_GRAY, "  operation   total launch  wait cal   bytes");
    for (uint32_t i = 0; i < r.stage_n; i++) {
        char key[64] = {0};
        const int id = r.stages[i].id;
        const char *label = key;
        if (t->view && id < t->view->n_stages) {
            snprintf(key, sizeof(key), "%s", t->view->stages[id].key);
            label = t->view->stages[id].label;
        }
        const ds4_prof_stage_use *u = &r.stages[i];
        char by[24], br[24];
        ds4_profiler_format_bytes(by, sizeof(by), u->bytes);
        ds4_profiler_format_bytes(br, sizeof(br), u->total_ns
            ? (uint64_t)((double)u->bytes / ((double)u->total_ns / 1e9)) : 0);
        if (dwide)
            doc_line_wide(d, ST_NORMAL, "  %-26.26s %8.3f  %8.3f  %8.3f  %6llu %10s %10s/s",
                          label, (double)u->total_ns / 1e6,
                          (double)u->encode_ns / 1e6, (double)u->wait_ns / 1e6,
                          (unsigned long long)u->count, by, br);
        else
            doc_line_wide(d, ST_NORMAL, "  %-18.18s %6.2f %6.2f %6.2f %4llu %8s",
                          label, (double)u->total_ns / 1e6,
                          (double)u->encode_ns / 1e6, (double)u->wait_ns / 1e6,
                          (unsigned long long)u->count, by);
    }
    if (r.stage_n == 0)
        doc_line(d, ST_YELLOW, "  no stages recorded (stage capture was off)");
    {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < r.stage_n; i++) sum += r.stages[i].total_ns;
        const double wall = r.ms;
        const double named = (double)sum / 1e6;
        if (wall >= named)
            doc_line(d, ST_GRAY,
                     "  wall %.3f ms; named stages %.3f ms; residual %.3f ms",
                     wall, named, wall - named);
        else
            doc_line(d, ST_GRAY,
                     "  wall %.3f ms; named stages %.3f ms", wall, named);
    }

    doc_line(d, ST_TITLE, " Devices at record start / end (all devices on the machine)");
    for (uint32_t i = 0; i < r.dev_n; i++) {
        const ds4_prof_rec_dev *dv = &r.dev[i];
        char b0[24], b1[24], bt[24];
        ds4_profiler_format_bytes(b0, sizeof(b0), dv->mem_start);
        ds4_profiler_format_bytes(b1, sizeof(b1), dv->mem_end);
        ds4_profiler_format_bytes(bt, sizeof(bt), dv->mem_total);
        doc_line(d, ST_NORMAL,
                 "  GPU%d util %3u%% -> %3u%%   vram %s -> %s of %s",
                 dv->device, dv->util_start, dv->util_end, b0, b1, bt);
    }
    doc_line(d, ST_TITLE, " Link traffic for this record");
    for (int tier = 0; tier < DS4_PROF_MACHINE_TIERS; tier++) {
        if (!r.tier_weight_bytes[tier] && !r.tier_activation_bytes[tier] &&
            !r.tier_tx_bytes[tier] && !r.tier_rx_bytes[tier]) continue;
        char bw[24], ba2[24], btx[24], brx[24];
        ds4_profiler_format_bytes(bw, sizeof(bw), r.tier_weight_bytes[tier]);
        ds4_profiler_format_bytes(ba2, sizeof(ba2), r.tier_activation_bytes[tier]);
        ds4_profiler_format_bytes(btx, sizeof(btx), r.tier_tx_bytes[tier]);
        ds4_profiler_format_bytes(brx, sizeof(brx), r.tier_rx_bytes[tier]);
        doc_line(d, ST_NORMAL,
                 "  tier %d  weights %s  activations %s  into %s  out %s",
                 tier, bw, ba2, btx, brx);
    }

    const ds4_prof_expert_trace *ex = t->detail_experts;
    const int exn = t->detail_expert_n;
    doc_line(d, ST_TITLE, " Expert uses (%d)", exn);
    if (exn) {
        doc_line(d, ST_GRAY, "  layer expert source          from -> to        bytes");
        for (int i = 0; i < exn; i++) {
            static const char *src[] = {"VRAM", "host RAM", "SSD", "CPU"};
            doc_line(d, ST_NORMAL, "  %5u %6u %-14s %4u -> %-3u %10u",
                     (unsigned)ex[i].layer, (unsigned)ex[i].expert,
                     ex[i].source < 4 ? src[ex[i].source] : "?",
                     ex[i].src_gpu, ex[i].dest_gpu, ex[i].bytes);
        }
    } else {
        doc_line(d, ST_GRAY, "  none recorded (streaming load did not trace this record)");
    }
    const ds4_prof_event *ev = t->detail_events;
    const int evn = t->detail_event_n;
    doc_line(d, ST_TITLE, " Ordered events (%d)", evn);
    if (dwide)
        doc_line(d, ST_GRAY, "      us   duration  event                      layer expert dev      bytes");
    else
        doc_line(d, ST_GRAY, "      us  dur.ms event      lay exp dv   bytes");
    for (int i = 0; i < evn; i++) {
        const char *name = ds4_profiler_event_kind(ev[i].kind);
        if (ev[i].kind == DS4_PEV_STAGE && t->view &&
            ev[i].stage < (uint16_t)t->view->n_stages)
            name = t->view->stages[ev[i].stage].label;
        if (dwide)
            doc_line(d, ST_NORMAL, "  %8u %8.3fms  %-25.25s %5u %6u %3u %10u",
                     ev[i].t_us, (double)ev[i].dur_ns / 1e6,
                     name, ev[i].layer,
                     ev[i].expert, ev[i].device, ev[i].bytes);
        else
            doc_line(d, ST_NORMAL, "  %8u %6.2f %-10.10s %3u %3u %2u %7u",
                     ev[i].t_us, (double)ev[i].dur_ns / 1e6,
                     name, ev[i].layer,
                     ev[i].expert, ev[i].device, ev[i].bytes);
    }
}

static void build_devices(ds4_tui *t) {
    tui_doc *d = &t->prof;
    doc_clear(d);
    ds4_prof_view *v = t->view;
    doc_line(d, ST_TITLE, " Devices (nvtop style)");
    const double ssd_peak = 2e9;
    double series[64];
    const int n = t->h_n < 60 ? t->h_n : 60;
    for (int i = 0; i < v->dev_n; i++) {
        const ds4_prof_dev_state *dev = &v->dev[i];
        sbuf ub, mb;
        memset(&ub, 0, sizeof(ub));
        memset(&mb, 0, sizeof(mb));
        bar(&ub, dev->util_pct, 100, 28, "\x1b[32m");
        bar(&mb, dev->mem_total ? 100.0 * (double)dev->mem_used / (double)dev->mem_total : 0,
            100, 28, "\x1b[35m");
        doc_line(d, ST_TITLE, " GPU%d  %s", dev->device, dev->name);
        doc_line(d, ST_NORMAL, "   util %s %3u%%", ub.s ? ub.s : "", dev->util_pct);
    {
        char bu[24], bt[24];
        ds4_profiler_format_bytes(bu, sizeof(bu), dev->mem_used);
        ds4_profiler_format_bytes(bt, sizeof(bt), dev->mem_total);
        doc_line(d, ST_NORMAL, "   vram %s %s/%s", mb.s ? mb.s : "", bu, bt);
    }
        hist_series_u8(t, t->h_util[i], series, n);
        doc_printf(d, ST_GRAY, "   recent ");
        sbuf sp;
        memset(&sp, 0, sizeof(sp));
        spark(&sp, series, n, 100, "\x1b[36m");
        doc_puts(d, sp.s ? sp.s : "", ST_GRAY);
        free(sp.s);
        doc_nl(d);
    }
    doc_line(d, ST_TITLE, " Host");
    sbuf ram;
    memset(&ram, 0, sizeof(ram));
    const double ram_used = (double)v->host_ram_used;
    const double ram_total = ram_used + (double)v->host_ram_avail;
    bar(&ram, ram_used, ram_total > 0 ? ram_total : 1, 28, "\x1b[34m");
    {
        char bu[24], bt[24];
        ds4_profiler_format_bytes(bu, sizeof(bu), (uint64_t)ram_used);
        ds4_profiler_format_bytes(bt, sizeof(bt), (uint64_t)ram_total);
        doc_line(d, ST_NORMAL, "   RAM  %s %s/%s", ram.s ? ram.s : "", bu, bt);
    }
    free(ram.s);
    hist_series(t, t->h_ssd, series, n);
    doc_printf(d, ST_GRAY, "   SSD read recent ");
    sbuf sp;
    memset(&sp, 0, sizeof(sp));
    spark(&sp, series, n, ssd_peak, "\x1b[33m");
    doc_puts(d, sp.s ? sp.s : "", ST_GRAY);
    free(sp.s);
    doc_nl(d);
    {
        char bs[24];
        ds4_profiler_format_bytes(bs, sizeof(bs), v->ssd_read_bytes);
        doc_line(d, ST_GRAY, "   current SSD total %s", bs);
    }
}

static void build_experts(ds4_tui *t) {
    tui_doc *d = &t->prof;
    doc_clear(d);
    ds4_prof_view *v = t->view;
    const ds4_profiler_stats *st = &v->stats;
    const ds4_profiler_host_stats *hs = &v->host;
    uint64_t gpu_exp = 0;
    for (int i = 0; i < st->n_devices; i++) gpu_exp += st->dev[i].expert_count;
    const uint64_t ram_exp = st->host_arena_capacity_bytes ? st->host_arena_experts
                                                           : st->host_experts_resident;
    const uint64_t total = hs->total_experts;
    const uint64_t ssd = total > gpu_exp + ram_exp ? total - gpu_exp - ram_exp : 0;
    doc_line(d, ST_TITLE, " Expert residency and routing");
    {
        char be[24];
        ds4_profiler_format_bytes(be, sizeof(be), hs->expert_bytes);
        doc_line(d, ST_NORMAL, "  total %llu  GPU %llu  RAM %llu  SSD %llu  bytes/expert %s",
                 (unsigned long long)total, (unsigned long long)gpu_exp,
                 (unsigned long long)ram_exp, (unsigned long long)ssd, be);
    }
    for (int i = 0; i < st->n_devices; i++) {
        const ds4_profiler_device *dev = &st->dev[i];
        char bb[24];
        ds4_profiler_format_bytes(bb, sizeof(bb), dev->expert_bytes);
        doc_line(d, ST_NORMAL, "  tier %d  %-26.26s residents %u  bytes %s  type %s",
                 i, dev->name, dev->expert_count, bb,
                 dev->expert_type[0] ? dev->expert_type : "-");
    }
    if (st->host_arena_capacity_bytes) {
        char bu[24], bc[24];
        ds4_profiler_format_bytes(bu, sizeof(bu), st->host_arena_used_bytes);
        ds4_profiler_format_bytes(bc, sizeof(bc), st->host_arena_capacity_bytes);
        doc_line(d, ST_NORMAL,
                 "  host arena used %s/%s  experts %llu  hits %llu  misses %llu  admissions %llu  evictions %llu",
                 bu, bc,
                 (unsigned long long)st->host_arena_experts,
                 (unsigned long long)st->host_arena_hits,
                 (unsigned long long)st->host_arena_misses,
                 (unsigned long long)st->host_arena_admissions,
                 (unsigned long long)st->host_arena_evictions);
    }
    doc_line(d, ST_TITLE, " Routing in the selected token");
    const ds4_prof_record_meta *m = sel_meta(t);
    if (!m) {
        doc_line(d, ST_GRAY, "  select a token in view 2");
        return;
    }
    ds4_prof_expert_trace *ex = m->experts ? malloc((size_t)m->experts * sizeof(*ex)) : NULL;
    const int exn = ex ? ds4_profiler_record_experts(m->seq, ex, (int)m->experts) : 0;
    doc_line(d, ST_GRAY, "  %s #%llu, %d expert uses", kind_short(m->kind),
             (unsigned long long)m->seq, exn);
    for (int i = 0; i < exn; i++) {
        static const char *src[] = {"VRAM", "host RAM", "SSD", "CPU"};
        char bb[24];
        ds4_profiler_format_bytes(bb, sizeof(bb), ex[i].bytes);
        doc_line(d, ST_NORMAL, "  layer %3u expert %3u  %-8s  tier %u -> %u  %s",
                 (unsigned)ex[i].layer, (unsigned)ex[i].expert,
                 ex[i].source < 4 ? src[ex[i].source] : "?",
                 ex[i].src_gpu, ex[i].dest_gpu, bb);
    }
    free(ex);
}

static void build_profiler(ds4_tui *t) {
    switch (t->pv) {
    case PV_TOKENS: build_tokens(t); break;
    case PV_DETAIL: build_detail(t); break;
    case PV_DEVICES: build_devices(t); break;
    case PV_EXPERTS: build_experts(t); break;
    default: build_overview(t); break;
    }
}

/* The help card next to the console. Everything that used to be a hint or an
 * explanation inside the profiler pane lives here instead: the pane needs its
 * columns for numbers, and this card can follow whichever view is on screen.
 * It is rebuilt per frame, which is cheap because it is a few dozen lines. */
static void build_help(ds4_tui *t) {
    tui_doc *d = &t->help;
    doc_clear(d);
    doc_line(d, ST_TITLE, " profiler: %s", view_name(t->pv));
    switch (t->pv) {
    case PV_TOKENS:
        doc_line(d, ST_GRAY,
                 " One record per decoded token (D) or prefill chunk (P), newest");
        doc_line(d, ST_GRAY,
                 " first. The list is frozen while generating so it cannot slide");
        doc_line(d, ST_GRAY,
                 " under the reader; u loads what arrived since. Hot stage is the");
        doc_line(d, ST_GRAY,
                 " largest measured stage of the selected record, not a");
        doc_line(d, ST_GRAY, " measurement of its own.");
        if (t->view) {
            const ds4_prof_record_meta *m = sel_meta(t);
            if (m && m->hot_stage < (uint16_t)t->view->n_stages)
                doc_line(d, ST_NORMAL, " %s = %s",
                         t->view->stages[m->hot_stage].label,
                         t->view->stages[m->hot_stage].hint);
        }
        break;
    case PV_DETAIL:
        doc_line(d, ST_GRAY,
                 " total = CPU launch time + device wait, so a stage never hides");
        doc_line(d, ST_GRAY,
                 " what its own launch cost. Stage times are per-device wall");
        doc_line(d, ST_GRAY,
                 " clock: one record runs on two GPUs at once, so their sum can");
        doc_line(d, ST_GRAY,
                 " exceed the record wall. The residual is sampling, detokenising");
        doc_line(d, ST_GRAY,
                 " and the gaps between stages. Expert uses show where a weight");
        doc_line(d, ST_GRAY, " lived (source) and the device that ran it (from -> to).");
        break;
    case PV_DEVICES:
        doc_line(d, ST_GRAY,
                 " Bars are the current value, the spark line the last 60 samples");
        doc_line(d, ST_GRAY,
                 " and the whole row the last 24. VRAM and RAM are used of total");
        doc_line(d, ST_GRAY,
                 " on that device and on the host; the SSD spark is scaled to");
        doc_line(d, ST_GRAY, " 2 GB/s so a slow disk still shows its shape.");
        break;
    case PV_EXPERTS:
        doc_line(d, ST_GRAY,
                 " Every expert lives in exactly one tier: VRAM on a GPU, the");
        doc_line(d, ST_GRAY,
                 " pinned host arena in RAM, or the SSD. Routing lists, for the");
        doc_line(d, ST_GRAY,
                 " selected token, where each expert was read from and where it");
        doc_line(d, ST_GRAY,
                 " ran. Arena hits were served from RAM, admissions are misses");
        doc_line(d, ST_GRAY,
                 " that took an empty slot, evictions reclaim a slot when full.");
        break;
    default:
        doc_line(d, ST_GRAY,
                 " D/P are averages since the last reset (r); ms/token is the whole");
        doc_line(d, ST_GRAY,
                 " decode record, so that is the latency that matters. Stage rows");
        doc_line(d, ST_GRAY,
                 " are per-device wall clock of one kind only: the two GPUs overlap");
        doc_line(d, ST_GRAY,
                 " inside one token, so their sum can pass the wall and each share");
        doc_line(d, ST_GRAY,
                 " is of that kind's own total. Capture (p) serialises the GPU, so");
        doc_line(d, ST_GRAY, " tok/s reads low while it is on.");
        break;
    }
    doc_line(d, ST_NORMAL, "");
    doc_line(d, ST_TITLE, " keys");
    doc_line(d, ST_GRAY, " profiler keys need the profiler pane focused (Alt-2).");
    doc_line(d, ST_NORMAL, " Tab / Shift-Tab   cycle the four panes");
    doc_line(d, ST_NORMAL, " Alt-1 .. Alt-4    chat, profiler, console, help");
    doc_line(d, ST_NORMAL, " Up / Down         move the row cursor");
    doc_line(d, ST_NORMAL, " PgUp / PgDn       scroll the pane; the cursor never blocks it");
    doc_line(d, ST_NORMAL, " Home / End        cursor to the first / last row");
    doc_line(d, ST_NORMAL, " 1 .. 5            overview, tokens, detail, devices, experts");
    doc_line(d, ST_NORMAL, " Enter             open the selected token or chunk");
    doc_line(d, ST_NORMAL, " Esc               detail back to the list; leave help/console");
    doc_line(d, ST_NORMAL, " f                 record filter: decode, prefill, all");
    doc_line(d, ST_NORMAL, " [ / ]             newer / older page of records");
    doc_line(d, ST_NORMAL, " u                 load records created while generating");
    doc_line(d, ST_NORMAL, " g / G             jump to the start / end");
    doc_line(d, ST_NORMAL, " p                 stage capture on / off");
    doc_line(d, ST_NORMAL, " r                 reset the profiler aggregates");
    doc_line(d, ST_NORMAL, " Ctrl-B            breakpoints: pause after each token/chunk");
    doc_line(d, ST_NORMAL, " Space / c / b     continue; b also drops breakpoints");
    doc_line(d, ST_NORMAL, " ?                 this card, from any pane");
    doc_line(d, ST_NORMAL, " Ctrl-Y            copy the focused pane to the clipboard");
    doc_line(d, ST_NORMAL, " chat: Enter sends, Ctrl-P/N move in the transcript,");
    doc_line(d, ST_NORMAL, " Ctrl-U pages it, Ctrl-C clears the line, Ctrl-D exits.");
    doc_line(d, ST_GRAY, " Chat takes bracketed pastes: multi-line text arrives as");
    doc_line(d, ST_GRAY, " one line, so a paste never runs the UI's keys.");
    doc_line(d, ST_GRAY, " A row wider than its pane slides sideways while it is the");
    doc_line(d, ST_GRAY, " selected row, and returns to column 0 when it is not.");
    if (t->pv == PV_DETAIL && t->view && t->detail_ready && t->detail.stage_n) {
        doc_line(d, ST_NORMAL, "");
        doc_line(d, ST_TITLE, " what each recorded operation means");
        for (uint32_t i = 0; i < t->detail.stage_n; i++) {
            const int id = t->detail.stages[i].id;
            if (id >= t->view->n_stages) continue;
            doc_line(d, ST_GRAY, "  %s: %s", t->view->stages[id].label,
                     t->view->stages[id].hint);
        }
    }
}

static void prof_refresh(ds4_tui *t, int force) {
    const double now = tui_now();
    if (!force && now - t->last_refresh < 0.50) return;
    t->last_refresh = now;
    if (!t->view) t->view = calloc(1, sizeof(*t->view));
    if (!t->view) return;
    ds4_profiler_view(t->view);
    list_refresh(t, 0);
    hist_push(t, t->view);
}

/* ----------------------------------------------------------------- frame -- */

static void drain_stderr(ds4_tui *t) {
    if (t->pipe_r < 0) return;
    char buf[4096];
    for (;;) {
        ssize_t n = read(t->pipe_r, buf, sizeof(buf) - 1);
        if (n <= 0) break;
        buf[n] = '\0';
        char *line = buf;
        for (char *p = buf; *p; p++) {
            if (*p == '\r') *p = '\n';
        }
        /* Add the whole chunk; append keeps embedded newlines. */
        (void)line;
        ds4_tui_log(t, 0, "%s", buf);
    }
}

void ds4_tui_redraw(ds4_tui *t) {
    if (!t || !t->active) return;
    term_size(t);
    const int rows = t->rows, cols = t->cols;
    screen_resize(&t->scr, rows, cols);
    screen_clear(&t->scr);
    if (rows < 16 || cols < 60) {
        screen_puts(&t->scr, 0, 0, ST_YELLOW,
                    "terminal too small: need at least 60x16", 38);
        screen_flush(&t->scr, 0, 0, 0);
        t->dirty = 0;
        t->last_draw = tui_now();
        return;
    }
    /* Top: chat over the left half, profiler over the right half. Bottom: the
     * console and the help card, aligned under those same two halves and each
     * in its own box. The bottom strip is about a third shorter than the old
     * single messages pane, so the panes that carry the numbers get the rows;
     * the help text wraps and scrolls, it does not need them all. */
    int top_h = (rows - 2) * 11 / 15;
    if (top_h < 8) top_h = 8;
    if (top_h > rows - 6) top_h = rows - 6;
    int msg_h = rows - 2 - top_h;
    if (msg_h < 4) msg_h = 4;
    const int chat_w = cols / 2;
    const int prof_x = chat_w + 1;
    const int prof_w = cols - chat_w - 1;
    const int msg_y = top_h;
    t->prof_rows = top_h - 2;
    t->prof_cols = prof_w - 2;

    prof_refresh(t, 0);
    build_profiler(t);
    build_help(t);

    draw_box_top(&t->scr, 0, 0, chat_w, "chat  [1]", t->focus == DS4_TUI_CHAT);
    char ptitle[80];
    snprintf(ptitle, sizeof(ptitle), "profiler [2] %s%s%s%s", view_name(t->pv),
             t->granular ? " *" : "",
             t->breakpoints ? " BP" : "",
             t->new_count ? " +" : "");
    draw_box_top(&t->scr, prof_x, 0, prof_w, ptitle, t->focus == DS4_TUI_PROFILER);
    draw_box_side(&t->scr, 0, 1, top_h - 2);
    draw_box_side(&t->scr, chat_w - 1, 1, top_h - 2);
    draw_box_side(&t->scr, prof_x, 1, top_h - 2);
    draw_box_side(&t->scr, cols - 1, 1, top_h - 2);
    draw_box_bottom(&t->scr, 0, top_h - 1, chat_w, t->focus == DS4_TUI_CHAT);
    draw_box_bottom(&t->scr, prof_x, top_h - 1, prof_w, t->focus == DS4_TUI_PROFILER);

    /* Only the focused pane shows a cursor; the others keep their scroll
     * position and stay plain so the eye knows where the keys act. */
    const double now = tui_now();
    const int reveal_cur = t->reveal_cur;
    t->reveal_cur = 0;
    int prof_cur = -1;
    if (t->pv == PV_TOKENS && t->sel_off_valid)
        prof_cur = doc_visual_line(&t->prof, prof_w - 2, t->sel_off);
    else if (t->pv != PV_TOKENS)
        prof_cur = t->cur_prof;
    t->chat_cols = chat_w - 2;
    t->msg_cols = chat_w - 2;
    t->help_cols = prof_w - 2;
    t->slide = 0;
    int chat_slide = 0, prof_slide = 0, msg_slide = 0, help_slide = 0;
    pane_view chat_pv = { &t->chat_top, &t->chat_follow,
                          t->focus == DS4_TUI_CHAT ? &t->cur_chat : NULL,
                          &t->hoff, &t->hoff_t, &chat_slide, now,
                          t->focus == DS4_TUI_CHAT && reveal_cur };
    /* In the token list the selection is the cursor and is always brought into
     * view; that view pages instead of scrolling freely, so it asks for the
     * reveal every frame and never drags the selection along with a scroll. */
    pane_view prof_pv = { &t->prof_scroll, NULL,
                          t->focus == DS4_TUI_PROFILER ? &prof_cur : NULL,
                          &t->hoff, &t->hoff_t, &prof_slide, now,
                          t->focus == DS4_TUI_PROFILER &&
                              (t->pv == PV_TOKENS || reveal_cur) };
    draw_doc(&t->scr, &t->chat, 1, 1, chat_w - 2, top_h - 2, &chat_pv);
    draw_doc(&t->scr, &t->prof, prof_x + 1, 1, prof_w - 2, top_h - 2, &prof_pv);
    if (t->focus == DS4_TUI_PROFILER && prof_cur >= 0) t->cur_prof = prof_cur;

    draw_box_top(&t->scr, 0, msg_y, chat_w, "console  [3]",
                 t->focus == DS4_TUI_MESSAGES);
    draw_box_top(&t->scr, prof_x, msg_y, prof_w, "help  [4]",
                 t->focus == DS4_TUI_HELP);
    draw_box_side(&t->scr, 0, msg_y + 1, msg_h - 2);
    draw_box_side(&t->scr, chat_w - 1, msg_y + 1, msg_h - 2);
    draw_box_side(&t->scr, prof_x, msg_y + 1, msg_h - 2);
    draw_box_side(&t->scr, cols - 1, msg_y + 1, msg_h - 2);
    draw_box_bottom(&t->scr, 0, msg_y + msg_h - 1, chat_w,
                    t->focus == DS4_TUI_MESSAGES);
    draw_box_bottom(&t->scr, prof_x, msg_y + msg_h - 1, prof_w,
                    t->focus == DS4_TUI_HELP);
    pane_view msg_pv = { &t->msg_top, &t->msg_follow,
                         t->focus == DS4_TUI_MESSAGES ? &t->cur_msg : NULL,
                         &t->hoff, &t->hoff_t, &msg_slide, now,
                         t->focus == DS4_TUI_MESSAGES && reveal_cur };
    pane_view help_pv = { &t->help_top, NULL,
                          t->focus == DS4_TUI_HELP ? &t->cur_help : NULL,
                          &t->hoff, &t->hoff_t, &help_slide, now,
                          t->focus == DS4_TUI_HELP && reveal_cur };
    draw_doc(&t->scr, &t->msg, 1, msg_y + 1, chat_w - 2, msg_h - 2, &msg_pv);
    draw_doc(&t->scr, &t->help, prof_x + 1, msg_y + 1, prof_w - 2, msg_h - 2,
             &help_pv);
    t->slide = chat_slide || prof_slide || msg_slide || help_slide;

    /* Input + status share one stable row; the hint row never changes. */
    const int input_y = rows - 2;
    int cursor_col = 0;
    if (!t->busy && t->focus == DS4_TUI_CHAT) {
        const char *prompt = t->prompt[0] ? t->prompt : "ds4> ";
        screen_puts(&t->scr, input_y, 0, ST_CYAN | CS_BOLD, prompt, strlen(prompt));
        int avail = cols - (int)strlen(prompt) - 1;
        if (avail < 8) avail = 8;
        const char *start = t->input ? t->input : "";
        int skip = 0;
        if (t->input_pos > (size_t)avail) skip = (int)t->input_pos - avail;
        if (skip > 0) start += skip;
        screen_fit(&t->scr, input_y, (int)strlen(prompt), ST_NORMAL, start, avail);
        cursor_col = 1 + (int)strlen(prompt) + (int)t->input_pos - skip;
        if (cursor_col > cols) cursor_col = cols;
    } else if (t->busy) {
        screen_puts(&t->scr, input_y, 0, ST_YELLOW | CS_BOLD,
                    t->status[0] ? t->status : "working", strlen(t->status[0] ? t->status : "working"));
    } else {
        screen_puts(&t->scr, input_y, 0, ST_GRAY,
                    "(chat pane not focused - Alt-1 or Tab returns)", 45);
    }
    if (t->status[0] && !t->busy && t->focus == DS4_TUI_CHAT) {
        int sl = (int)strlen(t->status);
        if (sl < cols - 20)
            screen_puts(&t->scr, input_y, cols - sl, ST_YELLOW, t->status, (size_t)sl);
    }

    screen_printf(&t->scr, rows - 1, 0, ST_GRAY,
                  " Tab/Shift-Tab pane | PgUp/PgDn scroll | 1-5 views | Ctrl-B BP %s | p capture | ? help",
                  t->breakpoints ? "ON" : "off");

    screen_flush(&t->scr, input_y + 1,
                 cursor_col > 0 ? cursor_col : 0,
                 cursor_col > 0);
    t->dirty = 0;
    t->last_draw = tui_now();
}

/* --------------------------------------------------------------- history -- */

static void hist_file(char *buf, size_t cap) {
    const char *h = getenv("HOME");
    snprintf(buf, cap, "%s/.ds4_history", h && h[0] ? h : ".");
}

static void tui_hist_load(ds4_tui *t) {
    char path[512];
    hist_file(path, sizeof(path));
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    char line[4096];
    while (fgets(line, sizeof(line), fp) && t->hist_n < TUI_HIST) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n) t->history[t->hist_n++] = strdup(line);
    }
    fclose(fp);
}

static void tui_hist_add(ds4_tui *t, const char *s) {
    if (!s || !s[0]) return;
    if (t->hist_n > 0 && !strcmp(t->history[t->hist_n - 1], s)) return;
    if (t->hist_n == TUI_HIST) {
        free(t->history[0]);
        memmove(t->history, t->history + 1, sizeof(t->history[0]) * (TUI_HIST - 1));
        t->hist_n--;
    }
    t->history[t->hist_n++] = strdup(s);
    char path[512];
    hist_file(path, sizeof(path));
    FILE *fp = fopen(path, "a");
    if (fp) { fprintf(fp, "%s\n", s); fclose(fp); }
}

/* ----------------------------------------------------------------- input -- */

static void input_ensure(ds4_tui *t, size_t n) {
    if (n + 1 <= t->input_cap) return;
    size_t cap = t->input_cap ? t->input_cap : 256;
    while (cap < n + 1) cap *= 2;
    char *p = realloc(t->input, cap);
    if (!p) return;
    t->input = p;
    t->input_cap = cap;
}

static void input_insert(ds4_tui *t, const char *s, size_t n) {
    input_ensure(t, t->input_len + n);
    if (t->input_len + n + 1 > t->input_cap) return;
    memmove(t->input + t->input_pos + n, t->input + t->input_pos,
            t->input_len - t->input_pos + 1);
    memcpy(t->input + t->input_pos, s, n);
    t->input_len += n;
    t->input_pos += n;
    t->dirty = 1;
}

static void input_backspace(ds4_tui *t) {
    if (t->input_pos == 0) return;
    memmove(t->input + t->input_pos - 1, t->input + t->input_pos,
            t->input_len - t->input_pos + 1);
    t->input_pos--;
    t->input_len--;
    t->dirty = 1;
}

static void input_delete(ds4_tui *t) {
    if (t->input_pos >= t->input_len) return;
    memmove(t->input + t->input_pos, t->input + t->input_pos + 1,
            t->input_len - t->input_pos);
    t->input_len--;
    t->dirty = 1;
}

static void input_set(ds4_tui *t, const char *s) {
    size_t n = s ? strlen(s) : 0;
    input_ensure(t, n);
    if (n + 1 <= t->input_cap && s) memcpy(t->input, s, n + 1);
    else if (t->input_cap) t->input[0] = '\0';
    t->input_len = n;
    t->input_pos = n;
    t->dirty = 1;
}

static int read_byte(ds4_tui *t) {
    unsigned char c;
    (void)t;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n == 1) return c;
    return -1;
}

/* Escape sequences and bracketed pastes arrive as one write, but stdin is
 * nonblocking: without the wait, a lone ESC byte would split ESC[200~ into
 * ESC followed by typed text. The wait is short enough to keep the key path
 * responsive. */
static int read_byte_wait(ds4_tui *t, int ms) {
    for (int waited = 0; waited <= ms; waited++) {
        int c = read_byte(t);
        if (c >= 0) return c;
        struct timespec ts = { 0, 1000000L };
        nanosleep(&ts, NULL);
    }
    return -1;
}

static tui_key read_key(ds4_tui *t, char *utf8, size_t *ulen) {
    int c = read_byte(t);
    *ulen = 0;
    if (c < 0) return K_NONE;
    if (c == 3) return K_CTRL_C;
    if (c == 4) return K_CTRL_D;
    if (c == 1) return K_CTRL_A;
    if (c == 2) return K_CTRL_B;
    if (c == 5) return K_CTRL_E;
    if (c == 11) return K_CTRL_K;
    if (c == 12) return K_CTRL_L;
    if (c == 21) return K_CTRL_U;
    if (c == 23) return K_CTRL_W;
    if (c == 14) return K_CTRL_N;
    if (c == 16) return K_CTRL_P;
    if (c == 25) return K_CTRL_Y;
    if (c == 9) return K_TAB;
    if (c == 13 || c == 10) return K_ENTER;
    if (c == 127 || c == 8) return K_BACKSPACE;
    if (c == 27) {
        int c2 = read_byte_wait(t, 25);
        if (c2 < 0) return K_ESC;
        if (c2 == '[') {
            int c3 = read_byte_wait(t, 25);
            if (c3 < 0) return K_ESC;
            if (c3 == 'A') return K_UP;
            if (c3 == 'B') return K_DOWN;
            if (c3 == 'C') return K_RIGHT;
            if (c3 == 'D') return K_LEFT;
            if (c3 == 'H') return K_HOME;
            if (c3 == 'F') return K_END;
            if (c3 == 'Z') return K_STAB;
            if (c3 >= '0' && c3 <= '9') {
                int c4 = read_byte_wait(t, 25);
                if (c4 == '~') {
                    if (c3 == '1' || c3 == '7') return K_HOME;
                    if (c3 == '3') return K_DELETE;
                    if (c3 == '4' || c3 == '8') return K_END;
                    if (c3 == '5') return K_PGUP;
                    if (c3 == '6') return K_PGDN;
                } else if (c3 == '2' && c4 == '0') {
                    /* ESC [ 2 0 0 ~ starts a paste, ESC [ 2 0 1 ~ ends one;
                     * a stray end marker is dropped. */
                    int c5 = read_byte_wait(t, 25);
                    if (c5 == '0' || c5 == '1') {
                        if (read_byte_wait(t, 25) == '~')
                            return c5 == '0' ? K_PASTE : K_NONE;
                    }
                    return K_NONE;
                } else if (c4 == ';') {
                    /* Modifier sequences: ESC [ 1 ; 5 A etc. Consume the
                     * final byte and treat as the base key. */
                    int c5 = read_byte_wait(t, 25);
                    (void)c5;
                }
            }
            return K_NONE;
        }
        if (c2 == '1') return K_ALT_1;
        if (c2 == '2') return K_ALT_2;
        if (c2 == '3') return K_ALT_3;
        if (c2 == '4') return K_ALT_4;
        return K_ESC;
    }
    utf8[0] = (char)c;
    *ulen = 1;
    if (c >= 0x80) {
        int need = 0;
        if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c >= 0xE0 && c <= 0xEF) need = 2;
        else if (c >= 0xF0 && c <= 0xF4) need = 3;
        for (int i = 0; i < need && *ulen < 8; i++) {
            int r = read_byte(t);
            if (r < 0) break;
            utf8[(*ulen)++] = (char)r;
        }
    }
    return K_CHAR;
}

static void scroll_focus(ds4_tui *t, int delta) {
    if (t->focus == DS4_TUI_CHAT) {
        t->chat_follow = 0;
        t->chat_top += delta;
        if (t->chat_top < 0) t->chat_top = 0;
    } else if (t->focus == DS4_TUI_MESSAGES) {
        t->msg_follow = 0;
        t->msg_top += delta;
        if (t->msg_top < 0) t->msg_top = 0;
    } else if (t->focus == DS4_TUI_HELP) {
        t->help_top += delta;
        if (t->help_top < 0) t->help_top = 0;
    } else {
        t->prof_scroll += delta;
        if (t->prof_scroll < 0) t->prof_scroll = 0;
    }
    t->dirty = 1;
}

static void prof_select(ds4_tui *t, int delta) {
    if (t->list_n <= 0) return;
    if (!t->have_sel) { t->sel = t->list_n - 1; t->have_sel = 1; }
    t->sel += delta;
    if (t->sel < 0) t->sel = 0;
    if (t->sel >= t->list_n) t->sel = t->list_n - 1;
    t->sel_seq = t->list[t->sel].seq;
    if (t->pv == PV_TOKENS) {
        const int ps = token_page_size(t);
        t->page = (t->list_n - 1 - t->sel) / ps;
    }
    t->dirty = 1;
}

static void prof_page_move(ds4_tui *t, int older) {
    if (t->list_n <= 0) return;
    const int ps = token_page_size(t);
    const int pages = (t->list_n + ps - 1) / ps;
    t->page += older ? 1 : -1;
    if (t->page < 0) t->page = 0;
    if (t->page >= pages) t->page = pages - 1;
    /* Keep the selection on the page the user is reading. */
    const int end = t->list_n - 1 - t->page * ps;
    t->sel = end < 0 ? 0 : end;
    t->sel_seq = t->list[t->sel].seq;
    t->have_sel = 1;
    t->dirty = 1;
}

/* The focused pane's document, cursor and width. The profiler in the tokens
 * view has no free cursor: its rows are records and Up/Down pick one, so the
 * highlight follows the selection instead. */
static tui_doc *cur_doc(ds4_tui *t) {
    switch (t->focus) {
    case DS4_TUI_PROFILER: return &t->prof;
    case DS4_TUI_MESSAGES: return &t->msg;
    case DS4_TUI_HELP: return &t->help;
    default: return &t->chat;
    }
}

static int *cur_line(ds4_tui *t) {
    switch (t->focus) {
    case DS4_TUI_PROFILER: return &t->cur_prof;
    case DS4_TUI_MESSAGES: return &t->cur_msg;
    case DS4_TUI_HELP: return &t->cur_help;
    default: return &t->cur_chat;
    }
}

static int cur_cols(const ds4_tui *t) {
    switch (t->focus) {
    case DS4_TUI_PROFILER: return t->prof_cols;
    case DS4_TUI_MESSAGES: return t->msg_cols;
    case DS4_TUI_HELP: return t->help_cols;
    default: return t->chat_cols;
    }
}

/* In the tokens view the record selection is the cursor, so the arrow keys
 * keep their meaning there and only the highlight scrolls sideways. */
static int cur_visible(ds4_tui *t) {
    return !(t->focus == DS4_TUI_PROFILER && t->pv == PV_TOKENS);
}

static void cursor_move(ds4_tui *t, int delta) {
    if (!cur_visible(t)) return;
    tui_doc *d = cur_doc(t);
    int *c = cur_line(t);
    const int w = cur_cols(t);
    if (!d || w <= 0) return;
    int n = 0;
    (void)doc_wrap(d, w, &n);
    if (n <= 0) { *c = -1; return; }
    if (*c < 0) *c = delta < 0 ? n - 1 : 0;
    else *c += delta;
    if (*c < 0) *c = 0;
    if (*c > n - 1) *c = n - 1;
    t->reveal_cur = 1;
    t->hoff = 0;
    t->hoff_t = 0;
    t->dirty = 1;
}

static void cursor_edge(ds4_tui *t, int last) {
    if (!cur_visible(t)) return;
    tui_doc *d = cur_doc(t);
    int *c = cur_line(t);
    const int w = cur_cols(t);
    if (!d || w <= 0) return;
    int n = 0;
    (void)doc_wrap(d, w, &n);
    *c = n > 0 ? (last ? n - 1 : 0) : -1;
    t->reveal_cur = 1;
    t->hoff = 0;
    t->hoff_t = 0;
    t->dirty = 1;
}

/* A paste is text, never keystrokes: newlines inside it become spaces, so a
 * multi-line paste lands on one chat line instead of submitting on its first
 * newline and spraying the rest into the next prompt. Control bytes are
 * dropped. */
static void paste_flush(ds4_tui *t) {
    t->pasting = 0;
    if (t->paste_len == 0) {
        free(t->paste);
        t->paste = NULL;
        t->paste_cap = 0;
        return;
    }
    sbuf text;
    memset(&text, 0, sizeof(text));
    for (size_t i = 0; i < t->paste_len; i++) {
        unsigned char c = (unsigned char)t->paste[i];
        if (c == '\r' || c == '\n' || c == '\t') { sb_add(&text, " ", 1); continue; }
        if (c < 0x20 || c == 0x7f) continue;
        sb_add(&text, (const char *)&c, 1);
    }
    if (text.len == 0) {
        ds4_tui_log(t, 0, "paste held no text (%zu bytes dropped)", t->paste_len);
    } else if (t->busy || (t->focus != DS4_TUI_CHAT && !t->readline)) {
        ds4_tui_log(t, 1, "paste of %zu bytes ignored: chat pane is not taking input",
                    text.len);
    } else {
        input_insert(t, text.s, text.len);
        ds4_tui_log(t, 0, "pasted %zu bytes into the chat line", text.len);
    }
    free(text.s);
    free(t->paste);
    t->paste = NULL;
    t->paste_len = 0;
    t->paste_cap = 0;
    t->dirty = 1;
}

/* One byte of a bracketed paste; returns 0 when the input has nothing more to
 * give right now, so a large paste can span several pumps. */
static int paste_step(ds4_tui *t) {
    int c = read_byte_wait(t, 5);
    if (c < 0) return 0;
    if (t->paste_len + 8 > t->paste_cap) {
        size_t cap = t->paste_cap ? t->paste_cap * 2 : 4096;
        char *p = realloc(t->paste, cap);
        if (!p) { t->pasting = 0; return 0; }
        t->paste = p;
        t->paste_cap = cap;
    }
    t->paste[t->paste_len++] = (char)c;
    if (t->paste_len > 262144) {          /* a paste, not a stream */
        paste_flush(t);
        return 0;
    }
    static const char end[] = "\x1b[201~";
    const size_t elen = sizeof(end) - 1;
    if (t->paste_len >= elen && memcmp(t->paste + t->paste_len - elen, end, elen) == 0) {
        t->paste_len -= elen;
        paste_flush(t);
    }
    return 1;
}

static void b64_add(sbuf *b, const char *s, size_t n) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = (unsigned)(unsigned char)s[i] << 16;
        if (i + 1 < n) v |= (unsigned)(unsigned char)s[i + 1] << 8;
        if (i + 2 < n) v |= (unsigned char)s[i + 2];
        char out[4];
        out[0] = tab[(v >> 18) & 63];
        out[1] = tab[(v >> 12) & 63];
        out[2] = (char)(i + 1 < n ? tab[(v >> 6) & 63] : '=');
        out[3] = (char)(i + 2 < n ? tab[v & 63] : '=');
        sb_add(b, out, 4);
    }
}

/* Ctrl-Y hands the focused pane to the terminal clipboard over OSC 52. Mouse
 * selection only ever sees the cells currently on screen and loses them as the
 * panes repaint, and the alternate screen never reaches the scrollback, so
 * this is the way to take text out of the UI (tmux needs set-clipboard on). */
static void copy_focused(ds4_tui *t) {
    const char *name = t->focus == DS4_TUI_CHAT ? "chat" :
                       t->focus == DS4_TUI_PROFILER ? "profiler" :
                       t->focus == DS4_TUI_MESSAGES ? "console" : "help";
    sbuf text;
    memset(&text, 0, sizeof(text));
    if (t->focus == DS4_TUI_PROFILER) {
        /* The profiler scrolls a wrapped doc, so copy what is on screen. */
        int n = 0;
        const int w = t->prof_cols > 0 ? t->prof_cols : 1;
        const vis_line *lines = doc_wrap(&t->prof, w, &n);
        int first = t->prof_scroll;
        if (first > n - 1) first = n - 1;
        if (first < 0) first = 0;
        for (int i = first; i < first + t->prof_rows && i < n; i++) {
            sb_add(&text, t->prof.s + lines[i].off, lines[i].len);
            sb_add(&text, "\n", 1);
        }
    } else {
        tui_doc *d = t->focus == DS4_TUI_CHAT ? &t->chat :
                     t->focus == DS4_TUI_MESSAGES ? &t->msg : &t->help;
        size_t take = d->len < 131072 ? d->len : 131072;   /* OSC 52 is small */
        if (take) sb_add(&text, d->s, take);
    }
    if (text.len == 0) {
        ds4_tui_log(t, 1, "nothing to copy in the %s pane", name);
        free(text.s);
        return;
    }
    size_t n = text.len;
    int cut = 0;
    if (n > 131072) { n = 131072; cut = 1; }   /* OSC 52 payloads have limits */
    sbuf b64;
    memset(&b64, 0, sizeof(b64));
    b64_add(&b64, text.s, n);
    fputs("\x1b]52;c;", stdout);
    fwrite(b64.s, 1, b64.len, stdout);
    fputc('\a', stdout);
    fflush(stdout);
    ds4_tui_log(t, 0, "copied %s pane to the clipboard (%zu bytes%s)", name, n,
                cut ? ", truncated" : "");
    ds4_tui_status(t, "copied %s to the clipboard", name);
    free(b64.s);
    free(text.s);
}

static void handle_key(ds4_tui *t, tui_key k, const char *utf8, size_t ulen) {
    /* Global pane switching. */
    if (k == K_TAB || k == K_STAB) {
        int f = (int)t->focus + (k == K_TAB ? 1 : -1);
        if (f < 0) f = DS4_TUI_PANE_COUNT - 1;
        if (f >= DS4_TUI_PANE_COUNT) f = 0;
        t->focus = (ds4_tui_focus)f;
        t->hoff = 0;
        t->hoff_t = 0;
        t->dirty = 1;
        return;
    }
    if (k == K_ALT_1) { t->focus = DS4_TUI_CHAT; t->dirty = 1; return; }
    if (k == K_ALT_2) { t->focus = DS4_TUI_PROFILER; t->dirty = 1; return; }
    if (k == K_ALT_3) { t->focus = DS4_TUI_MESSAGES; t->dirty = 1; return; }
    if (k == K_ALT_4) { t->focus = DS4_TUI_HELP; t->dirty = 1; return; }
    if (k == K_CTRL_L) { t->dirty = 1; return; }
    if (k == K_CTRL_Y) { copy_focused(t); return; }
    if (k == K_CTRL_B) {
        t->breakpoints = !t->breakpoints;
        if (t->paused) t->paused = 0;
        ds4_tui_status(t, t->breakpoints
                       ? "breakpoints on: pauses after each token/chunk"
                       : "breakpoints off");
        ds4_tui_log(t, 0, "breakpoints %s", t->breakpoints ? "on" : "off");
        return;
    }
    if (t->paused && k == K_CHAR && ulen == 1) {
        char c = utf8[0];
        if (c == ' ' || c == 'c') { t->paused = 0; t->status[0] = '\0'; t->dirty = 1; return; }
        if (c == 'b') {
            t->breakpoints = 0;
            t->paused = 0;
            t->status[0] = '\0';
            t->dirty = 1;
            return;
        }
    }

    if (t->focus == DS4_TUI_PROFILER) {
        if (k == K_LEFT || k == K_ESC) {
            if (t->pv == PV_DETAIL) { t->pv = PV_TOKENS; t->dirty = 1; }
            return;
        }
        if (k == K_ENTER) {
            if (t->pv == PV_TOKENS) {
                const ds4_prof_record_meta *m = sel_meta(t);
                if (m) detail_open(t, m->seq);
            }
            return;
        }
        if (k >= K_CHAR && ulen == 1) {
            char c = utf8[0];
            if (c >= '1' && c <= '5') {
                t->pv = (prof_view)(c - '1');
                t->prof_scroll = 0;
                t->cur_prof = -1;
                t->hoff = 0;
                t->hoff_t = 0;
                t->cur_help = -1;   /* the card now explains another view */
                t->help_top = 0;
                if (t->pv == PV_TOKENS) {
                    t->list_frozen = 1;
                    list_refresh(t, 1);
                    const int ps = token_page_size(t);
                    if (t->list_n > 0 && t->have_sel)
                        t->page = (t->list_n - 1 - t->sel) / ps;
                    else
                        t->page = 0;
                } else if (t->pv == PV_DETAIL) {
                    const ds4_prof_record_meta *m = sel_meta(t);
                    if (m) detail_open(t, m->seq);
                }
                t->dirty = 1;
                return;
            }
            if (c == 'f') {
                t->kind_filter = (t->kind_filter + 1) % 3;
                t->list_n = 0;
                t->list_frozen = 1;
                list_refresh(t, 1);
                t->page = 0;
                t->prof_scroll = 0;
                t->cur_prof = -1;
                t->cur_help = -1;
                t->help_top = 0;
                ds4_tui_log(t, 0, "records filter: %s only (%d shown)",
                            kind_filter_name(t), t->list_n);
                t->dirty = 1;
                return;
            }
            if (c == 'u') {
                t->list_frozen = 1;
                list_refresh(t, 1);
                ds4_tui_log(t, 0, "record list refreshed (%d entries)", t->list_n);
                t->dirty = 1;
                return;
            }
            if (c == '[') { prof_page_move(t, 1); return; }
            if (c == ']') { prof_page_move(t, -1); return; }
            if (c == 'p') {
                ds4_tui_set_granular(t, !t->granular);
                ds4_tui_log(t, 1, "stage capture %s", t->granular ? "on" : "off");
                t->dirty = 1;
                return;
            }
            if (c == 'r') {
                ds4_profiler_reset_timing();
                t->h_n = 0;
                t->h_head = 0;
                ds4_tui_log(t, 0, "profiler aggregates reset");
                t->dirty = 1;
                return;
            }
            if (c == 'g') { t->prof_scroll = 0; t->dirty = 1; return; }
            if (c == 'G') { t->prof_scroll = 1 << 20; t->dirty = 1; return; }
            if (c == '?') {     /* the help card holds the key map */
                t->focus = DS4_TUI_HELP;
                t->dirty = 1;
                return;
            }
        }
        if (k == K_UP) {
            if (t->pv == PV_TOKENS) prof_select(t, 1);
            else cursor_move(t, -1);
            return;
        }
        if (k == K_DOWN) {
            if (t->pv == PV_TOKENS) prof_select(t, -1);
            else cursor_move(t, 1);
            return;
        }
        if (k == K_PGUP) {
            if (t->pv == PV_TOKENS) prof_page_move(t, 1);
            else scroll_focus(t, -10);
            return;
        }
        if (k == K_PGDN) {
            if (t->pv == PV_TOKENS) prof_page_move(t, -1);
            else scroll_focus(t, 10);
            return;
        }
        if (k == K_HOME) {
            if (t->pv == PV_TOKENS && t->list_n > 0) {
                const int ps = token_page_size(t);
                t->page = (t->list_n + ps - 1) / ps - 1;
                t->dirty = 1;
            } else cursor_edge(t, 0);
            return;
        }
        if (k == K_END) {
            if (t->pv == PV_TOKENS) { t->page = 0; t->dirty = 1; }
            else cursor_edge(t, 1);
            return;
        }
        return;
    }
    if (t->focus == DS4_TUI_MESSAGES) {
        if (k == K_UP) cursor_move(t, -1);
        else if (k == K_DOWN) cursor_move(t, 1);
        else if (k == K_PGUP) scroll_focus(t, -10);
        else if (k == K_PGDN) scroll_focus(t, 10);
        else if (k == K_HOME) cursor_edge(t, 0);
        else if (k == K_END) cursor_edge(t, 1);
        else if (k == K_LEFT || k == K_ESC) { t->focus = DS4_TUI_CHAT; t->dirty = 1; }
        else if (k == K_CHAR && ulen == 1 && utf8[0] == '?') {
            t->focus = DS4_TUI_HELP;
            t->dirty = 1;
        }
        return;
    }
    if (t->focus == DS4_TUI_HELP) {
        if (k == K_UP) cursor_move(t, -1);
        else if (k == K_DOWN) cursor_move(t, 1);
        else if (k == K_PGUP) scroll_focus(t, -10);
        else if (k == K_PGDN) scroll_focus(t, 10);
        else if (k == K_HOME) cursor_edge(t, 0);
        else if (k == K_END) cursor_edge(t, 1);
        else if (k == K_LEFT || k == K_ESC) { t->focus = DS4_TUI_CHAT; t->dirty = 1; }
        return;
    }

    /* Chat pane. */
    if (k == K_PGUP) { scroll_focus(t, -10); return; }
    if (k == K_PGDN) { scroll_focus(t, 10); return; }
    if (k == K_CTRL_P) { cursor_move(t, -1); return; }
    if (k == K_CTRL_N) { cursor_move(t, 1); return; }
    if (k == K_HOME) { cursor_edge(t, 0); return; }
    if (k == K_END) { cursor_edge(t, 1); return; }
    if (k == K_CTRL_U) { scroll_focus(t, -10); return; }
    if (k == K_CTRL_D) {
        if (!t->readline || t->input_len == 0) { t->eof = 1; return; }
        input_delete(t);
        return;
    }
    if (t->busy) return;

    if (k == K_CHAR) { input_insert(t, utf8, ulen); return; }
    if (k == K_BACKSPACE) { input_backspace(t); return; }
    if (k == K_DELETE) { input_delete(t); return; }
    if (k == K_LEFT) { if (t->input_pos) t->input_pos--; t->dirty = 1; return; }
    if (k == K_RIGHT) { if (t->input_pos < t->input_len) t->input_pos++; t->dirty = 1; return; }
    if (k == K_CTRL_A) { t->input_pos = 0; t->dirty = 1; return; }
    if (k == K_CTRL_E) { t->input_pos = t->input_len; t->dirty = 1; return; }
    if (k == K_CTRL_K) {
        t->input_len = t->input_pos;
        if (t->input) t->input[t->input_len] = '\0';
        t->dirty = 1;
        return;
    }
    if (k == K_CTRL_W) {
        while (t->input_pos > 0 && t->input[t->input_pos - 1] == ' ') input_backspace(t);
        while (t->input_pos > 0 && t->input[t->input_pos - 1] != ' ') input_backspace(t);
        return;
    }
    if (k == K_UP || k == K_DOWN) {
        if (t->hist_n == 0) return;
        if (k == K_UP) {
            if (t->hist_pos < t->hist_n) t->hist_pos++;
            input_set(t, t->history[t->hist_n - t->hist_pos]);
        } else {
            if (t->hist_pos > 0) t->hist_pos--;
            input_set(t, t->hist_pos ? t->history[t->hist_n - t->hist_pos] : "");
        }
        return;
    }
    if (k == K_ENTER) {
        if (t->readline) t->submitted = 1;
        return;
    }
    if (k == K_ESC) return;
}

/* ---------------------------------------------------------------- public -- */


/* Synthetic rows for smoke tests (DS4_TUI_FAKE=1). They exercise the same
 * record/stage/event APIs the engine uses, so view bugs show up without a
 * model load. */
static void tui_fake_feed(ds4_tui *t) {
    ds4_tui_chat_role(t, 2, "system");
    ds4_tui_chat_append(t, "fake feed: no model loaded; profiler data is synthetic", strlen("fake feed: no model loaded; profiler data is synthetic"), 0);
    ds4_tui_chat_newline(t);
    ds4_tui_chat_role(t, 0, "you");
    ds4_tui_chat_append(t, "show me where the token time goes", strlen("show me where the token time goes"), 0);
    ds4_tui_chat_newline(t);
    ds4_tui_chat_role(t, 1, "assistant");
    ds4_tui_chat_append(t, "Looking at the stage rows: ", strlen("Looking at the stage rows: "), 1);
    ds4_tui_chat_append(t, "attention and routed experts dominate.", strlen("attention and routed experts dominate."), 0);
    ds4_tui_chat_newline(t);
    for (int i = 0; i < 24; i++) {
        const int n = 60 + (i % 5) * 4;
        const int refill = (i % 3) == 0;
        ds4_profiler_token_begin_ex(DS4_PREC_DECODE_TOKEN, 100 + i, n, 1);
        ds4_profiler_note_stage("decode", "router", 1, n, 1,
                                (2 + i % 3) * 100000ull, 30000, 0, 0);
        ds4_profiler_note_stage("decode", "q_path", 1, n, 1,
                                (4 + i % 4) * 100000ull, 50000, 0, 0);
        ds4_profiler_note_stage("decode", "kv_path", 1, n, 1,
                                (3 + i % 3) * 100000ull, 40000, 0, 0);
        ds4_profiler_note_stage("decode", "attn_output", 1, n, 1,
                                (5 + i % 6) * 100000ull, 60000, 0, 0);
        ds4_profiler_note_stage("decode", "shared_gate_up", 1, n, 1,
                                (6 + i % 5) * 100000ull, 70000, 0, 0);
        ds4_profiler_note_stage("decode", "routed_moe", 1, n, 1,
                                (refill ? 45 : 18 + i % 9) * 100000ull,
                                (refill ? 8 : 3) * 100000ull,
                                (refill ? 30 : 10) * 100000ull,
                                refill ? 137000000ull : 0);
        ds4_profiler_note_stage("decode", "shared_down", 1, n, 1,
                                (4 + i % 4) * 100000ull, 50000, 0, 0);
        ds4_profiler_note_stage("decode", "attn_hc_post", 1, n, 1,
                                (2 + i % 2) * 100000ull, 20000, 0, 0);
        ds4_profiler_note_stage("decode", "ffn_hc_post", 1, n, 1,
                                (2 + i % 2) * 100000ull, 20000, 0, 0);
        for (int e = 0; e < 6; e++) {
            ds4_profiler_trace_event(DS4_PEV_ROUTE, 1, (uint16_t)((i * 6 + e) % 256), 0, 0);
            ds4_profiler_trace_expert(1, (uint16_t)((i * 6 + e) % 256),
                                      refill && e == 0 ? DS4_PTRACE_SRC_SSD : DS4_PTRACE_SRC_POOL,
                                      0xFF, 0, refill && e == 0 ? 6750000u : 0);
        }
        ds4_profiler_trace_event(DS4_PEV_XFER_EXPERT_START, 1, (uint16_t)(i % 256), 0, 6750000);
        ds4_profiler_trace_event(DS4_PEV_XFER_EXPERT_END, 1, (uint16_t)(i % 256), 0, 6750000);
        ds4_profiler_trace_event(DS4_PEV_SAMPLE, 0, (uint16_t)(100 + i), 0, 0);
        ds4_profiler_token_end(1);
        char line[64];
        snprintf(line, sizeof(line), "token %d", i);
        ds4_tui_chat_append(t, line, strlen(line), 0);
        ds4_tui_chat_newline(t);
    }
    /* Wide and combining glyphs must not shift the right border. */
    static const char wide[] = "wide glyphs: \xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c"
                               " \xe2\x86\x92 ok \xe2\x9c\x93 e\xcc\x81";
    ds4_tui_chat_append(t, wide, sizeof(wide) - 1, 0);
    ds4_tui_chat_newline(t);
    ds4_tui_log(t, 0, "fake feed: 24 decode records with 9 stages and 6 experts each");
}

static int env_int(const char *name, int def, int lo, int hi) {
    const char *v = getenv(name);
    if (!v || !v[0]) return def;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v) return def;
    if (n < lo) n = lo;
    if (n > hi) n = hi;
    return (int)n;
}

ds4_tui *ds4_tui_create(void) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return 0;
    const char *env = getenv("DS4_TUI");
    if (env && (env[0] == '0' || !strcmp(env, "off"))) return 0;
    setlocale(LC_CTYPE, "");
    ds4_tui *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->saved_stderr = dup(STDERR_FILENO);
    t->pipe_r = t->pipe_w = -1;
    int pfd[2];
    if (pipe(pfd) == 0) {
        t->pipe_r = pfd[0];
        t->pipe_w = pfd[1];
        fcntl(t->pipe_r, F_SETFL, fcntl(t->pipe_r, F_GETFL, 0) | O_NONBLOCK);
#if defined(F_SETPIPE_SZ)
        fcntl(t->pipe_w, F_SETPIPE_SZ, 1 << 20);
#endif
        dup2(t->pipe_w, STDERR_FILENO);
    }
    if (!term_init(t)) {
        if (t->pipe_w >= 0) close(t->pipe_w);
        if (t->pipe_r >= 0) close(t->pipe_r);
        if (t->saved_stderr >= 0) close(t->saved_stderr);
        free(t);
        return 0;
    }
    t->active = 1;
    t->focus = DS4_TUI_CHAT;
    t->chat_follow = 1;
    t->msg_follow = 1;
    t->granular = 1;
    t->pv = PV_OVERVIEW;
    /* A slow software-rendered terminal wants fewer full frames; the frame
     * diff makes each one cheap, but the emulator still pays per frame. */
    t->frame_iv = 1.0 / (double)env_int("DS4_TUI_FPS", 10, 1, 60);
    t->scr.sync = env_int("DS4_TUI_SYNC", 0, 0, 1);
    t->sel = 0;
    t->kind_filter = 0;      /* decode only until the reader asks for more */
    snprintf(t->prompt, sizeof(t->prompt), "ds4> ");
    tui_hist_load(t);
    ds4_profiler_set_record_enabled(1);
    ds4_profiler_set_stage_capture(1);
    t->records_enabled = 1;
    term_size(t);
    ds4_tui_log(t, 3, "ds4 terminal UI: chat, live profiler, console and help. Tab cycles panes, Alt-1..4 jumps, ? opens help.");
    if (getenv("DS4_TUI_FAKE")) tui_fake_feed(t);
    ds4_tui_redraw(t);
    return t;
}

void ds4_tui_destroy(ds4_tui *t) {
    if (!t) return;
    drain_stderr(t);
    term_done(t);
    t->active = 0;
    free(t->chat.s); free(t->chat.st); free(t->chat.wrap);
    free(t->msg.s); free(t->msg.st); free(t->msg.wrap);
    free(t->help.s); free(t->help.st); free(t->help.wrap);
    free(t->prof.s); free(t->prof.st); free(t->prof.wrap);
    free(t->detail_events);
    free(t->detail_experts);
    screen_drop_prev(&t->scr);
    free(t->scr.cells);
    free(t->input);
    free(t->paste);
    free(t->view);
    for (int i = 0; i < t->hist_n; i++) free(t->history[i]);
    if (t->pipe_r >= 0) close(t->pipe_r);
    free(t);
}

int ds4_tui_active(const ds4_tui *t) { return t && t->active; }

void ds4_tui_chat_role(ds4_tui *t, int role, const char *label) {
    if (!t) return;
    if (role == 0) t->chat_follow = 1;
    if (t->chat.len && t->chat.s[t->chat.len - 1] != '\n') doc_nl(&t->chat);
    unsigned char st = role == 0 ? ST_USER : role == 1 ? ST_ASSISTANT : ST_YELLOW;
    doc_puts(&t->chat, label ? label : (role == 0 ? "you" : role == 1 ? "assistant" : "system"), st);
    doc_puts(&t->chat, "\n", ST_NORMAL);
    t->dirty = 1;
}

void ds4_tui_chat_append(ds4_tui *t, const char *text, size_t len, int think) {
    if (!t || !text || len == 0) return;
    doc_putn(&t->chat, text, len, think ? ST_DIM : ST_NORMAL);
    t->dirty = 1;
}

void ds4_tui_chat_newline(ds4_tui *t) {
    if (!t) return;
    if (t->chat.len && t->chat.s[t->chat.len - 1] != '\n') doc_nl(&t->chat);
    t->dirty = 1;
}

static void msg_add(ds4_tui *t, int level, const char *text) {
    static const unsigned char st[] = {ST_GRAY, ST_YELLOW, ST_RED, ST_CYAN};
    char stamp[16];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(stamp, sizeof(stamp), "%H:%M:%S", &tmv);
    doc_printf(&t->msg, ST_GRAY, "%s ", stamp);
    doc_puts(&t->msg, text, st[level < 0 ? 0 : (level > 3 ? 3 : level)]);
    if (!text[0] || text[strlen(text) - 1] != '\n') doc_nl(&t->msg);
    t->dirty = 1;
}

void ds4_tui_log(ds4_tui *t, int level, const char *fmt, ...) {
    if (!t) return;
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    /* Captured stderr may contain embedded newlines: keep each line. */
    char *save = NULL;
    for (char *line = strtok_r(tmp, "\n", &save); line; line = strtok_r(NULL, "\n", &save))
        msg_add(t, level, line);
    if (!tmp[0]) msg_add(t, level, "");
}

void ds4_tui_status(ds4_tui *t, const char *fmt, ...) {
    if (!t) return;
    char next[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(next, sizeof(next), fmt, ap);
    va_end(ap);
    if (strcmp(next, t->status) != 0) {
        memcpy(t->status, next, sizeof(t->status));
        t->dirty = 1;
    }
}

void ds4_tui_set_busy(ds4_tui *t, int busy, const char *what) {
    if (!t) return;
    t->busy = busy ? 1 : 0;
    if (!busy) t->list_frozen = 0;
    if (busy) ds4_tui_status(t, "working: %s  (Ctrl-C stops)", what ? what : "generation");
    else t->status[0] = '\0';
    t->dirty = 1;
}

void ds4_tui_pump(ds4_tui *t) {
    if (!t || !t->active) return;
    drain_stderr(t);
    if (t->irq && t->irq(t->irq_ud)) {
        if (t->readline) {
            if (t->irq_clear) t->irq_clear(t->irq_ud);
            input_set(t, "");
            t->hist_pos = 0;
            ds4_tui_status(t, "interrupted");
        } else {
            /* Busy: leave the flag set so the generation loop sees it. */
            ds4_tui_status(t, "stopping generation (Ctrl-C)");
        }
    }
    int keys = 0;
    for (int guard = 0; guard < 64; guard++) {
        char utf8[8];
        size_t ul = 0;
        tui_key k = read_key(t, utf8, &ul);
        if (k == K_NONE) break;
        keys++;
        if (k == K_CTRL_C) {
            /* With ISIG the signal also arrives; clearing here makes the idle
             * prompt behave like the classic REPL. */
            if (t->readline) { input_set(t, ""); t->hist_pos = 0; }
            continue;
        }
        if (k == K_PASTE) {
            /* The bytes after the marker are text, not keys: stop decoding
             * and let the drain below take them. */
            t->pasting = 1;
            t->paste_len = 0;
            break;
        }
        handle_key(t, k, utf8, ul);
        if (t->submitted || t->eof) break;
    }
    /* A paste is drained outside the key loop: the bytes are text, and a big
     * one can arrive over several pumps. */
    while (t->pasting && paste_step(t)) { }
    const double now = tui_now();
    if (now - t->last_size_check >= 0.10) {
        t->last_size_check = now;
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) {
            if ((int)ws.ws_row != t->rows || (int)ws.ws_col != t->cols) {
                t->rows = ws.ws_row;
                t->cols = ws.ws_col;
                t->resize_at = now;
                t->dirty = 1;
            }
        }
    }
    if (now - t->last_refresh >= 0.50) t->dirty = 1;   /* profiler tick */
    if (t->slide) t->dirty = 1;                       /* cursor row slides */
    if (!t->dirty) return;
    /* The terminal reflows rows on resize; painting while it settles leaves a
     * half-cleared screen, so skip frames for a beat after the size changed. */
    if (now - t->resize_at < 0.06 && keys == 0) return;
    /* Keystrokes draw at once (that is what "reactive" means); streaming data
     * is capped so the terminal is not asked for more frames than it can show. */
    if (keys == 0 && now - t->last_draw < t->frame_iv) return;
    ds4_tui_redraw(t);
}

void ds4_tui_set_interrupt(ds4_tui *t, int (*requested)(void *),
                           void (*clear)(void *), void *ud) {
    if (!t) return;
    t->irq = requested;
    t->irq_clear = clear;
    t->irq_ud = ud;
}

void ds4_tui_set_granular(ds4_tui *t, int on) {
    if (!t) return;
    t->granular = on ? 1 : 0;
    ds4_profiler_set_stage_capture(on);
    ds4_profiler_set_record_enabled(on);
}

int ds4_tui_granular(const ds4_tui *t) { return t && t->granular; }

void ds4_tui_set_breakpoints(ds4_tui *t, int on) {
    if (!t) return;
    t->breakpoints = on ? 1 : 0;
    if (t->breakpoints)
        ds4_tui_log(t, 1, "breakpoints on: pauses after each token/chunk (Ctrl-B toggles)");
}

int ds4_tui_breakpoints(const ds4_tui *t) { return t && t->breakpoints; }

void ds4_tui_breakpoint(ds4_tui *t, const char *label) {
    if (!t || !t->active || !t->breakpoints || t->paused) return;
    t->paused = 1;
    ds4_tui_status(t, "BREAKPOINT %s - Space continue, b off, Ctrl-C stop",
                   label ? label : "");
    t->dirty = 1;
    while (t->active && t->paused) {
        ds4_tui_pump(t);
        if (t->irq && t->irq(t->irq_ud)) break;
        struct pollfd p = { STDIN_FILENO, POLLIN, 0 };
        (void)poll(&p, 1, 50);
    }
    t->paused = 0;
    t->status[0] = '\0';
    t->dirty = 1;
}

char *ds4_tui_readline(ds4_tui *t, const char *prompt) {
    if (!t) return NULL;
    t->readline = 1;
    t->submitted = 0;
    t->eof = 0;
    t->focus = DS4_TUI_CHAT;
    t->hist_pos = 0;
    input_set(t, "");
    if (prompt) snprintf(t->prompt, sizeof(t->prompt), "%s", prompt);
    t->status[0] = '\0';
    t->dirty = 1;
    for (;;) {
        ds4_tui_pump(t);
        if (t->submitted) {
            t->submitted = 0;
            t->readline = 0;
            char *out = strdup(t->input ? t->input : "");
            tui_hist_add(t, out);
            input_set(t, "");
            return out;
        }
        if (t->eof) {
            t->eof = 0;
            t->readline = 0;
            return NULL;
        }
        struct pollfd p = { STDIN_FILENO, POLLIN, 0 };
        (void)poll(&p, 1, 100);
    }
}
