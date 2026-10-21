/* tests/test_profiler.c — telemetry JSON contract.
 *
 * Runs with -DDS4_NO_GPU, so the backend reports zero devices; this covers the
 * engine-owned parts (host stats, timing accumulation, serialization shape). */
#include "ds4_profiler.h"

#include <stdio.h>
#include <string.h>

static int fails;

static void check(int cond, const char *what) {
    if (!cond) {
        printf("FAIL: %s\n", what);
        fails++;
    }
}

/* Quote-aware brace/bracket balance, so a brace inside a label cannot fool it. */
static int balanced(const char *s) {
    int depth = 0, in_str = 0, esc = 0;
    for (; *s; s++) {
        if (in_str) {
            if (esc) esc = 0;
            else if (*s == '\\') esc = 1;
            else if (*s == '"') in_str = 0;
            continue;
        }
        if (*s == '"') in_str = 1;
        else if (*s == '{' || *s == '[') depth++;
        else if (*s == '}' || *s == ']') depth--;
    }
    return depth == 0;
}

int main(void) {
    ds4_profiler_host_stats hs;
    memset(&hs, 0, sizeof(hs));
    hs.has_ssd = 1;
    hs.has_ram = 1;
    hs.total_experts = 11008;
    hs.expert_bytes = 6750000;
    hs.ram_experts = 1203;
    hs.ram_bytes = 1203ull * 6750000ull;
    hs.model_bytes = 80ull << 30;
    hs.kv_bytes = 600000000ull;
    hs.dense_n = 1;
    memcpy(hs.dense[0].name, "embedding", 10);
    hs.dense[0].bytes = 1ull << 30;
    ds4_profiler_set_host_stats(&hs);

    ds4_profiler_reset_timing();
    ds4_profiler_note_prefill(100.0, 41, 1);

    static char buf[64 * 1024];
    const size_t n = ds4_profiler_json(buf, sizeof(buf));
    check(n > 0 && n < sizeof(buf), "json size");
    check(strstr(buf, "\"expert_distribution\"") != NULL, "distribution key");
    check(strstr(buf, "\"on_gpu\"") != NULL, "on_gpu key");
    check(strstr(buf, "\"in_host_ram\"") != NULL, "in_host_ram key");
    check(strstr(buf, "\"bytes_per_expert\":6750000") != NULL, "per-expert bytes");
    check(strstr(buf, "\"timing\"") != NULL, "timing key");
    check(strstr(buf, "\"per_token_ms\"") != NULL, "per_token_ms key");
    check(strstr(buf, "\"weight_transfer_ms\"") != NULL, "weight key");
    check(strstr(buf, "\"activation_transfer_ms\"") != NULL, "activation key");
    check(strstr(buf, "\"host_ms\"") != NULL, "host key");
    check(strstr(buf, "\"device_wait_ms\"") != NULL, "device wait key");
    check(strstr(buf, "\"prefill_ms\":{\"last\":100.000") != NULL, "prefill last");
    check(strstr(buf, "\"model_bytes\":85899345920") != NULL, "model bytes");
    check(balanced(buf), "balanced braces");

    /* Detailed records: one token, two measured stages, one expert use. */
    ds4_profiler_reset_timing();
    ds4_profiler_set_record_enabled(1);
    ds4_profiler_set_stage_capture(1);
    check(ds4_profiler_token_begin_ex(DS4_PREC_DECODE_TOKEN, 42, 7, 1) == 1,
          "record opens");
    check(ds4_profiler_token_begin_ex(DS4_PREC_DECODE_TOKEN, 43, 8, 1) == 0,
          "nested record refused");
    ds4_profiler_record_text("hello", 5);
    ds4_profiler_note_stage("decode", "routed_moe", 3, 7, 1,
                            1200000, 200000, 900000, 6750000);
    ds4_profiler_note_stage("decode", "q_path", 3, 7, 1,
                            300000, 50000, 200000, 0);
    ds4_profiler_trace_expert(3, 77, DS4_PTRACE_SRC_SSD, 0xFF, 0, 6750000);
    ds4_profiler_trace_event(DS4_PEV_ROUTE, 3, 77, 0, 0);
    ds4_profiler_token_end(1);

    ds4_prof_record_meta rm[4];
    const int rn = ds4_profiler_record_list(rm, 4);
    check(rn == 1, "one record listed");
    check(rn == 1 && rm[0].token == 42 && rm[0].pos == 7, "record identity");
    check(rn == 1 && rm[0].n_tokens == 1, "record token count");
    check(rn == 1 && strstr(rm[0].text, "hello") != NULL, "record text");
    ds4_prof_record rec;
    check(ds4_profiler_record_get(rm[0].seq, &rec) == 1, "record get");
    check(rec.stage_n == 2, "two stage rows in record");
    check(rec.stage_n == 2 && rec.stages[0].total_ns == 1200000, "stage total");
    check(rec.stage_n == 2 && rec.stages[0].encode_ns == 200000, "stage launch");
    check(rec.stage_n == 2 && rec.stages[0].wait_ns == 900000, "stage wait");
    check(rec.expert_n == 1, "expert use recorded");
    check(rec.event_n >= 3, "events recorded");
    ds4_prof_expert_trace ex;
    check(ds4_profiler_record_experts(rec.seq, &ex, 1) == 1, "expert get");
    check(ex.layer == 3 && ex.expert == 77 && ex.source == DS4_PTRACE_SRC_SSD,
          "expert fields");
    ds4_prof_event ev[8];
    const int evn = ds4_profiler_record_events(rec.seq, ev, 8);
    check(evn >= 3, "event get");
    check(evn >= 1 && ev[0].rec_seq == rec.seq, "event record link");
    ds4_prof_view view;
    const int sn = ds4_profiler_view(&view);
    check(sn == 2, "two stage views");
    check(view.n_records == 1, "view records");
    check(view.stages[0].total_ms.n == 1, "stage stat sampled");
    check(view.stages[0].total_ms.last == 1.2, "stage ms");

    const size_t n2 = ds4_profiler_json(buf, sizeof(buf));
    check(n2 > 0 && n2 < sizeof(buf), "json size with records");
    check(strstr(buf, "\"stage_rows\"") != NULL, "stage rows key");
    check(strstr(buf, "\"records\"") != NULL, "records key");
    check(strstr(buf, "\"seq\":") != NULL, "record sequence key");
    check(balanced(buf), "balanced braces with records");

    ds4_profiler_set_record_enabled(0);
    ds4_profiler_set_stage_capture(0);

    ds4_profiler_note_prefill(100.0, 41, 1);
    ds4_profiler_timing t;
    ds4_profiler_timing_snapshot(&t);
    check(t.prefill_ms.n == 1 && t.prefill_ms.last == 100.0, "prefill accum");
    check(t.prefill_tokens_per_s.last == 410.0, "prefill tokens/s");

    /* A record keeps its ordered events for as long as it stays in the record
     * ring. The event ring used to be far smaller than a token's event count
     * times the record ring, so every record older than ~40 tokens read as an
     * empty list while the record itself was still there. */
    {
        enum { RECS = 200, EV_PER_REC = 3000 };
        ds4_profiler_reset_timing();
        ds4_profiler_set_record_enabled(1);
        ds4_profiler_set_stage_capture(1);
        uint64_t first = 0;
        for (int i = 0; i < RECS; i++) {
            if (!ds4_profiler_token_begin_ex(DS4_PREC_DECODE_TOKEN, i, i, 1))
                break;
            for (int e = 0; e < EV_PER_REC; e++)
                ds4_profiler_note_stage("decode", "q_path", (uint32_t)(e % 43),
                                        i, 1, 100000, 50000, 0, 0);
            ds4_profiler_token_end(1);
            if (i == 0) {
                /* The list is oldest first, so the only record here is the one
                 * this loop has to still find its events on at the end. */
                ds4_prof_record_meta m;
                if (ds4_profiler_record_list(&m, 1) == 1) first = m.seq;
            }
        }
        ds4_prof_record r;
        check(ds4_profiler_record_get(first, &r) == 1, "oldest record kept");
        check(r.event_n >= EV_PER_REC, "oldest record counted its events");
        ds4_prof_event ev[8];
        check(ds4_profiler_record_events(first, ev, 8) == 8,
              "oldest record still has its ordered events");
        ds4_profiler_set_stage_capture(0);
        ds4_profiler_set_record_enabled(0);
        ds4_profiler_reset_timing();
    }

    if (fails) {
        printf("test_profiler: %d failure(s)\n", fails);
        return 1;
    }
    printf("test_profiler: ok (%zu bytes json)\n", n);
    return 0;
}
