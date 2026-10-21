/* ds4_dashboard.h — local web dashboard over the engine telemetry.
 *
 * A background thread serves two routes:
 *   GET /            the single-page dashboard (HTML/JS, no external assets)
 *   GET /stats.json  the ds4_profiler_json() snapshot
 * The page polls the JSON and renders it. The server is dependency-free; it
 * replaces the old animated terminal panel.
 *
 * It binds 127.0.0.1 by default. Pass an explicit `bind_addr` (or set
 * DS4_DASHBOARD_BIND) to serve the local network instead — 0.0.0.0 for every
 * interface, or one address. There is no authentication, so treat a
 * non-loopback bind as exposing engine telemetry to whoever can reach the
 * port, and prefer a trusted LAN.
 */
#ifndef DS4_DASHBOARD_H
#define DS4_DASHBOARD_H

#ifdef __cplusplus
extern "C" {
#endif

/* Start the dashboard. `bind_addr` NULL/"" means 127.0.0.1, else an IPv4
 * address literal or "0.0.0.0" for all interfaces. `port` 0 picks the default;
 * if the port is taken the next few are tried. `open_browser` launches the
 * default URL handler (loopback binds only — a headless server has no browser).
 * Returns 1 if a server is listening (idempotent), 0 if disabled or failed. */
int  ds4_dashboard_start(const char *bind_addr, int port, int open_browser);

/* Stop the server and join the thread. Safe to call when not started. */
void ds4_dashboard_stop(void);

int  ds4_dashboard_active(void);

/* The URL actually served (valid while active), e.g. "http://127.0.0.1:8765/". */
const char *ds4_dashboard_url(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_DASHBOARD_H */
