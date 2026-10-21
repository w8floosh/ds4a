/* tests/test_dashboard_bind.c — verify the telemetry dashboard's network bind.
 *
 * Starts the dashboard on 0.0.0.0 and fetches /stats.json through loopback and
 * through this host's own LAN address, so the two things that can break a
 * remote view (the bind address and serving off-loopback) are both exercised.
 * No model, no engine, no instance lock: the profiler simply reports zeroes.
 */
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "ds4_dashboard.h"

static int http_get_ok(const char *host, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    struct timeval tv = { 5, 0 };
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) { close(fd); return 0; }
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) { close(fd); return 0; }
    const char *req = "GET /stats.json HTTP/1.0\r\n\r\n";
    if (write(fd, req, strlen(req)) < 0) { close(fd); return 0; }
    char buf[2048];
    const ssize_t n = read(fd, buf, sizeof(buf) - 1u);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    /* The snapshot opens with expert_distribution; "devices" is far past any
     * sane read, so assert on the head of the body instead. */
    return strstr(buf, "200 OK") != NULL &&
           strstr(buf, "expert_distribution") != NULL;
}

static int fetch_with_retry(const char *host, int port) {
    for (int i = 0; i < 20; i++) {
        if (http_get_ok(host, port)) return 1;
        usleep(100000);
    }
    return 0;
}

int main(void) {
    const int port = 8799;
    if (!ds4_dashboard_start("0.0.0.0", port, 0)) {
        fprintf(stderr, "test_dashboard_bind: dashboard did not start\n");
        return 1;
    }
    int failures = 0;
    if (!fetch_with_retry("127.0.0.1", port)) {
        fprintf(stderr, "FAIL: loopback fetch\n");
        failures++;
    } else {
        printf("ok: loopback fetch\n");
    }
    struct ifaddrs *ifa = NULL;
    if (getifaddrs(&ifa) != 0) {
        fprintf(stderr, "FAIL: getifaddrs\n");
        failures++;
    } else {
        int checked = 0;
        for (struct ifaddrs *p = ifa; p; p = p->ifa_next) {
            if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
            const struct sockaddr_in *s = (const struct sockaddr_in *)p->ifa_addr;
            if ((ntohl(s->sin_addr.s_addr) >> 24) == 127u) continue;
            char host[INET_ADDRSTRLEN] = {0};
            if (!inet_ntop(AF_INET, &s->sin_addr, host, sizeof(host))) continue;
            checked++;
            if (fetch_with_retry(host, port)) {
                printf("ok: LAN fetch on %s (%s)\n", host, p->ifa_name);
            } else {
                fprintf(stderr, "FAIL: LAN fetch on %s (%s)\n", host, p->ifa_name);
                failures++;
            }
        }
        freeifaddrs(ifa);
        if (checked == 0) printf("note: no non-loopback IPv4 interface to test\n");
    }
    ds4_dashboard_stop();
    printf(failures ? "test_dashboard_bind: FAILED\n" : "test_dashboard_bind: PASS\n");
    return failures ? 1 : 0;
}
