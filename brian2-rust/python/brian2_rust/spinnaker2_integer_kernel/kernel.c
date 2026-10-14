#include "kernel.h"
#include <stddef.h>

static int32_t sat16(int32_t x) {
    return x < -32768 ? -32768 : (x > 32767 ? 32767 : x);
}
/* C division truncates toward zero; signed right shift is implementation-defined.
 * Spell out floor(x/2) so negative odd states behave identically on every host.
 */
static int32_t decay(int32_t x) {
    return (x < 0 ? -((-x + 1) / 2) : x / 2) + (x == 1);
}
static int weights_valid(const int32_t *w, int n) {
    for (int k = 0; k < n; ++k) if (w[k] < -128 || w[k] > 127) return 0;
    return 1;
}
int atlas_s2_integer_run(int ni, int nh, int no, int batch, int steps,
    const int32_t *wi, const int32_t *wo, const int32_t *wr,
    const int32_t *threshold, const uint8_t *input, int32_t *state, int32_t *trace) {
    if (ni < 1 || ni > 16 || nh < 1 || nh > 128 || no < 1 || no > 8 ||
        batch < 1 || batch > 128 || steps < 1 || steps > 128) return 1;
    if (!wi || !wo || !wr || !threshold || !input || !state || !trace) return 2;
    const int cells = nh + no;
    if (!weights_valid(wi, ni * nh) || !weights_valid(wo, nh * no) ||
        !weights_valid(wr, nh * nh)) return 3;
    for (int j = 0; j < cells; ++j)
        if (threshold[j] < 1 || threshold[j] > 32767) return 4;
    for (int k = 0; k < batch * steps * ni; ++k) if (input[k] > 15) return 5;
    for (int r = 0; r < batch; ++r) {
        const int32_t *s = state + r * 3 * cells;
        for (int j = 0; j < cells; ++j)
            if (s[j] < -32768 || s[j] > 32767 || s[cells+j] < -32768 ||
                s[cells+j] > 32767 || s[2*cells+j] < 0 ||
                s[2*cells+j] > (j < nh ? 31 : 1)) return 6;
    }
    for (int r = 0; r < batch; ++r) {
        int32_t *s = state + r * 3 * cells;
        int32_t *v = s + cells, *q = v + cells;
        for (int t = 0; t < steps; ++t) {
            const uint8_t *x = input + (r * steps + t) * ni;
            int32_t previous_q[128];
            for (int i = 0; i < nh; ++i) previous_q[i] = q[i];
            /* Source order and a clamp after every edge are part of the ABI.
             * Both recurrence and readout consume the previous tick's counts.
             */
            for (int j = 0; j < nh; ++j) {
                int32_t current = s[j];
                for (int i = 0; i < ni; ++i) current = sat16(current + wi[i*nh+j] * x[i]);
                for (int i = 0; i < nh; ++i) current = sat16(current + wr[i*nh+j] * previous_q[i]);
                s[j] = current;
            }
            for (int j = 0; j < no; ++j) {
                int32_t current = s[nh+j];
                for (int i = 0; i < nh; ++i) current = sat16(current + wo[i*no+j] * previous_q[i]);
                s[nh+j] = current;
            }
            for (int j = 0; j < cells; ++j) {
                s[j] -= decay(s[j]);
                int32_t u = sat16(v[j] + sat16(-decay(v[j]) + s[j]));
                int32_t count = u > 0 ? u / threshold[j] : 0;
                const int cap = j < nh ? 31 : 1;
                q[j] = count > cap ? cap : count;
                v[j] = u - q[j] * threshold[j];
            }
            int32_t *out = trace + ((r * steps + t) * 3 * cells);
            for (int j = 0; j < 3 * cells; ++j) out[j] = s[j];
        }
    }
    return 0;
}
