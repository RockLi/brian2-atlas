// Shared by the Metal host bridge and generated scalar CPU mirror.
// Match executor.rs: minimum live tick*dt, with min(dt)*1e-12 coalescence.
#include <stdint.h>
#include <math.h>
static int b2_active_clocks(const int64_t *ticks, const int64_t *ends,
                            const double *dt, uint32_t clocks, uint8_t *active) {
    double next=INFINITY, epsilon=INFINITY;
    for (uint32_t c=0; c<clocks; ++c) {
        if (dt[c]<epsilon) epsilon=dt[c];
        double time=(double)ticks[c]*dt[c];
        if (ticks[c]<ends[c] && time<next) next=time;
    }
    if (next==INFINITY) return 0;
    epsilon*=1e-12;
    for (uint32_t c=0; c<clocks; ++c)
        active[c]=ticks[c]<ends[c] && fabs((double)ticks[c]*dt[c]-next)<=epsilon;
    return 1;
}
