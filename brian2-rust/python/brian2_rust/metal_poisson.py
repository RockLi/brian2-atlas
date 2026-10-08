"""Poisson counter sampling with stable f32 central log probabilities.

PTRS follows the algorithm already used by the Rust reference. The acceptance
probability is evaluated with a deviance/Stirling decomposition to avoid
subtracting O(lambda log lambda) quantities in the GPU's f32 profile.
"""
import math


def _source():
    # Near the mean, (1+u)*log(1+u)-u = sum_{n>=2} (-u)^n/(n*(n-1)).
    # Horner evaluation of terms 2..12 avoids cancellation. Elsewhere use the
    # direct expression; the candidate is already a nonnegative integer.
    polynomial='1.0f/132.0f'
    for n in range(11,1,-1):
        polynomial=f'({(-1)**n}.0f/{n*(n-1)}.0f + u*{polynomial})'
    small='\n'.join(f'case {k}: return -rate+float(k)*log(rate)-{math.lgamma(k+1):.17g}f;'
                    if k>1 else f'case {k}: return -rate+float(k)*log(rate);'
                    for k in range(1,16))
    return r'''
inline float b2_poisson_log_mass(long k, float rate, long center, float fraction) {
    if (k==0) return -rate;
    switch (k) {
SMALL_CASES
    }
    float x=float(k), delta=float(k-center)-fraction, u=delta/rate;
    float deviance;
    if (abs(u)<0.125f) deviance=(delta*u)*POLYNOMIAL;
    else deviance=x*log(x/rate)-delta;
    float inverse=1.0f/x, square=inverse*inverse;
    float correction=inverse*(1.0f/12.0f+square*(-1.0f/360.0f+
                         square*(1.0f/1260.0f-square/1680.0f)));
    return -deviance-0.9189385332046727f-0.5f*log(x)-correction;
}

inline float b2_poisson(ulong seed, ulong stream, ulong tick, ulong index,
                        float rate, thread bool *fault) {
    if (!(isfinite(rate) && rate>=0.0f && rate<=1.0e12f)) {
        *fault=true; return 0.0f;
    }
    if (rate==0.0f) return 0.0f;
    if (rate<10.0f) {
        float limit=b2_exp(-rate), product=1.0f;
        for (ulong draw=0;draw<8192; ++draw) {
            product*=b2_uniform(seed,stream,tick,index,draw);
            if (product<=limit) return float(draw);
        }
        *fault=true; return 0.0f;
    }
    float root=sqrt(rate), b=0.931f+2.53f*root;
    float a=-0.059f+0.02483f*b;
    float inverse_alpha=1.1239f+1.1328f/(b-3.4f);
    float squeeze=0.9277f-3.6224f/(b-2.0f);
    long center=long(rate);
    float fraction=rate-float(center);
    for (ulong draw=0;draw<8192;draw+=2) {
        float u=b2_uniform(seed,stream,tick,index,draw)-0.5f;
        float v=b2_uniform(seed,stream,tick,index,draw+1);
        float us=0.5f-abs(u);
        if (us==0.0f) continue;
        float offset=floor((2.0f*a/us+b)*u+fraction+0.43f);
        // The largest offset under the U24 grid and rate bound is < 2^41.
        // Keep the candidate's low integer bits until acceptance; adding it
        // to a large f32 mean first would quantize the rejection decision.
        long k=center+long(offset);
        if (k<0) continue;
        if (us>=0.07f && v<=squeeze) return float(k);
        if (us<0.013f && v>us) continue;
        float left=log(v*inverse_alpha/(a/(us*us)+b));
        if (left<=b2_poisson_log_mass(k,rate,center,fraction)) return float(k);
    }
    *fault=true; return 0.0f;
}
'''.replace('SMALL_CASES',small).replace('POLYNOMIAL',polynomial)


SOURCE=_source()
