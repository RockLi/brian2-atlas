"""GPU counter RNG: AtlasIR counters with an explicit 24-bit uniform projection.

The helpers run on the GPU; no host-side sample table or mutable RNG state is
used. Continuous distributions follow the GPU f32 arithmetic contract. Binomial
inversion switches to bounded BTRS when its initial mass is unrepresentable.
"""
from .metal_binomial import SOURCE as _BINOMIAL_SOURCE

RNG_PROFILE = "b2-counter-f32-u24-v0"
RANDOM_OPS = frozenset({"rand", "randn", "binomial", "poisson"})


def has_random(value):
    if isinstance(value, dict):
        return value.get("op") in RANDOM_OPS or any(has_random(v) for v in value.values())
    return isinstance(value, (list, tuple)) and any(has_random(v) for v in value)


SOURCE = r'''
inline ulong b2_mix64(ulong value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}
inline float b2_uniform(ulong seed, ulong stream, ulong tick, ulong index, ulong draw) {
    ulong value = seed ^ stream*0x9e3779b97f4a7c15ul ^ tick*0xd1b54a32d192ed03ul
                 ^ index*0x94d049bb133111ebul ^ draw*0x369dea0f31a53f85ul;
    // Exactly representable samples in [0,1), never rounded up to one.
    return float(b2_mix64(value) >> 40) * 0x1p-24f;
}
inline float b2_normal(ulong seed, ulong stream, ulong tick, ulong index, thread bool *fault) {
    for (ulong draw=0; draw<8192; draw+=2) {
        float a=2.0f*b2_uniform(seed,stream,tick,index/2,draw)-1.0f;
        float b=2.0f*b2_uniform(seed,stream,tick,index/2,draw+1)-1.0f;
        float radius=a*a+b*b;
        if (radius<1.0f && radius!=0.0f)
            return sqrt(-2.0f*log(radius)/radius) * ((index & 1ul) ? b : a);
    }
    *fault=true; return 0.0f;
}
''' + _BINOMIAL_SOURCE + r'''
inline float b2_binomial(ulong seed, ulong stream, ulong tick, ulong index,
                         ulong n, float p, bool approximate, thread bool *fault) {
    if (!(isfinite(p) && p>=0.0f && p<=1.0f)) { *fault=true; return 0.0f; }
    if (p==0.0f) return 0.0f;
    if (p==1.0f) return float(n);
    float mean=float(n)*p, complement=float(n)*(1.0f-p);
    if (approximate && mean>5.0f && complement>5.0f)
        return b2_normal(seed,stream,tick,index,fault)*sqrt(mean*(1.0f-p))+mean;
    bool reverse=p>0.5f;
    float probability=reverse ? 1.0f-p : p;
    float q=1.0f-probability;
    // log1p avoids losing low probabilities when q rounds to one.
    float initial=b2_exp(float(n)*b2_log1p(-probability));
    if (!(initial>0.0f && isfinite(initial))) {
        ulong sample=b2_binomial_btrs(seed,stream,tick,index,n,probability,fault);
        return float(reverse ? n-sample : sample);
    }
    float bound=float(n)*probability+10.0f*sqrt(float(n)*probability*q+1.0f);
    bound=bound<float(n) ? bound : float(n);
    for (ulong draw=0; draw<4096; ++draw) {
        ulong x=0;
        float mass=initial, u=b2_uniform(seed,stream,tick,index,draw);
        while (true) {
            if (u<=mass) return float(reverse ? n-x : x);
            ++x;
            if (float(x)>bound) break;
            u-=mass;
            mass=(float(n-x+1)*probability*mass)/(float(x)*q);
            if (!(mass>=0.0f && isfinite(mass))) { *fault=true; return 0.0f; }
        }
    }
    *fault=true; return 0.0f;
}
'''
