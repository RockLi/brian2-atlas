"""Native BTRS for binomial inversion's unrepresentable initial mass.

Hörmann's transformed-rejection envelope is evaluated against stable binomial
log probabilities. Candidate and complement counts stay integer until output.
"""
import math


def _source():
    polynomial='1.0f/132.0f'
    for n in range(11,1,-1):
        polynomial=f'({(-1)**n}.0f/{n*(n-1)}.0f + u*{polynomial})'
    tails='\n'.join(f'case {k}: return {math.lgamma(k+1)-(k+.5)*math.log(k)+k-.5*math.log(2*math.pi):.17g}f;'
                    for k in range(1,16))
    return r'''
inline float b2_binomial_stirling(long count) {
    switch (count) {
TAILS
    }
    float inverse=1.0f/float(count), square=inverse*inverse;
    return inverse*(1.0f/12.0f+square*(-1.0f/360.0f+
                    square*(1.0f/1260.0f-square/1680.0f)));
}
inline float b2_binomial_deviance(long count, float mean, float delta) {
    float u=delta/mean;
    if (abs(u)<0.125f) return (delta*u)*POLYNOMIAL;
    return float(count)*log(float(count)/mean)-delta;
}
inline float b2_binomial_mean_tail(ulong n, float p, float mean) {
    // Explicit fma recovers the product residual even with contraction disabled.
    // The integer residual also preserves n when it exceeds f32's exact range.
    float nf=float(n);
    return fma(nf,p,-mean)+float(long(n)-long(nf))*p;
}
inline float b2_binomial_log_mass(ulong n, long k, float p,
                                   long center, float fraction, float mean) {
    if (k==0) return float(n)*b2_log1p(-p);
    if (k==long(n)) return float(n)*log(p);
    long other=long(n)-k;
    float delta=float(k-center)-fraction;
    float complement=float(long(n)-center)-fraction;
    float deviance=b2_binomial_deviance(k,mean,delta)+
                   b2_binomial_deviance(other,complement,-delta);
    float normalizer=1.8378770664093455f+log(float(k))+log(float(other))-log(float(n));
    return b2_binomial_stirling(long(n))-b2_binomial_stirling(k)-b2_binomial_stirling(other)
           -deviance-0.5f*normalizer;
}
inline ulong b2_binomial_btrs(ulong seed, ulong stream, ulong tick, ulong index,
                             ulong n, float p, thread bool *fault) {
    float mean=float(n)*p;
    long center=long(mean);
    float fraction=(mean-float(center))+b2_binomial_mean_tail(n,p,mean);
    float sigma=sqrt(mean*(1.0f-p));
    float b=1.15f+2.53f*sigma, a=-0.0873f+0.0248f*b+0.01f*p;
    float squeeze=0.92f-4.2f/b, alpha=(2.83f+5.1f/b)*sigma;
    long mode=center+long(floor(fraction+p));
    float peak=b2_binomial_log_mass(n,mode,p,center,fraction,mean);
    for (ulong draw=0;draw<8192;draw+=2) {
        float horizontal=b2_uniform(seed,stream,tick,index,draw)-0.5f;
        float vertical=b2_uniform(seed,stream,tick,index,draw+1);
        float gap=0.5f-abs(horizontal);
        if (gap==0.0f) continue;
        long offset=long(floor((2.0f*a/gap+b)*horizontal+fraction+0.5f));
        long candidate=center+offset;
        if (candidate<0 || candidate>long(n)) continue;
        if (gap>=0.07f && vertical<=squeeze) return ulong(candidate);
        float envelope=log(vertical*alpha/(a/(gap*gap)+b));
        if (envelope<=b2_binomial_log_mass(n,candidate,p,center,fraction,mean)-peak)
            return ulong(candidate);
    }
    *fault=true; return 0ul;
}
'''.replace('TAILS',tails).replace('POLYNOMIAL',polynomial)


SOURCE=_source()
