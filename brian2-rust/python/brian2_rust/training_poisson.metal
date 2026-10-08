uint atlas_dynamic_poisson_bits(float x){return as_type<uint>(x);}
float atlas_dynamic_poisson_from_bits(uint x){return as_type<float>(x);}
// Version-one native Poisson numerical core. Integer candidates remain int64
// until admission to int32; adding an O(sqrt(rate)) offset in float32 would
// otherwise quantize large Poisson counts to multiples of 64 or 128.
ulong atlas_dynamic_poisson_mix(ulong x){
 x=(x^(x>>30))*0xbf58476d1ce4e5b9ul;x=(x^(x>>27))*0x94d049bb133111ebul;return x^(x>>31);
}
float atlas_dynamic_poisson_uniform(ulong key,uint draw){
 ulong bits=atlas_dynamic_poisson_mix(key^atlas_dynamic_poisson_mix(ulong(draw)+0x9e3779b97f4a7c15ul));
 ulong odd=((bits>>12)<<1)|1ul;
 return min(float(odd)*1.1102230246251565e-16f,0.999999940395355224609375f);
}
float atlas_dynamic_poisson_log1p(float x){
 float u=1+x;if(u==1)return x;return log(u)*(x/(u-1));
}
float atlas_dynamic_poisson_log_probability(long count,float rate){
 if(count==0)return atlas_dynamic_poisson_from_bits(atlas_dynamic_poisson_bits(rate)^0x80000000u);
 if(count<16){float factorial=0;for(int k=2;k<=count;k++)factorial+=log(float(k));return float(count)*log(rate)-rate-factorial;}
 float k=float(count),inv=1/k,inv2=inv*inv;
 float stirling=inv*(1.0f/12-inv2*(1.0f/360-inv2*(1.0f/1260-inv2*(1.0f/1680-inv2/1188))));
 long base=long(floor(rate));float delta=float(count-base)-(rate-float(base));float deviance;
 if(abs(delta)<.1f*(k+rate)){
  float v=delta/(k+rate),v2=v*v,sum=delta*v,term=2*k*v;
  for(int j=1;j<100;j++){term*=v2;float next=sum+term/float(2*j+1);if(next==sum)break;sum=next;}
  deviance=sum;
 }else deviance=k*log(k/rate)+rate-k;
 return -stirling-deviance-.5f*log(6.2831853071795864769f*k);
}
// error: 1 invalid rate, 3 exhausted draw budget, 4 accepted int32 overflow.
int atlas_dynamic_poisson(float rate,ulong key,thread uint& draws,thread uint& error){
 draws=0;error=0;
 uint bits=atlas_dynamic_poisson_bits(rate),magnitude=bits&0x7fffffffu;
 if(magnitude>=0x4f000000u||((bits&0x80000000u)&&magnitude)){error=1;return 0;}
 if(magnitude==0)return 0;
 // Metal arithmetic flushes subnormals. Every positive subnormal is below
 // the smallest possible open uniform (2^-53), so its first waiting time
 // always exceeds rate; still count that draw and keep its score distinct
 // from the zero-rate boundary. Negative subnormals are rejected above.
 if(magnitude<0x00800000u){draws=1;return 0;}
 if(rate<10){
  float arrival=0;int count=0;
  while(draws<100000){float u=atlas_dynamic_poisson_uniform(key,draws++);arrival-=atlas_dynamic_poisson_log1p(-u);if(arrival>=rate)return count;count++;}
  error=3;return 0;
 }
 float b=.931f+2.53f*sqrt(rate),a=-.059f+.02483f*b;
 float alpha=1.1239f+1.1328f/(b-3.4f),vr=.9277f-3.6224f/(b-2);
 long base=long(floor(rate));float fraction=rate-float(base);
 while(draws<100000){
  float u=atlas_dynamic_poisson_uniform(key,draws++)-.5f,v=atlas_dynamic_poisson_uniform(key,draws++),us=.5f-abs(u);
  if(us==0)continue;
  long count=base+long(floor((2*a/us+b)*u+fraction+.43f));
  bool accept=us>=.07f&&v<=vr;
  if(!accept){if(count<0||(us<.013f&&v>us))continue;accept=log(v)+log(alpha)-log(a/(us*us)+b)<=atlas_dynamic_poisson_log_probability(count,rate);}
  if(accept){if(count<0||count>2147483647l){error=4;return 0;}return int(count);}
 }
 error=3;return 0;
}
float atlas_dynamic_poisson_score(float rate,int count){
 uint bits=atlas_dynamic_poisson_bits(rate),magnitude=bits&0x7fffffffu;
 if(magnitude==0||magnitude>=0x4f000000u||(bits&0x80000000u)||count<0)return NAN;
 if(count==0)return -1;
 if(magnitude<0x00800000u)return (float(count)/float(magnitude))*8.507059173023462e37f*8388608.0f;
 long base=long(floor(rate));return (float(long(count)-base)-(rate-float(base)))/rate;
}

// A restore dispatch contains at most 100 records. The normal sampler's
// 100000-uniform bound thus also bounds work before the host checks its sum.
kernel void atlas_poisson_checkpoint_validate(
 device const ulong* sizes [[buffer(0)]],device const ulong* keys [[buffer(1)]],
 device const float* rates [[buffer(2)]],device const int* expected [[buffer(3)]],
 device uint* work [[buffer(4)]],device uint* errors [[buffer(5)]], uint b [[thread_position_in_grid]]) {
 if(ulong(b)>=sizes[0])return;
 uint draws=0,error=0;int count=atlas_dynamic_poisson(rates[b],keys[b],draws,error);
 work[b]=draws;errors[b]=error?error:count==expected[b]?0:5;
}
