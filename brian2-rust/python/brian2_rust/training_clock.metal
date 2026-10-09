// Exact binary64 clock subtraction and comparison using integer arithmetic.
// Clock values are detached; model state and gradients retain their GPU dtype.
ulong atlas_clock_shr_jam(ulong x,uint shift){
 if(shift==0)return x;
 if(shift>=64)return ulong(x!=0);
 return (x>>shift)|ulong((x<<(64-shift))!=0);
}
ulong atlas_clock_sub(ulong a,ulong b){
 const ulong fraction=0x000fffffffffffffUL,hidden=0x0010000000000000UL;
 ulong am=a&fraction,bm=b&fraction;int ae=int((a>>52)&2047),be=int((b>>52)&2047);
 bool as=(a>>63)!=0,bs=(b>>63)==0;
 if(ae)am|=hidden;else ae=1;
 if(be)bm|=hidden;else be=1;
 if(ae<be||(ae==be&&am<bm)){
  ulong m=am;am=bm;bm=m;int e=ae;ae=be;be=e;bool s=as;as=bs;bs=s;
 }
 am<<=3;bm=atlas_clock_shr_jam(bm<<3,uint(ae-be));
 ulong m=as==bs?am+bm:am-bm;
 if(m==0)return as==bs&&as?0x8000000000000000UL:0;
 if(m>=0x0100000000000000UL){m=atlas_clock_shr_jam(m,1);ae++;}
 while(m<0x0080000000000000UL&&ae>1){m<<=1;ae--;}
 ulong tail=m&7,significand=m>>3;
 if(tail>4||(tail==4&&(significand&1)))significand++;
 if(significand>=0x0020000000000000UL){significand>>=1;ae++;}
 ulong sign=as?0x8000000000000000UL:0;
 if(ae>=2047)return sign|0x7ff0000000000000UL;
 return sign|(significand>=hidden?ulong(ae)<<52:0)|(significand&fraction);
}
bool atlas_clock_compare(ulong a,ulong b,ulong kind){
 bool equal=a==b||((a&0x7fffffffffffffffUL)==0&&(b&0x7fffffffffffffffUL)==0);
 bool negative_a=(a>>63)!=0,negative_b=(b>>63)!=0;
 bool less=!equal&&(negative_a!=negative_b?negative_a:negative_a?a>b:a<b);
 return kind==0?!less&&!equal:kind==1?!less:kind==2?less:kind==3?less||equal:kind==4?equal:!equal;
}
ulong atlas_clock_from_float(float x){
 uint raw=as_type<uint>(x),fraction=raw&0x007fffffu,exponent=(raw>>23)&255;
 ulong sign=ulong(raw>>31)<<63;
 if(exponent==255)return sign|0x7ff0000000000000UL|(ulong(fraction)<<29);
 if(exponent==0){
  if(fraction==0)return sign;
  int e=-126;while((fraction&0x00800000u)==0){fraction<<=1;e--;}
  return sign|(ulong(e+1023)<<52)|(ulong(fraction&0x007fffffu)<<29);
 }
 return sign|(ulong(exponent+896)<<52)|(ulong(fraction)<<29);
}
