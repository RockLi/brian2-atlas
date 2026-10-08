#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <iostream>
using uint=unsigned int;
using ulong=unsigned long;
template<class T,class U> T as_type(U x){static_assert(sizeof(T)==sizeof(U));T y;std::memcpy(&y,&x,sizeof y);return y;}
#include "../python/brian2_rust/training_clock.metal"
int main(){
 std::mt19937_64 rng(19281127);
 for(int i=0;i<1000000;i++){
  ulong a=rng(),b=rng();
  if(((a>>52)&2047)==2047)a^=1UL<<52;
  if(((b>>52)&2047)==2047)b^=1UL<<52;
  double da=as_type<double>(a),db=as_type<double>(b);
  ulong expected=as_type<ulong>(da-db),actual=atlas_clock_sub(a,b);
  if(actual!=expected){std::cerr<<"sub "<<std::hex<<a<<" "<<b<<" "<<actual<<" "<<expected<<"\n";return 1;}
  bool comparisons[]={da>db,da>=db,da<db,da<=db,da==db,da!=db};
  for(int k=0;k<6;k++)if(atlas_clock_compare(a,b,k)!=comparisons[k])return 2;
  uint f=uint(rng());if(((f>>23)&255)==255)f^=1u<<23;
  float fv=as_type<float>(f);
  if(atlas_clock_from_float(fv)!=as_type<ulong>(double(fv)))return 3;
 }
 ulong cases[]={0,0x8000000000000000UL,1,0x8000000000000001UL,0x0010000000000000UL,0x7fefffffffffffffUL};
 for(ulong a:cases)for(ulong b:cases){
  double da=as_type<double>(a),db=as_type<double>(b);
  if(atlas_clock_sub(a,b)!=as_type<ulong>(da-db))return 4;
 }
 std::cout<<"1000000 finite random binary64 subtraction/comparison and float conversion cases, plus 36 edge pairs passed\n";
}
