// v5r5 per-action owner compute / ordered collective / replicated apply.
kernel void atlas_dynamic_mpi_phase(
 device const ulong* m [[buffer(0)]],device const float* p [[buffer(1)]],
 device const float* input [[buffer(2)]],device const float* weights [[buffer(3)]],
 device const float* initial [[buffer(4)]],device const ulong* labels [[buffer(5)]],
 device float* tape [[buffer(6)]],device float* spikes [[buffer(7)]],
 device float* live [[buffer(8)]],device float* gradient [[buffer(9)]],
 device float* adjoint [[buffer(10)]],device float* logits [[buffer(11)]],
 device float* losses [[buffer(12)]],device float* ds [[buffer(13)]],
 device float* delta [[buffer(14)]], uint b [[thread_position_in_grid]]) {
 ulong B=m[0],T=m[1],I=m[2],N=m[3],W=m[4],E=m[5],C=m[6],Q=m[8];if(b>=B)return;
 ulong ctl=m[14],ranks=m[ctl],rank=m[ctl+1],phase=m[ctl+2],t=m[ctl+3],a=m[ctl+4];
 ulong base=ulong(b)*W,gb=ulong(b)*E,sb=ulong(b)*N,D=atlas_dynamic_cache_enabled(m)?194:130,db=ulong(b)*D;
 if(phase==0){
  for(ulong k=0;k<W;k++){live[base+k]=initial[base+k];adjoint[base+k]=0;}
  for(ulong k=0;k<E;k++)gradient[gb+k]=0;
  for(ulong k=0;k<T*N;k++)spikes[ulong(b)*T*N+k]=0;
  for(ulong k=0;k<T*Q;k++)tape[ulong(b)*T*Q+k]=0;
  atlas_dynamic_cache_initialize(m,p,tape,b);
  losses[b]=0;return;
 }
 if(phase==3){
  if(!isfinite(losses[b]))return;
  for(ulong j=0;j<C;j++){
   float value=0;for(ulong tick=0;tick<T;tick++)value+=spikes[(ulong(b)*T+tick)*N+N-C+j]*p[2]/float(m[m[30]]);
   logits[ulong(b)*C+j]=value;
  }
  float maximum=-INFINITY,total=0;
  for(ulong j=0;j<C;j++)maximum=max(maximum,logits[ulong(b)*C+j]);
  for(ulong j=0;j<C;j++)total+=exp(logits[ulong(b)*C+j]-maximum);
  losses[b]=maximum+log(total)-logits[ulong(b)*C+labels[b]];return;
 }
 if(phase==4){
  float maximum=-INFINITY,total=0;
  for(ulong j=0;j<C;j++)maximum=max(maximum,logits[ulong(b)*C+j]);
  for(ulong j=0;j<C;j++)total+=exp(logits[ulong(b)*C+j]-maximum);
  for(ulong k=0;k<N;k++)ds[sb+k]=0;
  if(m[32]||atlas_dynamic_frame(m,t))for(ulong j=0;j<C;j++)ds[sb+N-C+j]=(exp(logits[ulong(b)*C+j]-maximum)/total-float(j==labels[b]))/float(B)*p[2]/float(m[m[30]]);
  return;
 }
 if(phase==8){
  if(atlas_dynamic_cut(m,t))for(ulong k=0;k<W;k++)adjoint[base+k]=0;
  return;
 }
 if(phase==7){
  if(m[9]==1&&m[26]&&rank==0)for(ulong k=0;k<W;k++){ulong ref=m[m[27]+k];if(ref)gradient[gb+ref-1]+=adjoint[base+k];}
  return;
 }
 ulong h=m[13]+16*a,c=m[h+1],r=m[h+2],w=m[h+3],wr=m[h+4],at=(ulong(b)*T+t)*Q+m[h];
 ulong nt=(ulong(b)*T+t)*N,clock=m[22]+t*m[23],noise=m[24]+(ulong(b)*T+t)*m[25]+m[h+12];
 ulong cid=m[m[31]+a];float time=p[clock+cid];ulong exact_time=m[29]?m[m[29]+t*m[23]+cid]:0;float context[64],gstate[64];
 if(phase==1||phase==5)for(ulong j=0;j<D;j++)delta[db+j]=0;
 if(!atlas_dynamic_active(m,t,a)){
  return;
 }
 if(phase==1)atlas_dynamic_gather(m,h,live,base,tape,at,context);
 else for(ulong j=0;j<c;j++)context[j]=tape[at+j];
 for(ulong j=0;j<c;j++)gstate[j]=0;
 bool owned=m[h+13]*ranks/N==rank;
 ulong kind=m[h+7],index=m[h+8];float gate=kind==0?1:kind==1?input[(ulong(b)*T+t)*I+index]:kind==2?spikes[nt+index]:context[index];
 if(phase==1){
  if(!owned)return;
  if(m[h+6]){
   ulong k=m[h+6]-1,ref=m[h+14]?0:m[m[18]+k];float theta=m[h+14]?0:ref?weights[ref-1]:p[m[19]+k];
   if((c>1&&context[1]!=0&&context[1]!=1)||(m[h+14]==3&&context[0]!=0&&context[0]!=1)){delta[db+129]=1;return;}
   delta[db]=float((c==1||context[1]==1)&&(m[h+14]==2?context[0]>=theta:context[0]>theta));return;
  }
  if(!m[h+10])return;
  if(gate!=0&&!atlas_dynamic_reads_valid(m,h,tape,at)){delta[db+129]=1;return;}
  float out[64];
  if(!atlas_dynamic_outputs(m,p,weights,h,gate,time,exact_time,clock,noise,context,gradient,gb,out,gstate,tape)){delta[db+129]=1;return;}
  bool targets=atlas_dynamic_targets(m,h,live,base,tape,at,context,out);
  if(gate!=0&&!targets){delta[db+129]=1;return;}
  if(atlas_dynamic_cache_enabled(m)&&(m[m[h+5]]&1024))for(ulong s=0;s<16;s++){
   ulong at=atlas_dynamic_cache_slot(p,noise,s);
   if(at&&atlas_dynamic_cache_origin(p,tape,noise,s)){for(ulong k=0;k<4;k++)delta[db+130+4*s+k]=tape[at+k];}
  }
  for(ulong j=0;j<w;j++){
   delta[db+j]=out[j];if(m[h+15])delta[db+w+j]=float(atlas_dynamic_target(m,h,tape,at,j));
  }
  return;
 }
 if(phase==2){
  if(m[h+6]){ulong k=m[h+6]-1;spikes[nt+k]=delta[db];if(m[32])live[base+m[m[32]+k]]=delta[db];}
  else if(m[h+10]){
   if(atlas_dynamic_cache_enabled(m)&&(m[m[h+5]]&1024))for(ulong s=0;s<16;s++){
    if(delta[db+130+4*s]==1){ulong at=atlas_dynamic_cache_slot(p,noise,s);
     if(!at){losses[b]=NAN;return;}for(ulong k=0;k<4;k++)tape[at+k]=delta[db+130+4*s+k];
    }
   }
   if(m[h+15])for(ulong j=0;j<w;j++){
    int target=int(delta[db+w+j]);
    if(target>=int(W)||(gate!=0&&target<0)){losses[b]=NAN;return;}
    tape[at+2*c+j]=float(target);tape[at+2*c+w+j]=target>=0?live[base+ulong(target)]:0;
   }
   if(gate!=0)for(ulong j=0;j<w;j++)live[base+ulong(atlas_dynamic_target(m,h,tape,at,j))]=delta[db+j];
  }
  return;
 }
 if(phase==5){
  if(!owned)return;
  if(m[h+6]){
   if(m[9]==2)return;
   ulong k=m[h+6]-1,ref=m[h+14]?0:m[m[18]+k];float theta=m[h+14]?0:ref?weights[ref-1]:p[m[19]+k];
   float seed=ds[sb+k]+(m[32]?adjoint[base+m[m[32]+k]]:0);
   if(c==1||context[1]==1){float z=1+p[0]*abs(context[0]-theta),g=m[h+14]==3?seed:seed*p[1]/(z*z);delta[db]=g;if(ref)gradient[gb+ref-1]-=g;}
   return;
  }
  if(!m[h+10])return;
  bool differentiate=!m[h+9]&&(kind==2||(kind==3&&!m[m[16]+m[r+index]]));float dg=0;
  float old_gradient[64];
  if(!atlas_dynamic_reverse(m,p,weights,h,tape,at,adjoint,base,gate,differentiate,losses[b]/float(B),time,exact_time,clock,noise,context,gradient,gb,gstate,old_gradient,dg)){delta[db+129]=1;return;}
  for(ulong j=0;j<w;j++)delta[db+c+j]=old_gradient[j];
  for(ulong j=0;j<c;j++)delta[db+j]=gstate[j];delta[db+c+w]=dg;return;
 }
 if(phase==6){
  if(m[9]==2)return;
  if(m[h+6]){if(m[32])adjoint[base+m[m[32]+m[h+6]-1]]=0;adjoint[base+m[r]]+=delta[db];}
  else if(m[h+10]){
   bool differentiate=!m[h+9]&&(kind==2||(kind==3&&!m[m[16]+m[r+index]]));
   if(!(m[h+15]&&gate==0&&!differentiate)){
    float old_gradient[64];for(ulong j=0;j<c;j++)gstate[j]=delta[db+j];
    for(ulong j=0;j<w;j++)old_gradient[j]=delta[db+c+j];
    atlas_dynamic_apply_reverse(m,h,tape,at,adjoint,base,gstate,old_gradient);
    if(kind==2)ds[sb+index]+=delta[db+c+w];else if(kind==3&&!m[m[16]+m[r+index]])adjoint[base+m[r+index]]+=delta[db+c+w];
   }
  }
 }
}
