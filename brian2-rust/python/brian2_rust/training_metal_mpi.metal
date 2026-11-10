// Host-driven MPI barriers separate neuron/edge forward and reverse phases.
// Each rank computes only its contiguous target-neuron domain. The host sums
// float32 buffers in rank order using the existing float64 MPI transport.
kernel void atlas_graph_mpi_phase(
 device const ulong* m [[buffer(0)]], device const float* p [[buffer(1)]],
 device const float* x [[buffer(2)]], device const float* w [[buffer(3)]],
 device const float* initial [[buffer(4)]], device const ulong* labels [[buffer(5)]],
 device float* pre [[buffer(6)]], device float* spike [[buffer(7)]],
 device float* membrane [[buffer(8)]], device float* grad [[buffer(9)]],
 device float* initial_grad [[buffer(10)]], device float* logits [[buffer(11)]],
 device float* losses [[buffer(12)]], device float* carry [[buffer(13)]],
 device float* ds [[buffer(14)]], device float* previous [[buffer(15)]], uint b [[thread_position_in_grid]]) {
 ulong B=m[0],T=m[1],L=m[2],I=m[3],N=m[4],E=m[5],C=m[16+L],G=m[10];
 if(b>=B)return;
 ulong base=ulong(b)*N,cells=B*T*N;
 ulong ranks=m[m[15]],rank=m[m[15]+1],phase=m[m[15]+2],t=m[m[15]+3];
 ulong at=(ulong(b)*T+t)*N;
 if(phase==0){
 if(m[80]){
  ulong h=m[80],begin=m[h+1]+ulong(b)*m[h+2],end=begin+m[h+2];
  for(ulong k=begin;k<end;k++)pre[k]=0;
  for(ulong j=0;j<m[h+4];j++){
   ulong q=m[h+3]+3*j,slot=ulong(uint(atlas_dynamic_int(p[q])));
   if(slot>=begin&&slot<end){pre[slot]=1;pre[slot+1]=p[q+1];pre[slot+2]=p[q+2];pre[slot+3]=0;}
  }
  if(m[h+6])for(ulong k=0;k<T*N*2;k++)pre[m[h+5]+ulong(b)*T*N*2+k]=0;
 }
 for(ulong k=0;k<N;k++){membrane[base+k]=initial[base+k];carry[base+k]=0;}
 for(ulong e=0;e<E;e++)grad[ulong(b)*E+e]=0;
 return;}
 if(phase==1){
  for(ulong l=0;l<L;l++)for(ulong j=0;j<m[17+l];j++){
   ulong k=m[40+l]+j;
   if(k*ranks/N!=rank){membrane[base+k]=0;spike[at+k]=0;continue;}
   if(m[11])pre[2*cells+at+k]=membrane[base+k];
   float u=m[11]?atlas_scalar_equation(m,p,w,l,j,b,t,membrane[base+k],grad,ulong(b)*E,0,false,pre):p[l]*membrane[base+k];
   ulong threshold_id=m[14]?m[m[14]+l]:0;float theta=threshold_id?w[threshold_id-1]:p[L+l];
   float s=float(u>theta);
   pre[at+k]=u;spike[at+k]=s;membrane[base+k]=m[6]?u:u-theta*s;
  }
 return;}
 if(phase==2){
  for(ulong e=0;e<G;e++){
   ulong q=84+4*e,source=m[q+1],target=m[q+2],param=m[q+3];
   if(target*ranks/N!=rank)continue;
   float input=m[q]?x[(ulong(b)*T+t)*I+source]:spike[at+source];
   membrane[base+target]+=w[param]*input;
  }
  for(ulong k=0;k<N;k++){
   pre[cells+at+k]=membrane[base+k];
   if(m[6])membrane[base+k]*=1-spike[at+k];
  } return;}
 if(phase==3){
 float scale=p[2*L+2];
 for(ulong j=0;j<C;j++){
  float sum=0;for(ulong t=0;t<T;t++)sum+=spike[(ulong(b)*T+t)*N+m[39+L]+j]*scale/float(T);
  logits[ulong(b)*C+j]=sum;
 }
 float maxlog=-INFINITY;for(ulong j=0;j<C;j++)maxlog=max(maxlog,logits[ulong(b)*C+j]);
 float denom=0;for(ulong j=0;j<C;j++)denom+=exp(logits[ulong(b)*C+j]-maxlog);
 losses[b]=maxlog+log(denom)-logits[ulong(b)*C+labels[b]];
 return;}
 if(phase==4){
 float scale=p[2*L+2],maxlog=-INFINITY;
 for(ulong j=0;j<C;j++)maxlog=max(maxlog,logits[ulong(b)*C+j]);
 float denom=0;for(ulong j=0;j<C;j++)denom+=exp(logits[ulong(b)*C+j]-maxlog);
  for(ulong k=0;k<N;k++){ds[base+k]=0;previous[base+k]=0;}
  if(rank==0)for(ulong j=0;j<C;j++){
   float d=(exp(logits[ulong(b)*C+j]-maxlog)/denom-float(j==labels[b]))/float(B);
   ds[base+m[39+L]+j]=d*scale/float(T);
  }
  for(ulong e=0;e<G;e++){
   ulong q=84+4*e,source=m[q+1],target=m[q+2],param=m[q+3];
   if(target*ranks/N!=rank)continue;
   float g=carry[base+target]*(m[6]?1-spike[at+target]:1);
   float input=m[q]?x[(ulong(b)*T+t)*I+source]:spike[at+source];
   grad[ulong(b)*E+param]+=g*input;
   if(!m[q])ds[base+source]+=g*w[param];
  }
 return;}
 if(phase==5){
  for(ulong l=0;l<L;l++)for(ulong j=0;j<m[17+l];j++){
   ulong k=m[40+l]+j;if(k*ranks/N!=rank)continue;float u=pre[at+k],s=spike[at+k];
   ulong threshold_id=m[14]?m[m[14]+l]:0;float theta=threshold_id?w[threshold_id-1]:p[L+l];
   float q=1+p[2*L]*abs(u-theta),phi=p[2*L+1]/(q*q);
   float derivative=m[6]?(1-s-(m[7]?0:pre[cells+at+k]*phi)):(1-(m[7]?0:theta*phi));
   if(threshold_id){
    float reset_theta=m[6]?(m[7]?0:pre[cells+at+k]*phi):(-s+(m[7]?0:theta*phi));
    grad[ulong(b)*E+threshold_id-1]+=carry[base+k]*reset_theta-ds[base+k]*phi;
   }
   float du=carry[base+k]*derivative+ds[base+k]*phi;
   float score_adjoint=0;
   if(m[80]&&(m[m[11]+3*l]&256)){
    float values[1]={pre[2*cells+at+k]},adjoints[1]={0};int sources[1]={int(k)};
    ulong noise=m[m[81]]+(ulong(b)*T+t)*m[m[81]+1]+m[m[81]+3+2*l]+j*m[m[81]+2+2*l];
    float time=m[82]?p[m[82]+t]:0;bool valid=false;uint scored=0,needed=0;
    atlas_dynamic_equation_scored(m,p,w,m[11]+3*l,j,time,0,0,noise,values,grad,
      ulong(b)*E,0,true,adjoints,valid,losses[b]/float(B),true,scored,needed,sources,1,2,pre,false);
    if(!valid)losses[b]=NAN;
    if(m[m[80]+7])pre[m[m[80]+5]+((ulong(b)*T+t)*N+k)*2]=float(needed);
    score_adjoint=adjoints[0];
   }
   previous[base+k]=(m[11]?atlas_scalar_equation(m,p,w,l,j,b,t,pre[2*cells+at+k],grad,ulong(b)*E,du,true,pre):p[l]*du)+score_adjoint;
  }
  for(ulong k=0;k<N;k++)carry[base+k]=(t>0&&t%m[8]==0)?0:previous[base+k]; return;}
 if(phase==6)for(ulong k=0;k<N;k++)initial_grad[base+k]=carry[base+k];
}
