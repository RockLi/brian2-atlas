#include "training_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dlfcn.h>
#include <vector>
#include <exception>
#include <string.h>

// Kernels and collectives alternate only after cudaDeviceSynchronize. MPI
// receives host buffers, so a CUDA-aware MPI installation is not required.
static int execute_cuda(const uint64_t* m,const float* params,const float* inputs,
 const float* weights,const float* initial,const uint64_t* labels,float* pre,
 float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,
 float* losses,char* message,size_t capacity) {
 struct Resources {
  void* p[16]={};void* mpi=nullptr;
  ~Resources(){for(auto v:p)if(v)cudaFree(v);if(mpi)dlclose(mpi);}
 } resources;
 auto check=[&](cudaError_t status){
  if(status==cudaSuccess)return true;
  snprintf(message,capacity,"CUDA: %s",cudaGetErrorString(status));return false;
 };
 int count=0;
 if(!check(cudaGetDeviceCount(&count)))return 1;
 if(count<1){snprintf(message,capacity,"no CUDA GPU");return 1;}
 // Visibility can be bound per rank by the launcher. The default is logical
 // device 0, including multiple ranks intentionally sharing one GPU.
 int device=0;
 if(const char* choice=getenv("B2_TRAIN_CUDA_DEVICE")){
  errno=0;char* end=nullptr;long value=strtol(choice,&end,10);
  if(errno||end==choice||*end||value<0||value>=count){snprintf(message,capacity,"invalid B2_TRAIN_CUDA_DEVICE");return 1;}
  device=(int)value;
 }
 if(!check(cudaSetDevice(device)))return 1;
 const uint64_t B=m[0],T=m[1],L=m[2],I=m[3],N=m[4],E=m[5],C=m[16+L],M=m[83]?m[m[83]]:N;
 const bool phased=m[15]!=0,distributed=phased&&m[m[15]]>1;
 const size_t sizes[]={(m[12]?m[12]:84+m[10]*4)*8,(m[13]?m[13]:2*L+3)*4,B*T*I*4,E*4,B*M*4,B*8,
  4*(m[80]?m[m[80]]:B*T*(m[83]?2*M+N:N*(m[11]?3:m[10]?2:1))),B*T*N*4,B*M*4,B*E*4,B*M*4,B*C*4,B*4,B*M*4,B*M*4,B*M*4};
 const void* source[]={m,params,inputs,weights,initial,labels};
 for(int i=0;i<16;i++){
  if(!check(cudaMalloc(&resources.p[i],sizes[i])))return 2;
  if(i<6&&!check(cudaMemcpy(resources.p[i],source[i],sizes[i],cudaMemcpyHostToDevice)))return 3;
 }
 int (*sum)(double*,uint64_t)=nullptr;
 std::vector<float> staging;
 std::vector<double> scratch;
 if(distributed){
  const char* path=getenv("B2_TRAIN_MPI_LIB");resources.mpi=path?dlopen(path,RTLD_NOW):nullptr;
  sum=resources.mpi?(int(*)(double*,uint64_t))dlsym(resources.mpi,"b2_train_mpi_sum"):nullptr;
  if(!sum){snprintf(message,capacity,"CUDA MPI transport unavailable");return 6;}
  staging.resize(M);scratch.resize(M);
 }
 auto& p=resources.p;
 #define ARGS (const uint64_t*)p[0],(const float*)p[1],(const float*)p[2],(const float*)p[3],(const float*)p[4],(const uint64_t*)p[5],(float*)p[6],(float*)p[7],(float*)p[8],(float*)p[9],(float*)p[10],(float*)p[11],(float*)p[12],(float*)p[13],(float*)p[14],(float*)p[15]
 auto dispatch=[&](uint64_t phase,uint64_t tick){
  if(phased){
   const uint64_t control[]={phase,tick};
   if(!check(cudaMemcpy((uint64_t*)p[0]+m[15]+2,control,sizeof(control),cudaMemcpyHostToDevice)))return false;
  }
  if(m[83])atlas_state_phase<<<(B+63)/64,64>>>(ARGS);
  else if(phased)atlas_graph_mpi_phase<<<(B+63)/64,64>>>(ARGS);
  else if(m[10]||m[11])atlas_graph_bptt<<<(B+63)/64,64>>>(ARGS);
  else atlas_lif_bptt<<<(B+63)/64,64>>>(ARGS);
  return check(cudaGetLastError())&&check(cudaDeviceSynchronize());
 };
 #undef ARGS
 auto reduce=[&](int buffer,uint64_t offset,uint64_t stride,uint64_t length){
  if(!distributed)return true;
  for(uint64_t b=0;b<B;b++){
   float* values=(float*)p[buffer]+offset+b*stride;
   if(!check(cudaMemcpy(staging.data(),values,length*4,cudaMemcpyDeviceToHost)))return false;
   for(uint64_t k=0;k<length;k++)scratch[k]=staging[k];
   if(sum(scratch.data(),length)){snprintf(message,capacity,"CUDA MPI ordered reduction failed");return false;}
   for(uint64_t k=0;k<length;k++)staging[k]=(float)scratch[k];
   if(!check(cudaMemcpy(values,staging.data(),length*4,cudaMemcpyHostToDevice)))return false;
  }
  return true;
 };
 if(!dispatch(0,0))return 4;
 if(phased){
  for(uint64_t t=0;t<T;t++)if(!(dispatch(1,t)&&reduce(7,t*N,T*N,N)&&dispatch(2,t)&&reduce(8,0,M,M)))return 4;
  if(!dispatch(3,0))return 4;
  if(m[9]&1)for(uint64_t t=T;t-->0;)if(!(dispatch(4,t)&&reduce(14,0,M,N)&&dispatch(5,t)&&reduce(13,0,M,M)))return 4;
  if(!dispatch(6,0))return 4;
 }
 void* target[]={pre,spikes,membrane,gradients,initial_grad,logits,losses};
 for(int i=0;i<7;i++)if(!check(cudaMemcpy(target[i],p[i+6],sizes[i+6],cudaMemcpyDeviceToHost)))return 5;
 return 0;
}

extern "C" int b2_train_cuda_v2(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* pre,float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,float* losses,char* error,size_t capacity) {
 try {return execute_cuda(m,p,x,w,initial,y,pre,spikes,membrane,gradients,initial_grad,logits,losses,error,capacity);}
 catch(const std::exception& e){snprintf(error,capacity,"CUDA host allocation/runtime: %s",e.what());return 7;}
 catch(...){snprintf(error,capacity,"CUDA host runtime failed");return 7;}
}
#define ALIAS(name) extern "C" int name(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* pre,float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,float* losses,char* error,size_t capacity) {return b2_train_cuda_v2(m,p,x,w,initial,y,pre,spikes,membrane,gradients,initial_grad,logits,losses,error,capacity);}
ALIAS(b2_train_cuda_v3)
ALIAS(b2_train_cuda_mpi_v1)
ALIAS(b2_train_cuda_v4r6)
ALIAS(b2_train_cuda_v4r7)

extern "C" int b2_train_cuda_dynamic_single_v5r9(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* tape,float* spikes,float* live,float* gradients,float* initial_grad,float* logits,float* losses,char* message,size_t capacity) {
 try {
  struct Resources {void* p[14]={};~Resources(){for(auto v:p)if(v)cudaFree(v);}} resources;
  auto check=[&](cudaError_t status){if(status==cudaSuccess)return true;snprintf(message,capacity,"CUDA: %s",cudaGetErrorString(status));return false;};
  int count=0;if(!check(cudaGetDeviceCount(&count)))return 1;
  if(count<1){snprintf(message,capacity,"no CUDA GPU");return 1;}
  int device=0;if(const char* choice=getenv("B2_TRAIN_CUDA_DEVICE")){
   errno=0;char* end=nullptr;long value=strtol(choice,&end,10);
   if(errno||end==choice||*end||value<0||value>=count){snprintf(message,capacity,"invalid B2_TRAIN_CUDA_DEVICE");return 1;}
   device=(int)value;
  }
  if(!check(cudaSetDevice(device)))return 1;
  const uint64_t B=m[0],T=m[1],I=m[2],N=m[3],W=m[4],E=m[5],C=m[6],Q=m[8];
  const size_t lengths[]={m[11]*8,m[12]*4,B*T*I*4,E*4,B*W*4,B*8,B*T*Q*4,B*T*N*4,B*W*4,B*E*4,B*W*4,B*C*4,B*4,B*N*4};
  const void* source[]={m,p,x,w,initial,y};
  for(int j=0;j<14;j++){
   if(!check(cudaMalloc(&resources.p[j],lengths[j])))return 2;
   if(j<6&&!check(cudaMemcpy(resources.p[j],source[j],lengths[j],cudaMemcpyHostToDevice)))return 3;
  }
  auto& v=resources.p;
  atlas_dynamic_bptt<<<(B+63)/64,64>>>((const uint64_t*)v[0],(const float*)v[1],(const float*)v[2],(const float*)v[3],(const float*)v[4],(const uint64_t*)v[5],(float*)v[6],(float*)v[7],(float*)v[8],(float*)v[9],(float*)v[10],(float*)v[11],(float*)v[12],(float*)v[13]);
  if(!check(cudaGetLastError())||!check(cudaDeviceSynchronize()))return 4;
  void* target[]={tape,spikes,live,gradients,initial_grad,logits,losses};
  for(int j=0;j<7;j++)if(!check(cudaMemcpy(target[j],v[j+6],lengths[j+6],cudaMemcpyDeviceToHost)))return 5;
  return 0;
 }catch(const std::exception& e){snprintf(message,capacity,"dynamic CUDA host runtime: %s",e.what());return 7;}
 catch(...){snprintf(message,capacity,"dynamic CUDA host runtime failed");return 7;}
}

extern "C" int b2_train_cuda_v5r9(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* tape,float* spikes,float* live,float* gradients,float* initial_grad,float* logits,float* losses,char* message,size_t capacity) {
 if(m[m[14]]==1)return b2_train_cuda_dynamic_single_v5r9(m,p,x,w,initial,y,tape,spikes,live,gradients,initial_grad,logits,losses,message,capacity);
 try {
  struct Resources {void* p[15]={};void* mpi=nullptr;~Resources(){for(auto v:p)if(v)cudaFree(v);if(mpi)dlclose(mpi);}} resources;
  auto check=[&](cudaError_t status){if(status==cudaSuccess)return true;snprintf(message,capacity,"CUDA: %s",cudaGetErrorString(status));return false;};
  int count=0;if(!check(cudaGetDeviceCount(&count)))return 1;
  if(count<1){snprintf(message,capacity,"no CUDA GPU");return 1;}
  int device=0;if(const char* choice=getenv("B2_TRAIN_CUDA_DEVICE")){
   errno=0;char* end=nullptr;long value=strtol(choice,&end,10);
   if(errno||end==choice||*end||value<0||value>=count){snprintf(message,capacity,"invalid B2_TRAIN_CUDA_DEVICE");return 1;}device=(int)value;
  }
  if(!check(cudaSetDevice(device)))return 1;
  const uint64_t B=m[0],T=m[1],I=m[2],N=m[3],W=m[4],E=m[5],C=m[6],A=m[7],Q=m[8];
  const uint64_t D=((m[13]==34&&m[33]==1)||(m[13]==36&&m[33]==2))?194:130;
  const size_t lengths[]={m[11]*8,m[12]*4,B*T*I*4,E*4,B*W*4,B*8,B*T*Q*4,B*T*N*4,B*W*4,B*E*4,B*W*4,B*C*4,B*4,B*N*4,B*D*4};
  const void* source[]={m,p,x,w,initial,y};
  for(int j=0;j<15;j++){
   if(!check(cudaMalloc(&resources.p[j],lengths[j])))return 2;
   if(j<6&&!check(cudaMemcpy(resources.p[j],source[j],lengths[j],cudaMemcpyHostToDevice)))return 3;
  }
  const char* path=getenv("B2_TRAIN_MPI_LIB");resources.mpi=path?dlopen(path,RTLD_NOW):nullptr;
  auto sum=resources.mpi?(int(*)(double*,uint64_t))dlsym(resources.mpi,"b2_train_mpi_sum"):nullptr;
  if(!sum){snprintf(message,capacity,"dynamic CUDA MPI transport unavailable");return 6;}
  auto& v=resources.p;
  auto dispatch=[&](uint64_t phase,uint64_t tick,uint64_t action){
   const uint64_t control[]={phase,tick,action};
   if(!check(cudaMemcpy((uint64_t*)v[0]+m[14]+2,control,sizeof(control),cudaMemcpyHostToDevice)))return false;
   atlas_dynamic_mpi_phase<<<(B+63)/64,64>>>((const uint64_t*)v[0],(const float*)v[1],(const float*)v[2],(const float*)v[3],(const float*)v[4],(const uint64_t*)v[5],(float*)v[6],(float*)v[7],(float*)v[8],(float*)v[9],(float*)v[10],(float*)v[11],(float*)v[12],(float*)v[13],(float*)v[14]);
   return check(cudaGetLastError())&&check(cudaDeviceSynchronize());
  };
  auto reduce=[&](uint64_t count,uint64_t h,bool forward){
   float staging[194];double scratch[130],cached[64];
   for(uint64_t b=0;b<B;b++){
    float* values=(float*)v[14]+b*D;
    if(!check(cudaMemcpy(staging,values,D*4,cudaMemcpyDeviceToHost)))return false;
    for(uint64_t k=0;k<count;k++){
     bool integer=forward&&!m[h+6]&&k<m[h+3]&&m[28]&&m[m[28]+m[m[h+4]+k]];
     if(integer){int32_t bits;memcpy(&bits,staging+k,4);scratch[k]=bits;}else scratch[k]=staging[k];
    }
    scratch[count]=staging[129];
    if(sum(scratch,count+1)){snprintf(message,capacity,"dynamic CUDA MPI reduction failed");return false;}
    if(scratch[count]!=0){snprintf(message,capacity,"nonfinite or invalid dynamic CUDA MPI action");return false;}
    for(uint64_t k=0;k<count;k++){
     if(!isfinite(scratch[k])){snprintf(message,capacity,"nonfinite dynamic CUDA MPI action");return false;}bool integer=forward&&!m[h+6]&&k<m[h+3]&&m[28]&&m[m[28]+m[m[h+4]+k]];
     if(integer){
      if(scratch[k]<-2147483648.0||scratch[k]>2147483647.0||trunc(scratch[k])!=scratch[k]){snprintf(message,capacity,"integer MPI value outside int32");return false;}
      int32_t bits=(int32_t)scratch[k];memcpy(staging+k,&bits,4);
     }else staging[k]=(float)scratch[k];
    }
    if(!check(cudaMemcpy(values,staging,count*4,cudaMemcpyHostToDevice)))return false;
    if(forward&&D==194&&m[h+5]&&(m[m[h+5]]&1024)){
     for(uint64_t k=0;k<64;k++){
      if(k%4==1||k%4==3){int32_t bits;memcpy(&bits,staging+130+k,4);cached[k]=bits;}
      else cached[k]=staging[130+k];
     }
     if(sum(cached,64)){snprintf(message,capacity,"shared Poisson MPI reduction failed");return false;}
     for(uint64_t k=0;k<64;k++){
      if(!isfinite(cached[k])){snprintf(message,capacity,"invalid shared Poisson MPI value");return false;}
      if(k%4==1||k%4==3){
       if(cached[k]<-2147483648.0||cached[k]>2147483647.0||trunc(cached[k])!=cached[k]){snprintf(message,capacity,"invalid shared Poisson integer payload");return false;}
       int32_t bits=(int32_t)cached[k];memcpy(staging+130+k,&bits,4);
      }else staging[130+k]=(float)cached[k];
     }
     if(!check(cudaMemcpy(values+130,staging+130,64*4,cudaMemcpyHostToDevice)))return false;
    }
   }
   return true;
  };
  if(!dispatch(0,0,0))return 4;
  for(uint64_t t=0;t<T;t++)for(uint64_t a=0;a<A;a++){
   uint64_t h=m[13]+16*a;if(!m[m[30]+2+t*(m[23]+1)+m[m[31]+a]]||(!m[h+6]&&!m[h+10]))continue;
   if(!(dispatch(1,t,a)&&reduce(m[h+6]?1:m[h+3]*(m[h+15]?2:1),h,true)&&dispatch(2,t,a)))return 4;
  }
  if(!dispatch(3,0,0))return 4;
  if(m[9])for(uint64_t t=T;t-->0;){
   if(!dispatch(4,t,0))return 4;
   for(uint64_t a=A;a-->0;){uint64_t h=m[13]+16*a;if(!m[m[30]+2+t*(m[23]+1)+m[m[31]+a]]||(!m[h+6]&&!m[h+10]))continue;if(!(dispatch(5,t,a)&&reduce(m[h+6]?1:m[h+1]+m[h+3]+1,h,false)&&dispatch(6,t,a)))return 4;}
   uint64_t frame=m[m[30]+1+t*(m[23]+1)];
   if(m[10]&&frame>1&&(frame-1)%m[10]==0&&!dispatch(8,t,0))return 4;
  }
  if(!dispatch(7,0,0))return 4;
  void* target[]={tape,spikes,live,gradients,initial_grad,logits,losses};
  for(int j=0;j<7;j++)if(!check(cudaMemcpy(target[j],v[j+6],lengths[j+6],cudaMemcpyDeviceToHost)))return 5;
  return 0;
 }catch(const std::exception& e){snprintf(message,capacity,"dynamic CUDA MPI host runtime: %s",e.what());return 7;}
 catch(...){snprintf(message,capacity,"dynamic CUDA MPI host runtime failed");return 7;}
}

extern "C" uint64_t b2_train_math_v1(void){return 1;}

extern "C" uint64_t b2_train_poisson_v1(void){return 1;}

extern "C" uint64_t b2_train_poisson_vjp_v1(void){return 1;}

extern "C" uint64_t b2_train_poisson_boundary_v1(void){return 1;}

extern "C" uint64_t b2_train_vjp_activity_v1(void){return 1;}

extern "C" uint64_t b2_train_static_vjp_activity_v1(void){return 1;}
extern "C" uint64_t b2_train_static_timed_input_v1(void){return 1;}
extern "C" uint64_t b2_train_static_poisson_v1(void){return 1;}
uint64_t b2_train_scalar_context_v1(void){return 1;}
uint64_t b2_train_bitwise_v1(void){return 1;}
uint64_t b2_train_sequence_v1(void){return 1;}
uint64_t b2_train_boolean_eager_v1(void){return 1;}

extern "C" uint64_t b2_train_poisson_shared_v1(void){return 1;}

extern "C" uint64_t b2_train_poisson_checkpoint_v1(void){return 1;}
extern "C" int b2_train_cuda_poisson_validate_v1(uint64_t n,const uint64_t* keys,const float* rates,const int32_t* expected,uint32_t* work,uint32_t* errors,char* message,size_t capacity){
 if(n>100){snprintf(message,capacity,"Poisson checkpoint validation chunk exceeds 100");return 1;}if(!n)return 0;
 try {
  struct Resources{void* p[6]={};~Resources(){for(auto p:p)if(p)cudaFree(p);}} resources;
  auto check=[&](cudaError_t status){if(status==cudaSuccess)return true;snprintf(message,capacity,"CUDA: %s",cudaGetErrorString(status));return false;};
  int count=0;if(!check(cudaGetDeviceCount(&count)))return 1;
  if(count<1){snprintf(message,capacity,"no CUDA GPU");return 1;}
  int device=0;if(const char* choice=getenv("B2_TRAIN_CUDA_DEVICE")){
   errno=0;char* end=nullptr;long value=strtol(choice,&end,10);
   if(errno||end==choice||*end||value<0||value>=count){snprintf(message,capacity,"invalid B2_TRAIN_CUDA_DEVICE");return 1;}device=(int)value;
  }
  if(!check(cudaSetDevice(device)))return 1;
  size_t lengths[]={8,n*8,n*4,n*4,n*4,n*4};const void* source[]={&n,keys,rates,expected,nullptr,nullptr};
  for(int j=0;j<6;j++){
   if(!check(cudaMalloc(resources.p+j,lengths[j])))return 3;
   if(source[j]&&!check(cudaMemcpy(resources.p[j],source[j],lengths[j],cudaMemcpyHostToDevice)))return 3;
  }
  atlas_poisson_checkpoint_validate<<<(n+63)/64,64>>>((const uint64_t*)resources.p[0],(const uint64_t*)resources.p[1],(const float*)resources.p[2],(const int32_t*)resources.p[3],(uint32_t*)resources.p[4],(uint32_t*)resources.p[5]);
  if(!check(cudaGetLastError())||!check(cudaDeviceSynchronize()))return 4;
  if(!check(cudaMemcpy(work,resources.p[4],n*4,cudaMemcpyDeviceToHost))||!check(cudaMemcpy(errors,resources.p[5],n*4,cudaMemcpyDeviceToHost)))return 5;
  return 0;
 }catch(const std::exception& e){snprintf(message,capacity,"Poisson checkpoint CUDA host runtime: %s",e.what());return 7;}
 catch(...){snprintf(message,capacity,"Poisson checkpoint CUDA host runtime failed");return 7;}
}

extern "C" uint64_t b2_train_poisson_persistent_v1(void){return 1;}
