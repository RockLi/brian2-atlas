#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <math.h>

// MSL is passed as a generated C string in the isolated build directory.
#include "training_shader.h"

// A weak boundary gradient replays the complete network many times in one
// process. Compile immutable shader/pipeline objects once per loaded library;
// request buffers, queues and stochastic state remain private to each call.
static id<MTLDevice> atlas_training_device(void) {
 static dispatch_once_t once;static id<MTLDevice> device;
 dispatch_once(&once, ^{device=MTLCreateSystemDefaultDevice();});
 return device;
}
static id<MTLComputePipelineState> atlas_training_pipeline(NSString* name,NSError** error) {
 static dispatch_once_t once;static id<MTLLibrary> library;
 static NSError* compile_error;static NSMutableDictionary* pipelines;
 id<MTLDevice> device=atlas_training_device();
 dispatch_once(&once, ^{
  MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
  NSError* initial_error=nil;
  library=[device newLibraryWithSource:[NSString stringWithUTF8String:ATLAS_TRAIN_SHADER] options:options error:&initial_error];
  compile_error=initial_error;
  pipelines=[NSMutableDictionary new];
 });
 if(!library){if(error)*error=compile_error;return nil;}
 @synchronized(device) {
  id<MTLComputePipelineState> pipeline=pipelines[name];
  if(!pipeline){
   id<MTLFunction> function=[library newFunctionWithName:name];
   pipeline=function?[device newComputePipelineStateWithFunction:function error:error]:nil;
   if(pipeline)pipelines[name]=pipeline;
  }
  return pipeline;
 }
}
int b2_train_metal_v2(const uint64_t* meta,const float* params,const float* inputs,
 const float* weights,const float* initial,const uint64_t* labels,float* pre,
 float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,
 float* losses,char* message,size_t capacity) {
 @autoreleasepool {
  NSError* error=nil;
  id<MTLDevice> device=atlas_training_device();
  if(!device){snprintf(message,capacity,"no Metal GPU");return 1;}
  id<MTLComputePipelineState> pipeline=atlas_training_pipeline(meta[83]?@"atlas_state_phase":meta[15]?@"atlas_graph_mpi_phase":((meta[10]||meta[11])?@"atlas_graph_bptt":@"atlas_lif_bptt"),&error);
  if(!pipeline){snprintf(message,capacity,"%s",error.localizedDescription.UTF8String?:"Metal shader failed");return 2;}
  uint64_t B=meta[0],T=meta[1],L=meta[2],I=meta[3],N=meta[4],E=meta[5],C=meta[16+L];
  uint64_t M=meta[83]?meta[meta[83]]:N;
  BOOL distributed=meta[15]&&meta[meta[15]]>1;
  size_t lengths[]={(meta[12]?meta[12]:84+meta[10]*4)*8,(meta[13]?meta[13]:2*L+3)*4,B*T*I*4,E*4,B*M*4,B*8,4*(meta[80]?meta[meta[80]]:B*T*(meta[83]?2*M+N:N*(meta[11]?3:meta[10]?2:1))),B*T*N*4,B*M*4,B*E*4,B*M*4,B*C*4,B*4,B*M*4,B*M*4,B*M*4};
  const void* source[]={meta,params,inputs,weights,initial,labels,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
  id<MTLBuffer> buffers[16];
  for(int i=0;i<16;i++){
   buffers[i]=source[i]?[device newBufferWithBytes:source[i] length:lengths[i] options:MTLResourceStorageModeShared]:[device newBufferWithLength:lengths[i] options:MTLResourceStorageModeShared];
   if(!buffers[i]){snprintf(message,capacity,"Metal allocation failed");return 3;}
  }
  id<MTLCommandQueue> queue=[device newCommandQueue];
  // MPI is initialized/finalized by Rust; this library only calls the
  // checked reduction symbol from that same shim, between completed kernels.
  void* mpi=NULL;
  int (*sum)(double*,uint64_t)=NULL;
  double* scratch=NULL;
  if(distributed){
   const char* path=getenv("B2_TRAIN_MPI_LIB");
   mpi=path?dlopen(path,RTLD_NOW):NULL;
   sum=mpi?(int(*)(double*,uint64_t))dlsym(mpi,"b2_train_mpi_sum"):NULL;
   scratch=malloc(M*sizeof(double));
   if(!sum||!scratch){if(mpi)dlclose(mpi);free(scratch);snprintf(message,capacity,"Metal MPI transport unavailable");return 6;}
  }
  id<MTLBuffer> __strong* shared_buffers=buffers;
  BOOL (^dispatch)(uint64_t,uint64_t)=^BOOL(uint64_t phase,uint64_t tick){
   if(meta[15]){
    uint64_t* mutable_meta=shared_buffers[0].contents;
    mutable_meta[meta[15]+2]=phase;mutable_meta[meta[15]+3]=tick;
   }
   id<MTLCommandBuffer> command=[queue commandBuffer];
   id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
   if(!encoder){snprintf(message,capacity,"Metal encoder unavailable");return NO;}
   [encoder setComputePipelineState:pipeline];
   for(int i=0;i<16;i++)[encoder setBuffer:shared_buffers[i] offset:0 atIndex:i];
   NSUInteger width=MIN((NSUInteger)B,pipeline.maxTotalThreadsPerThreadgroup);
   [encoder dispatchThreads:MTLSizeMake(B,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
   [encoder endEncoding];[command commit];[command waitUntilCompleted];
   if(command.status!=MTLCommandBufferStatusCompleted){snprintf(message,capacity,"%s",command.error.localizedDescription.UTF8String?:"Metal command failed");return NO;}
   return YES;
  };
  BOOL (^reduce)(int,uint64_t,uint64_t,uint64_t)=^BOOL(int buffer,uint64_t offset,uint64_t stride,uint64_t count){
   if(!distributed)return YES;
   float* values=shared_buffers[buffer].contents;
   for(uint64_t b=0;b<B;b++){
    for(uint64_t k=0;k<count;k++)scratch[k]=values[offset+b*stride+k];
    if(sum(scratch,count)){snprintf(message,capacity,"Metal MPI ordered reduction failed");return NO;}
    for(uint64_t k=0;k<count;k++)values[offset+b*stride+k]=(float)scratch[k];
   }
   return YES;
  };
  BOOL ok=dispatch(0,0);
  if(meta[15]){
   for(uint64_t t=0;ok&&t<T;t++){
    ok=dispatch(1,t)&&reduce(7,t*N,T*N,N)&&dispatch(2,t)&&reduce(8,0,M,M);
   }
   if(ok)ok=dispatch(3,0);
   if(meta[9]&1)for(uint64_t t=T;ok&&t-->0;){
    ok=dispatch(4,t)&&reduce(14,0,M,N)&&dispatch(5,t)&&reduce(13,0,M,M);
   }
   if(ok)ok=dispatch(6,0);
  }
  free(scratch);if(mpi)dlclose(mpi);
  if(!ok)return 5;
  void* target[]={pre,spikes,membrane,gradients,initial_grad,logits,losses};
  for(int i=0;i<7;i++)memcpy(target[i],buffers[i+6].contents,lengths[i+6]);
  return 0;
 }
}

// v3 equation metadata cannot be handed to a legacy v2 library.
int b2_train_metal_v3(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* pre,float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,float* losses,char* error,size_t capacity) {
 return b2_train_metal_v2(m,p,x,w,initial,y,pre,spikes,membrane,gradients,initial_grad,logits,losses,error,capacity);
}

// Distinct ABI symbol prevents old one-dispatch libraries ignoring MPI metadata.
int b2_train_metal_mpi_v1(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* pre,float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,float* losses,char* error,size_t capacity) {
 return b2_train_metal_v2(m,p,x,w,initial,y,pre,spikes,membrane,gradients,initial_grad,logits,losses,error,capacity);
}

int b2_train_metal_v4r6(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* pre,float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,float* losses,char* error,size_t capacity) {
 return b2_train_metal_v2(m,p,x,w,initial,y,pre,spikes,membrane,gradients,initial_grad,logits,losses,error,capacity);
}

int b2_train_metal_v4r7(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* pre,float* spikes,float* membrane,float* gradients,float* initial_grad,float* logits,float* losses,char* error,size_t capacity) {
 return b2_train_metal_v2(m,p,x,w,initial,y,pre,spikes,membrane,gradients,initial_grad,logits,losses,error,capacity);
}

// Separate v5 metadata and symbol: no legacy layer/tick interpretation.
int b2_train_metal_dynamic_single_v5r9(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* tape,float* spikes,float* live,float* gradients,float* initial_grad,float* logits,float* losses,char* message,size_t capacity) {
 @autoreleasepool {
  NSError* error=nil;id<MTLDevice> device=atlas_training_device();
  if(!device){snprintf(message,capacity,"no Metal GPU");return 1;}
  id<MTLComputePipelineState> pipeline=atlas_training_pipeline(@"atlas_dynamic_bptt",&error);
  if(!pipeline){snprintf(message,capacity,"%s",error.localizedDescription.UTF8String?:"dynamic Metal shader failed");return 2;}
  uint64_t B=m[0],T=m[1],I=m[2],N=m[3],W=m[4],E=m[5],C=m[6],Q=m[8];
  size_t lengths[]={m[11]*8,m[12]*4,B*T*I*4,E*4,B*W*4,B*8,B*T*Q*4,B*T*N*4,B*W*4,B*E*4,B*W*4,B*C*4,B*4,B*N*4};
  const void* source[]={m,p,x,w,initial,y,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
  id<MTLBuffer> buffers[14];
  for(int j=0;j<14;j++){
   buffers[j]=source[j]?[device newBufferWithBytes:source[j] length:lengths[j] options:MTLResourceStorageModeShared]:[device newBufferWithLength:lengths[j] options:MTLResourceStorageModeShared];
   if(!buffers[j]){snprintf(message,capacity,"dynamic Metal allocation failed");return 3;}
  }
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  if(!encoder){snprintf(message,capacity,"dynamic Metal encoder unavailable");return 4;}
  [encoder setComputePipelineState:pipeline];for(int j=0;j<14;j++)[encoder setBuffer:buffers[j] offset:0 atIndex:j];
  [encoder dispatchThreads:MTLSizeMake(B,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(B,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted){snprintf(message,capacity,"%s",command.error.localizedDescription.UTF8String?:"dynamic Metal command failed");return 5;}
  void* target[]={tape,spikes,live,gradients,initial_grad,logits,losses};
  for(int j=0;j<7;j++)memcpy(target[j],buffers[j+6].contents,lengths[j+6]);
  return 0;
 }
}

int b2_train_metal_v5r9(const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* tape,float* spikes,float* live,float* gradients,float* initial_grad,float* logits,float* losses,char* message,size_t capacity) {
 if(m[m[14]]==1)return b2_train_metal_dynamic_single_v5r9(m,p,x,w,initial,y,tape,spikes,live,gradients,initial_grad,logits,losses,message,capacity);
 @autoreleasepool {
  NSError* error=nil;id<MTLDevice> device=atlas_training_device();
  if(!device){snprintf(message,capacity,"no Metal GPU");return 1;}
  id<MTLComputePipelineState> pipeline=atlas_training_pipeline(@"atlas_dynamic_mpi_phase",&error);
  if(!pipeline){snprintf(message,capacity,"%s",error.localizedDescription.UTF8String?:"dynamic MPI Metal shader failed");return 2;}
  uint64_t B=m[0],T=m[1],I=m[2],N=m[3],W=m[4],E=m[5],C=m[6],A=m[7],Q=m[8];
  uint64_t D=((m[13]==34&&m[33]==1)||(m[13]==36&&m[33]==2))?194:130;
  size_t lengths[]={m[11]*8,m[12]*4,B*T*I*4,E*4,B*W*4,B*8,B*T*Q*4,B*T*N*4,B*W*4,B*E*4,B*W*4,B*C*4,B*4,B*N*4,B*D*4};
  const void* source[]={m,p,x,w,initial,y,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
  id<MTLBuffer> buffers[15];
  for(int j=0;j<15;j++){
   buffers[j]=source[j]?[device newBufferWithBytes:source[j] length:lengths[j] options:MTLResourceStorageModeShared]:[device newBufferWithLength:lengths[j] options:MTLResourceStorageModeShared];
   if(!buffers[j]){snprintf(message,capacity,"dynamic MPI Metal allocation failed");return 3;}
  }
  const char* path=getenv("B2_TRAIN_MPI_LIB");void* mpi=path?dlopen(path,RTLD_NOW):NULL;
  int (*sum)(double*,uint64_t)=mpi?(int(*)(double*,uint64_t))dlsym(mpi,"b2_train_mpi_sum"):NULL;
  if(!sum){if(mpi)dlclose(mpi);snprintf(message,capacity,"dynamic Metal MPI transport unavailable");return 6;}
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLBuffer> __strong* shared=buffers;
  BOOL (^dispatch)(uint64_t,uint64_t,uint64_t)=^BOOL(uint64_t phase,uint64_t tick,uint64_t action){
   uint64_t* control=shared[0].contents;control[m[14]+2]=phase;control[m[14]+3]=tick;control[m[14]+4]=action;
   id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
   if(!encoder){snprintf(message,capacity,"dynamic MPI Metal encoder unavailable");return NO;}
   [encoder setComputePipelineState:pipeline];for(int j=0;j<15;j++)[encoder setBuffer:shared[j] offset:0 atIndex:j];
   [encoder dispatchThreads:MTLSizeMake(B,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(B,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
   [encoder endEncoding];[command commit];[command waitUntilCompleted];
   if(command.status!=MTLCommandBufferStatusCompleted){snprintf(message,capacity,"%s",command.error.localizedDescription.UTF8String?:"dynamic MPI Metal command failed");return NO;}
   return YES;
  };
  BOOL (^reduce)(uint64_t,uint64_t,BOOL)=^BOOL(uint64_t count,uint64_t h,BOOL forward){
   float* values=shared[14].contents;double scratch[130],cached[64];
   for(uint64_t b=0;b<B;b++){
    for(uint64_t k=0;k<count;k++){
     BOOL integer=forward&&!m[h+6]&&k<m[h+3]&&m[28]&&m[m[28]+m[m[h+4]+k]];
     if(integer){int32_t bits;memcpy(&bits,values+b*D+k,4);scratch[k]=bits;}
     else scratch[k]=values[b*D+k];
    }
    scratch[count]=values[b*D+129];
    if(sum(scratch,count+1)){snprintf(message,capacity,"dynamic Metal MPI reduction failed");return NO;}
    if(scratch[count]!=0){snprintf(message,capacity,"nonfinite or invalid dynamic Metal MPI action");return NO;}
    for(uint64_t k=0;k<count;k++){
     if(!isfinite(scratch[k])){snprintf(message,capacity,"nonfinite dynamic Metal MPI action");return NO;}
     BOOL integer=forward&&!m[h+6]&&k<m[h+3]&&m[28]&&m[m[28]+m[m[h+4]+k]];
     if(integer){
      if(scratch[k]<-2147483648.0||scratch[k]>2147483647.0||trunc(scratch[k])!=scratch[k]){snprintf(message,capacity,"integer MPI value outside int32");return NO;}
      int32_t bits=(int32_t)scratch[k];memcpy(values+b*D+k,&bits,4);
     }else values[b*D+k]=(float)scratch[k];
    }
    if(forward&&D==194&&m[h+5]&&(m[m[h+5]]&1024)){
     for(uint64_t k=0;k<64;k++){
      if(k%4==1||k%4==3){int32_t bits;memcpy(&bits,values+b*D+130+k,4);cached[k]=bits;}
      else cached[k]=values[b*D+130+k];
     }
     if(sum(cached,64)){snprintf(message,capacity,"shared Poisson MPI reduction failed");return NO;}
     for(uint64_t k=0;k<64;k++){
      if(!isfinite(cached[k])){snprintf(message,capacity,"invalid shared Poisson MPI value");return NO;}
      if(k%4==1||k%4==3){
       if(cached[k]<-2147483648.0||cached[k]>2147483647.0||trunc(cached[k])!=cached[k]){snprintf(message,capacity,"invalid shared Poisson integer payload");return NO;}
       int32_t bits=(int32_t)cached[k];memcpy(values+b*D+130+k,&bits,4);
      }else values[b*D+130+k]=(float)cached[k];
     }
    }
   }
   return YES;
  };
  BOOL ok=dispatch(0,0,0);
  for(uint64_t t=0;ok&&t<T;t++)for(uint64_t a=0;ok&&a<A;a++){
   uint64_t h=m[13]+16*a;if(!m[m[30]+2+t*(m[23]+1)+m[m[31]+a]]||(!m[h+6]&&!m[h+10]))continue;
   ok=dispatch(1,t,a)&&reduce(m[h+6]?1:m[h+3]*(m[h+15]?2:1),h,YES)&&dispatch(2,t,a);
  }
  if(ok)ok=dispatch(3,0,0);
  if(m[9])for(uint64_t t=T;ok&&t-->0;){
   ok=dispatch(4,t,0);
   for(uint64_t a=A;ok&&a-->0;){uint64_t h=m[13]+16*a;if(!m[m[30]+2+t*(m[23]+1)+m[m[31]+a]]||(!m[h+6]&&!m[h+10]))continue;ok=dispatch(5,t,a)&&reduce(m[h+6]?1:m[h+1]+m[h+3]+1,h,NO)&&dispatch(6,t,a);}
   uint64_t frame=m[m[30]+1+t*(m[23]+1)];
   if(ok&&m[10]&&frame>1&&(frame-1)%m[10]==0)ok=dispatch(8,t,0);
  }
  if(ok)ok=dispatch(7,0,0);dlclose(mpi);if(!ok)return 5;
  void* target[]={tape,spikes,live,gradients,initial_grad,logits,losses};
  for(int j=0;j<7;j++)memcpy(target[j],buffers[j+6].contents,lengths[j+6]);
  return 0;
 }
}

uint64_t b2_train_math_v1(void){return 1;}

uint64_t b2_train_poisson_v1(void){return 1;}

uint64_t b2_train_poisson_vjp_v1(void){return 1;}

uint64_t b2_train_poisson_boundary_v1(void){return 1;}

uint64_t b2_train_vjp_activity_v1(void){return 1;}

uint64_t b2_train_static_vjp_activity_v1(void){return 1;}
uint64_t b2_train_static_timed_input_v1(void){return 1;}
uint64_t b2_train_static_poisson_v1(void){return 1;}
uint64_t b2_train_scalar_context_v1(void){return 1;}
uint64_t b2_train_bitwise_v1(void){return 1;}
uint64_t b2_train_sequence_v1(void){return 1;}
uint64_t b2_train_boolean_eager_v1(void){return 1;}

uint64_t b2_train_poisson_shared_v1(void){return 1;}

uint64_t b2_train_poisson_checkpoint_v1(void){return 1;}
int b2_train_metal_poisson_validate_v1(uint64_t n,const uint64_t* keys,const float* rates,const int32_t* expected,uint32_t* work,uint32_t* errors,char* message,size_t capacity){
 if(n>100){snprintf(message,capacity,"Poisson checkpoint validation chunk exceeds 100");return 1;}
 if(!n)return 0;
 @autoreleasepool {
  NSError* error=nil;id<MTLDevice> device=atlas_training_device();
  if(!device){snprintf(message,capacity,"no Metal GPU");return 1;}
  id<MTLComputePipelineState> pipeline=atlas_training_pipeline(@"atlas_poisson_checkpoint_validate",&error);
  if(!pipeline){snprintf(message,capacity,"%s",error.localizedDescription.UTF8String?:"Poisson checkpoint Metal shader failed");return 2;}
  size_t lengths[]={8,n*8,n*4,n*4,n*4,n*4};const void* source[]={&n,keys,rates,expected,NULL,NULL};
  id<MTLBuffer> buffers[6];for(int j=0;j<6;j++){
   buffers[j]=source[j]?[device newBufferWithBytes:source[j] length:lengths[j] options:MTLResourceStorageModeShared]:[device newBufferWithLength:lengths[j] options:MTLResourceStorageModeShared];
   if(!buffers[j]){snprintf(message,capacity,"Poisson checkpoint Metal allocation failed");return 3;}
  }
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  if(!encoder){snprintf(message,capacity,"Poisson checkpoint Metal encoder unavailable");return 4;}
  [encoder setComputePipelineState:pipeline];for(int j=0;j<6;j++)[encoder setBuffer:buffers[j] offset:0 atIndex:j];
  [encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(n,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted){snprintf(message,capacity,"%s",command.error.localizedDescription.UTF8String?:"Poisson checkpoint Metal command failed");return 5;}
  memcpy(work,buffers[4].contents,n*4);memcpy(errors,buffers[5].contents,n*4);return 0;
 }
}

uint64_t b2_train_poisson_persistent_v1(void){return 1;}
