#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int poisson_metal_batch(const char* shader,uint64_t n,const float* rates,const uint64_t* keys,
                       int32_t* counts,uint32_t* draws,uint32_t* errors,float* scores,float* logps,
                       char* message,size_t capacity) {
 @autoreleasepool {
  static id<MTLDevice> device;static id<MTLComputePipelineState> pipeline;static NSString* compiled;
  if(!device)device=MTLCreateSystemDefaultDevice();
  if(!device){snprintf(message,capacity,"no Metal device");return 1;}
  NSString* source=[NSString stringWithUTF8String:shader];NSError* error=nil;
  if(!pipeline||![compiled isEqualToString:source]){
   MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
   id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
   id<MTLFunction> function=[library newFunctionWithName:@"poisson_probe"];
   pipeline=function?[device newComputePipelineStateWithFunction:function error:&error]:nil;
   if(!pipeline){snprintf(message,capacity,"%s",error.localizedDescription.UTF8String?:"Poisson shader failed");return 2;}
   compiled=source;
  }
  if(n==0)return 0;
  size_t lengths[]={n*4,n*8,n*4,n*4,n*4,n*4,n*4,8};
  const void* input[]={rates,keys,NULL,NULL,NULL,NULL,NULL,&n};id<MTLBuffer> buffers[8];
  for(int j=0;j<8;j++){
   buffers[j]=input[j]?[device newBufferWithBytes:input[j] length:lengths[j] options:MTLResourceStorageModeShared]:[device newBufferWithLength:lengths[j] options:MTLResourceStorageModeShared];
   if(!buffers[j]){snprintf(message,capacity,"Poisson buffer allocation failed");return 3;}
  }
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
  for(int j=0;j<8;j++)[encoder setBuffer:buffers[j] offset:0 atIndex:j];
  [encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(n,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted){snprintf(message,capacity,"%s",command.error.localizedDescription.UTF8String?:"Poisson dispatch failed");return 4;}
  void* output[]={counts,draws,errors,scores,logps};for(int j=0;j<5;j++)memcpy(output[j],buffers[j+2].contents,lengths[j+2]);
  return 0;
 }
}
