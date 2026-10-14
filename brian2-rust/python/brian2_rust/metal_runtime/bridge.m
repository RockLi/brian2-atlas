// C ABI for the experimental Metal executor. Model decisions live in Python.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef struct {
    void *device; void *queue; void *pipeline; void *function;
    void *dag_storage; void *dag_sizes; void *dag_indirect;
} B2Metal;
void b2_metal_clear_dag(void *opaque) {
    B2Metal *handle = opaque;
    if (!handle) return;
    if (handle->dag_indirect) CFRelease(handle->dag_indirect);
    handle->dag_indirect = NULL;
    if (handle->dag_storage) CFRelease(handle->dag_storage);
    if (handle->dag_sizes) CFRelease(handle->dag_sizes);
    handle->dag_storage = NULL; handle->dag_sizes = NULL;
}
int b2_metal_move_dag(void *destination, void *source, uint32_t count, const uint8_t *reuse) {
    @autoreleasepool {
    B2Metal *to=destination,*from=source;
    if (!to || !from || to==from || to->dag_storage || !from->dag_storage) return 0;
    if (((__bridge id<MTLDevice>)to->device).registryID != ((__bridge id<MTLDevice>)from->device).registryID) return 0;
    // Keep matching slots only; release unmatched allocations before growing any.
    NSMutableArray *old=(__bridge NSMutableArray *)from->dag_storage;
    NSMutableArray *selected=[NSMutableArray arrayWithCapacity:count];
    for (uint32_t i=0;i<count;++i) {
        if (reuse[i] && i>=old.count) return 0;
        [selected addObject:reuse[i] ? old[i] : [NSNull null]];
    }
    to->dag_storage=(__bridge_retained void *)selected;
    b2_metal_clear_dag(from);
    return 1;
    }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void fail(char *error, size_t capacity, NSString *message) {
    if (capacity) snprintf(error, capacity, "%s", message.UTF8String ?: "Metal failure");
}
void *b2_metal_create(const char *source, const char *name, char *error, size_t capacity) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        // A headless/session configuration can have enumerated GPUs but no
        // system default device (observed on macOS 14.5 / M1 Ultra).
        if (!device) {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            device = devices.firstObject;
        }
        if (!device) { fail(error, capacity, @"No Metal device available"); return NULL; }
        NSError *problem = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        // SDK availability is separate from runtime OS availability.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else
#endif
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library = [device newLibraryWithSource:@(source) options:options error:&problem];
        if (!library) { fail(error, capacity, problem.localizedDescription); return NULL; }
        id<MTLFunction> function = [library newFunctionWithName:@(name)];
        if (!function) { fail(error, capacity, @"Missing planned Metal entry point"); return NULL; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&problem];
        if (!pipeline) { fail(error, capacity, problem.localizedDescription); return NULL; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { fail(error, capacity, @"Cannot create Metal command queue"); return NULL; }
        B2Metal *handle = calloc(1, sizeof(B2Metal));
        if (!handle) { fail(error, capacity, @"Cannot allocate Metal handle"); return NULL; }
        handle->device = (__bridge_retained void *)device;
        handle->queue = (__bridge_retained void *)queue;
        handle->pipeline = (__bridge_retained void *)pipeline;
        handle->function = (__bridge_retained void *)function;
        return handle;
    }
}
void b2_metal_destroy(void *opaque) {
    if (!opaque) return;
    B2Metal *handle = opaque;
    b2_metal_clear_dag(handle);
    CFRelease(handle->function); CFRelease(handle->pipeline); CFRelease(handle->queue); CFRelease(handle->device); free(handle);
}
void *b2_metal_clone_pipeline(void *opaque) {
    @autoreleasepool {
        B2Metal *old=opaque;
        if (!old) return NULL;
        id<MTLDevice> current=MTLCreateSystemDefaultDevice();
        if (!current) current=MTLCopyAllDevices().firstObject;
        if (!current || current.registryID!=((__bridge id<MTLDevice>)old->device).registryID) return NULL;
        B2Metal *copy=calloc(1,sizeof(B2Metal));
        if (!copy) return NULL;
        copy->device=(void *)CFRetain(old->device);
        copy->queue=(void *)CFRetain(old->queue);
        copy->pipeline=(void *)CFRetain(old->pipeline);
        copy->function=(void *)CFRetain(old->function);
        // Model data ownership deliberately starts empty.
        return copy;
    }
}
void b2_metal_device_name(void *opaque, char *name, size_t capacity) {
    B2Metal *handle = opaque;
    id<MTLDevice> device = (__bridge id<MTLDevice>)handle->device;
    fail(name, capacity, device.name);
}
// timings: allocation/input copy, command wall, GPU interval, output copy.
static int run_buffers(void *opaque, void **data, const uint64_t *sizes, uint32_t buffers,
                 uint32_t neurons, uint32_t readback_mask, double *timings,
                 char *error, size_t capacity, uint32_t single_group) {
    @autoreleasepool {
        if (!opaque || !neurons || buffers > 31) { fail(error, capacity, @"Invalid Metal dispatch"); return 1; }
        B2Metal *handle = opaque;
        id<MTLDevice> device = (__bridge id<MTLDevice>)handle->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)handle->queue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)handle->pipeline;
        double started = now();
        NSMutableArray<id<MTLBuffer>> *storage = [NSMutableArray arrayWithCapacity:buffers];
        for (uint32_t i = 0; i < buffers; ++i) {
            if (!sizes[i] || sizes[i] > device.maxBufferLength) { fail(error, capacity, @"Planned buffer exceeds Metal capacity"); return 1; }
            id<MTLBuffer> buffer = [device newBufferWithBytes:data[i] length:sizes[i] options:MTLResourceStorageModeShared];
            if (!buffer) { fail(error, capacity, @"Metal buffer allocation failed"); return 1; }
            [storage addObject:buffer];
        }
        timings[0] = now() - started;
        started = now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) { fail(error, capacity, @"Metal command creation failed"); return 1; }
        [encoder setComputePipelineState:pipeline];
        for (uint32_t i = 0; i < buffers; ++i) [encoder setBuffer:storage[i] offset:0 atIndex:i];
        NSUInteger lanes = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        if (single_group) {
            if (neurons>pipeline.maxTotalThreadsPerThreadgroup) {
                [encoder endEncoding]; fail(error,capacity,@"Workgroup exceeds compiled pipeline capacity"); return 1;
            }
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(neurons,1,1)];
        } else {
            [encoder dispatchThreads:MTLSizeMake(neurons, 1, 1) threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
        }
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        timings[1] = now() - started;
        if (command.status != MTLCommandBufferStatusCompleted) { fail(error, capacity, command.error.localizedDescription); return 1; }
        timings[2] = command.GPUEndTime - command.GPUStartTime;
        started = now();
        for (uint32_t i = 0; i < buffers; ++i)
            if (readback_mask & (1u << i)) memcpy(data[i], storage[i].contents, sizes[i]);
        timings[3] = now() - started;
        return 0;
    }
}

int b2_metal_run(void *opaque, void **data, const uint64_t *sizes, uint32_t buffers,
                 uint32_t neurons, uint32_t mask, double *timings, char *error, size_t capacity) {
    return run_buffers(opaque,data,sizes,buffers,neurons,mask,timings,error,capacity,0);
}
int b2_metal_run_workgroup(void *opaque, void **data, const uint64_t *sizes, uint32_t buffers,
                 uint32_t threads, uint32_t mask, double *timings, char *error, size_t capacity) {
    return run_buffers(opaque,data,sizes,buffers,threads,mask,timings,error,capacity,1);
}

// Shared storage for a canonical dispatch DAG; resident mode retains buffers
// and transfers only writable state on replay. Buffer barriers order every stage, including
// the last stage of one tick before the first stage of the next.
#include "clocks.h"
int b2_metal_run_dag(void **opaque, uint32_t stages, void **data,
                     const uint64_t *sizes, uint32_t buffers,
                     const uint32_t *bindings, const uint32_t *counts,
                     const uint32_t *lanes, const uint32_t *stage_clocks,
                     const int64_t *starts, const int64_t *ends, const double *dt, uint32_t clocks,
                     const uint8_t *writable, const uint8_t *upload, const uint8_t *readback, const uint32_t *spike_counts, const uint8_t *omit_upload, uint32_t retain, uint32_t explicit_barriers, uint64_t max_bytes, uint64_t *stats,
                     double *timings, char *error, size_t capacity) {
    @autoreleasepool {
        if (!stages || !opaque[0] || !clocks) { fail(error, capacity, @"Empty Metal DAG"); return 1; }
        if (explicit_barriers>1) { fail(error, capacity, @"Invalid Metal DAG synchronization policy"); return 1; }
        B2Metal *first = opaque[0];
        if (retain>2 || (retain==2 && !explicit_barriers)) { fail(error,capacity,@"Indirect DAG requires explicit stage barriers"); return 1; }
        if (retain!=2 && first->dag_indirect) { CFRelease(first->dag_indirect); first->dag_indirect=NULL; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)first->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)first->queue;
        for (uint32_t s=0; s<stages; ++s) {
            B2Metal *h = opaque[s];
            if (!h || ((__bridge id<MTLDevice>)h->device).registryID != device.registryID || counts[s] > 30 || !lanes[s] || stage_clocks[s]>=clocks) {
                fail(error, capacity, @"Invalid Metal DAG stage"); return 1;
            }
            for (uint32_t b=0; b<counts[s]; ++b) if (bindings[s*31+b] >= buffers) {
                fail(error, capacity, @"Invalid Metal DAG binding"); return 1;
            }
        }
        double started = now();
        // stats: allocation bytes, upload bytes, readback bytes, retained bytes,
        // cache hit, dispatches, barriers, reused buffer count/bytes.
        memset(stats,0,12*sizeof(uint64_t));
        NSMutableArray<id<MTLBuffer>> *storage = nil;
        if (!retain) b2_metal_clear_dag(first);
        if (retain && first->dag_storage) {
            storage = (__bridge NSMutableArray *)first->dag_storage;
            if (storage.count != buffers) { storage=nil; b2_metal_clear_dag(first); }
        }
        if (!storage) {
            storage = [NSMutableArray arrayWithCapacity:buffers];
            for (uint32_t b=0;b<buffers;++b) [(NSMutableArray *)storage addObject:[NSNull null]];
        }
        for (uint32_t b=0; b<buffers; ++b) {
            if (!sizes[b] || sizes[b] > device.maxBufferLength) { fail(error, capacity, @"Planned DAG buffer exceeds Metal capacity"); return 1; }
            id entry=storage[b];
            if (entry==[NSNull null] || ((id<MTLBuffer>)entry).length!=sizes[b]) {
                [(NSMutableArray *)storage replaceObjectAtIndex:b withObject:[NSNull null]];
                MTLResourceOptions options=MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
                id<MTLBuffer> buffer = omit_upload[b]
                    ? [device newBufferWithLength:sizes[b] options:options]
                    : [device newBufferWithBytes:data[b] length:sizes[b] options:options];
                if (!buffer) { fail(error, capacity, @"Metal DAG allocation failed"); return 1; }
                storage[b]=buffer;
                stats[0] += sizes[b]; if (!omit_upload[b]) stats[1] += sizes[b];
            } else {
                stats[4]=1; ++stats[7]; stats[8]+=sizes[b];
                if (!omit_upload[b] && (writable[b] || upload[b])) { memcpy(storage[b].contents,data[b],sizes[b]); stats[1]+=sizes[b]; }
            }
        }
        if (retain && !first->dag_storage) {
            first->dag_storage = (__bridge_retained void *)storage;
        }
        // Metal 3 automatically synchronizes tracked, directly bound resources.
        // Fail closed if future allocation code changes this prerequisite.
        for (uint32_t b=0; b<buffers; ++b) {
            if (storage[b].hazardTrackingMode != MTLHazardTrackingModeTracked) {
                fail(error, capacity, @"Metal DAG requires tracked buffers"); return 1;
            }
            if (retain) stats[3] += sizes[b];
        }
        timings[0] = now()-started;
        timings[2] = 0;
        NSMutableData *tick_storage = [NSMutableData dataWithBytes:starts length:clocks*sizeof(int64_t)];
        NSMutableData *active_storage = [NSMutableData dataWithLength:clocks];
        int64_t *ticks = tick_storage.mutableBytes;
        uint8_t *active = active_storage.mutableBytes;
        int more = b2_active_clocks(ticks,ends,dt,clocks,active);
        started = now();
        if (retain==2) {
            if (@available(macOS 11.0, *)) {
                NSDictionary *cached=(__bridge NSDictionary *)first->dag_indirect;
                stats[10]=cached!=nil;
                if (!cached) {
                    if (stages>64) { fail(error,capacity,@"Indirect DAG exceeds 64 stages"); return 1; }
                    uint64_t maximum=0;
                    for (uint32_t s=0;s<stages;++s) {
                        int64_t n=ends[stage_clocks[s]]-starts[stage_clocks[s]];
                        if (n<0 || n>1000000 || maximum>1000000-(uint64_t)n) { fail(error,capacity,@"Indirect DAG exceeds 1000000 dispatches"); return 1; }
                        maximum+=(uint64_t)n;
                    }
                    MTLIndirectCommandBufferDescriptor *descriptor=[MTLIndirectCommandBufferDescriptor new];
                    descriptor.commandTypes=MTLIndirectCommandTypeConcurrentDispatch;
                    descriptor.inheritPipelineState=NO; descriptor.inheritBuffers=NO;
                    descriptor.maxKernelBufferBindCount=31;
                    id<MTLIndirectCommandBuffer> indirect=[device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:MAX(maximum,1) options:MTLResourceStorageModePrivate];
                    id<MTLBuffer> arguments=[device newBufferWithLength:MAX(maximum,1)*sizeof(int64_t) options:MTLResourceStorageModeShared];
                    if (!indirect || !arguments) { fail(error,capacity,@"Metal indirect command allocation unavailable"); return 1; }
                    if (indirect.allocatedSize>max_bytes || arguments.allocatedSize>max_bytes-indirect.allocatedSize || stats[3]>max_bytes-indirect.allocatedSize-arguments.allocatedSize) {
                        fail(error,capacity,@"Indirect DAG exceeds total memory budget"); return 1;
                    }
                    NSMutableArray *pipelines=[NSMutableArray arrayWithCapacity:stages];
                    for (uint32_t s=0;s<stages;++s) {
                        B2Metal *h=opaque[s]; NSError *problem=nil;
                        MTLComputePipelineDescriptor *pd=[MTLComputePipelineDescriptor new];
                        pd.computeFunction=(__bridge id<MTLFunction>)h->function; pd.supportIndirectCommandBuffers=YES;
                        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:NULL error:&problem];
                        if (!pipeline || !pipeline.supportIndirectCommandBuffers) { fail(error,capacity,problem.localizedDescription ?: @"Metal indirect compute unsupported"); return 1; }
                        [pipelines addObject:pipeline];
                    }
                    NSMutableArray *chunks=[NSMutableArray array]; NSUInteger commandIndex=0;
                    while (more) {
                        NSUInteger begin=commandIndex;
                        for (uint32_t t=0;t<64 && more;++t) {
                            for (uint32_t s=0;s<stages;++s) {
                                if (!active[stage_clocks[s]]) continue;
                                if (commandIndex>=maximum) { fail(error,capacity,@"Indirect schedule exceeds planned count"); return 1; }
                                id<MTLComputePipelineState> pipeline=pipelines[s];
                                id<MTLIndirectComputeCommand> entry=[indirect indirectComputeCommandAtIndex:commandIndex];
                                [entry setBarrier];
                                [entry setComputePipelineState:pipeline];
                                for (uint32_t b=0;b<counts[s];++b) [entry setKernelBuffer:storage[bindings[s*31+b]] offset:0 atIndex:b];
                                ((int64_t *)arguments.contents)[commandIndex]=ticks[stage_clocks[s]];
                                [entry setKernelBuffer:arguments offset:commandIndex*sizeof(int64_t) atIndex:counts[s]];
                                [entry concurrentDispatchThreads:MTLSizeMake(lanes[s],1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
                                ++commandIndex;
                            }
                            for (uint32_t c=0;c<clocks;++c) ticks[c]+=active[c];
                            more=b2_active_clocks(ticks,ends,dt,clocks,active);
                        }
                        [chunks addObject:[NSValue valueWithRange:NSMakeRange(begin,commandIndex-begin)]];
                    }
                    if (commandIndex!=maximum) { fail(error,capacity,@"Indirect schedule count mismatch"); return 1; }
                    cached=@{@"commands":indirect,@"ticks":arguments,@"pipelines":pipelines,@"chunks":chunks,@"count":@(commandIndex)};
                    first->dag_indirect=(__bridge_retained void *)cached;
                    stats[11]=commandIndex;
                }
                id<MTLIndirectCommandBuffer> indirect=cached[@"commands"];
                id<MTLBuffer> arguments=cached[@"ticks"];
                stats[9]=indirect.allocatedSize+arguments.allocatedSize;
                if (stats[9]>max_bytes || stats[3]>max_bytes-stats[9]) { fail(error,capacity,@"Indirect DAG exceeds total memory budget"); return 1; }
                for (NSValue *chunk in cached[@"chunks"]) {
                    id<MTLCommandBuffer> command=[queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    if (!command || !encoder) { fail(error,capacity,@"Cannot create indirect DAG encoder"); return 1; }
                    for (uint32_t b=0;b<buffers;++b) [encoder useResource:storage[b] usage:MTLResourceUsageRead | (writable[b] ? MTLResourceUsageWrite : 0)];
                    [encoder useResource:arguments usage:MTLResourceUsageRead];
                    [encoder executeCommandsInBuffer:indirect withRange:chunk.rangeValue];
                    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
                    if (command.status!=MTLCommandBufferStatusCompleted) { fail(error,capacity,command.error.localizedDescription); return 1; }
                    timings[2]+=command.GPUEndTime-command.GPUStartTime;
                }
                stats[5]=[cached[@"count"] unsignedLongLongValue]; stats[6]=stats[5];
                more=0;
            } else { fail(error,capacity,@"Indirect compute requires macOS 11 or later"); return 1; }
        }
        // Bound command memory and watchdog exposure, retaining GPU storage.
        while (more) {
            @autoreleasepool {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                if (!command) { fail(error, capacity, @"Cannot create DAG command buffer"); return 1; }
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                if (!encoder) { fail(error, capacity, @"Cannot create DAG encoder"); return 1; }
                if (encoder.dispatchType != MTLDispatchTypeSerial) {
                    [encoder endEncoding]; fail(error, capacity, @"Metal DAG requires serial dispatch"); return 1;
                }
                for (uint32_t t=0; t<64 && more; ++t) {
                    for (uint32_t s=0; s<stages; ++s) {
                        if (!active[stage_clocks[s]]) continue;
                        int64_t tick = ticks[stage_clocks[s]];
                        B2Metal *h = opaque[s];
                        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)h->pipeline;
                        [encoder setComputePipelineState:pipeline];
                        for (uint32_t b=0; b<counts[s]; ++b) [encoder setBuffer:storage[bindings[s*31+b]] offset:0 atIndex:b];
                        [encoder setBytes:&tick length:sizeof(tick) atIndex:counts[s]];
                        [encoder dispatchThreads:MTLSizeMake(lanes[s],1,1)
                           threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
                        ++stats[5];
                        if (explicit_barriers) {
                            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]; ++stats[6];
                        }
                    }
                    for (uint32_t c=0;c<clocks;++c) ticks[c]+=active[c];
                    more = b2_active_clocks(ticks,ends,dt,clocks,active);
                }
                [encoder endEncoding];
                [command commit]; [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted) { fail(error, capacity, command.error.localizedDescription); return 1; }
                timings[2] += command.GPUEndTime-command.GPUStartTime;
            }
        }
        timings[1] = now()-started;
        started = now();
        for (uint32_t b=0; b<buffers; ++b) if (readback[b]) {
            if (spike_counts[b]) {
                uint32_t count_binding=spike_counts[b]-1;
                if (count_binding>=buffers || !readback[count_binding] || !sizes[count_binding] || sizes[count_binding]%4) {
                    fail(error,capacity,@"Invalid spike readback count binding"); return 1;
                }
                uint64_t rows=sizes[count_binding]/4;
                if (!sizes[b] || sizes[b]%8 || (sizes[b]/8)%rows) {
                    fail(error,capacity,@"Invalid spike readback capacity"); return 1;
                }
                uint64_t row_capacity=sizes[b]/8/rows;
                const uint32_t *recorded=storage[count_binding].contents;
                for (uint64_t row=0;row<rows;++row) if (recorded[row]>row_capacity) {
                    fail(error,capacity,@"GPU spike capacity invariant violated during readback"); return 1;
                }
                for (uint64_t row=0;row<rows;++row) {
                    uint64_t bytes=(uint64_t)recorded[row]*8, offset=row*row_capacity*8;
                    if (bytes) memcpy((uint8_t*)data[b]+offset,(const uint8_t*)storage[b].contents+offset,bytes);
                    stats[2]+=bytes;
                }
            } else {
                memcpy(data[b],storage[b].contents,sizes[b]); stats[2] += sizes[b];
            }
        }
        timings[3] = now()-started;
        return 0;
    }
}
