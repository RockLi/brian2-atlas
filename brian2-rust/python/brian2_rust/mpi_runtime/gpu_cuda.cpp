// Host ABI for generated CUDA updates. All copies complete before host events.
#include <stdio.h>
#include <stdint.h>
#include <new>
struct MpiCuda {
    uint32_t kernel; int device;
    float *values = nullptr; uint64_t *meta = nullptr; uint32_t *faults = nullptr;
    uint64_t value_capacity = 0, fault_capacity = 0;
};
static int gpu_failure(cudaError_t code, char *error, size_t capacity) {
    if (code == cudaSuccess) return 0;
    snprintf(error, capacity, "%s", cudaGetErrorString(code)); return 1;
}
extern "C" void b2gpu_destroy(void *opaque) {
    auto *h = static_cast<MpiCuda *>(opaque);
    if (!h) return;
    cudaSetDevice(h->device); cudaFree(h->values); cudaFree(h->meta); cudaFree(h->faults); delete h;
}
extern "C" void *b2gpu_create(uint32_t kernel, int device, char *error, size_t capacity) {
    if (kernel >= KERNEL_COUNT) { snprintf(error, capacity, "invalid MPI GPU kernel"); return nullptr; }
    if (gpu_failure(cudaSetDevice(device), error, capacity)) return nullptr;
    auto *h = new (std::nothrow) MpiCuda;
    if (!h) { snprintf(error, capacity, "MPI GPU host allocation failed"); return nullptr; }
    h->kernel = kernel; h->device = device;
    if (gpu_failure(cudaMalloc(&h->meta, 5*sizeof(uint64_t)), error, capacity)) { b2gpu_destroy(h); return nullptr; }
    return h;
}
extern "C" int b2gpu_run(void *opaque, float *values, uint64_t length, uint64_t *meta,
                          uint32_t *faults, char *error, size_t capacity) {
    auto *h = static_cast<MpiCuda *>(opaque);
#define GPU_CHECK(call) do { if (gpu_failure((call), error, capacity)) return 1; } while (0)
    GPU_CHECK(cudaSetDevice(h->device));
    if (length > h->value_capacity) {
        GPU_CHECK(cudaFree(h->values)); h->values = nullptr; h->value_capacity = 0;
        GPU_CHECK(cudaMalloc(&h->values, length*sizeof(float))); h->value_capacity = length;
    }
    if (meta[0] > h->fault_capacity) {
        GPU_CHECK(cudaFree(h->faults)); h->faults = nullptr; h->fault_capacity = 0;
        GPU_CHECK(cudaMalloc(&h->faults, meta[0]*sizeof(uint32_t))); h->fault_capacity = meta[0];
    }
    GPU_CHECK(cudaMemcpy(h->values, values, length*sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(h->meta, meta, 5*sizeof(uint64_t), cudaMemcpyHostToDevice));
    uint32_t blocks = (uint32_t)((meta[0]+255)/256);
    switch (h->kernel) { KERNEL_CASES default: return 1; }
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaMemcpy(values, h->values, length*sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(faults, h->faults, meta[0]*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return 0;
#undef GPU_CHECK
}
