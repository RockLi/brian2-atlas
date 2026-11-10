/* MPI handles/types remain inside the implementation's own C headers.
 * Rust sees only fixed-width scalars and buffers, never vendor MPI constants. */
#ifdef __APPLE__
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static int local_rank = 0;
int b2mpi_local_rank(void) { return local_rank; }

int b2mpi_init(int *rank, int *size) {
    int rc = MPI_Init(NULL, NULL);
    if (rc != MPI_SUCCESS) return rc;
    rc = MPI_Comm_rank(MPI_COMM_WORLD, rank);
    if (rc != MPI_SUCCESS) return rc;
    rc = MPI_Comm_size(MPI_COMM_WORLD, size);
    if (rc != MPI_SUCCESS) return rc;
    /* ch3:sock may return singleton MPI_COMM_TYPE_SHARED groups even for
     * ranks on one physical host. Match MPI processor names explicitly. */
    char name[MPI_MAX_PROCESSOR_NAME] = {0};
    int length = 0;
    rc = MPI_Get_processor_name(name, &length);
    if (rc != MPI_SUCCESS) return rc;
    char *names = calloc((size_t)*size, MPI_MAX_PROCESSOR_NAME);
    if (!names) return -1;
    rc = MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names,
                       MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    if (rc == MPI_SUCCESS) {
        for (int i = 0; i < *rank; ++i)
            if (strncmp(name, names + (size_t)i*MPI_MAX_PROCESSOR_NAME,
                        MPI_MAX_PROCESSOR_NAME) == 0) ++local_rank;
    }
    free(names);
    return rc;
}
int b2mpi_finalize(void) { return MPI_Finalize(); }
void b2mpi_abort(void) {
    /* Hydra may terminate its I/O proxy before forwarding an error emitted
     * immediately before MPI_Abort. Give the proxy a bounded drain interval.
     * Never synchronize here: only one rank may have encountered the error. */
    fflush(stderr);
    const struct timespec diagnostic_grace = {0, 100000000};
    (void)nanosleep(&diagnostic_grace, NULL);
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, 1);
    _Exit(1);
}
int b2mpi_barrier(void) { return MPI_Barrier(MPI_COMM_WORLD); }

uint64_t b2mpi_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#ifdef __APPLE__
    return (uint64_t)usage.ru_maxrss;
#else
    return (uint64_t)usage.ru_maxrss * 1024u;
#endif
}

int b2mpi_spikes(const uint64_t *send, int count, uint64_t *recv,
                 int capacity, int *received) {
    int size, rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rc != MPI_SUCCESS) return rc;
    int *counts = calloc((size_t)size, sizeof(int));
    int *offsets = calloc((size_t)size, sizeof(int));
    if (!counts || !offsets) { free(counts); free(offsets); return -1; }
    rc = MPI_Allgather(&count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int total = 0;
    if (rc == MPI_SUCCESS) {
        for (int i = 0; i < size; ++i) {
            if (counts[i] < 0 || counts[i] > capacity - total) { rc = -1; break; }
            offsets[i] = total;
            total += counts[i];
        }
    }
    if (rc == MPI_SUCCESS)
        rc = MPI_Allgatherv(send, count, MPI_UINT64_T, recv, counts, offsets,
                            MPI_UINT64_T, MPI_COMM_WORLD);
    *received = total;
    free(counts);
    free(offsets);
    return rc;
}

/* Exactly one owner contributes each cell. OR transports bits without changing
 * signed zero, subnormals or rounding through floating point arithmetic. */
int b2mpi_merge(uint64_t *bits, int count) {
    return MPI_Allreduce(MPI_IN_PLACE, bits, count, MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
}
int b2mpi_gather(const uint64_t *send, int count, uint64_t *recv) {
    return MPI_Allgather(send, count, MPI_UINT64_T, recv, count, MPI_UINT64_T, MPI_COMM_WORLD);
}
int b2mpi_host(uint64_t *bytes, int capacity) {
    char name[MPI_MAX_PROCESSOR_NAME];
    int length = 0, rc = MPI_Get_processor_name(name, &length);
    if (rc != MPI_SUCCESS) return rc;
    if (length >= capacity) return -1;
    for (int i = 0; i < capacity; ++i) bytes[i] = i < length ? (unsigned char)name[i] : 0;
    return MPI_SUCCESS;
}

/* Rank-local contiguous vectors are collected only on the output rank.
 * This tag belongs to the generated executable. Each collection completes
 * globally before reuse. Synchronous sends keep sparse owners' buffers alive
 * until the root has received them, without replicating full arrays. */
int b2mpi_collect(const uint64_t *send, int count, uint64_t *recv, int total, int owner) {
    int size, rank, rc;
    const int tag = 17013;
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rc != MPI_SUCCESS) return rc;
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rc != MPI_SUCCESS) return rc;
    if (total < 0 || count < 0 || owner < -1 || owner >= size) return -1;
    int begin = owner < 0 ? (int)((int64_t)total*rank/size) : 0;
    int expected = owner < 0 ? (int)((int64_t)total*(rank+1)/size)-begin : (rank == owner ? total : 0);
    if (count != expected) return -1;
    if (rank == 0) {
        if (count) memmove(recv + begin, send, (size_t)count*sizeof(uint64_t));
        for (int source = 1; source < size; ++source) {
            int offset = owner < 0 ? (int)((int64_t)total*source/size) : 0;
            int length = owner < 0 ? (int)((int64_t)total*(source+1)/size)-offset : (source == owner ? total : 0);
            if (length) {
                rc = MPI_Recv(recv + offset, length, MPI_UINT64_T, source, tag,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rc != MPI_SUCCESS) return rc;
            }
        }
    } else if (count) {
        rc = MPI_Ssend(send, count, MPI_UINT64_T, 0, tag, MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS) return rc;
    }
    return MPI_Barrier(MPI_COMM_WORLD);
}

/* Reconstruct target-owned synapse state in original global edge order.
 * Each non-root rank streams bounded identity/value chunks, avoiding a second
 * global edge vector or a rank-sized receive allocation on the output rank. */
int b2mpi_collect_indexed(const uint64_t *ids, const uint64_t *send, int count,
                          uint64_t *recv, int total) {
    int size, rank, rc;
    const int id_tag = 17015, value_tag = 17016;
    const int chunk = 65536;
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rc != MPI_SUCCESS) return rc;
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rc != MPI_SUCCESS) return rc;
    int locally_valid = count >= 0 && total >= 0;
    if (locally_valid)
        for (int i = 0; i < count; ++i)
            if (ids[i] >= (uint64_t)total) { locally_valid = 0; break; }
    int globally_valid = 0;
    rc = MPI_Allreduce(&locally_valid, &globally_valid, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) return rc;
    if (!globally_valid) return -1;
    int *counts = rank == 0 ? calloc((size_t)size, sizeof(int)) : NULL;
    int counts_allocated = rank != 0 || counts != NULL;
    rc = MPI_Allreduce(MPI_IN_PLACE, &counts_allocated, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS || !counts_allocated) {
        free(counts); return rc == MPI_SUCCESS ? -1 : rc;
    }
    rc = MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) { free(counts); return rc; }
    int counts_valid = 1;
    if (rank == 0) {
        int64_t sum = 0;
        for (int source = 0; source < size; ++source) sum += counts[source];
        counts_valid = sum == total;
    }
    rc = MPI_Bcast(&counts_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS || !counts_valid) { free(counts); return rc == MPI_SUCCESS ? -1 : rc; }
    if (rank == 0) {
        for (int i = 0; i < count; ++i) {
            recv[ids[i]] = send[i];
        }
        uint64_t *id_buffer = malloc((size_t)chunk*sizeof(uint64_t));
        uint64_t *value_buffer = malloc((size_t)chunk*sizeof(uint64_t));
        int buffers_valid = id_buffer != NULL && value_buffer != NULL;
        rc = MPI_Bcast(&buffers_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS || !buffers_valid) {
            free(id_buffer); free(value_buffer); free(counts);
            return rc == MPI_SUCCESS ? -1 : rc;
        }
        for (int source = 1; source < size; ++source) {
            for (int offset = 0; offset < counts[source]; offset += chunk) {
                int length = counts[source]-offset < chunk ? counts[source]-offset : chunk;
                rc = MPI_Recv(id_buffer, length, MPI_UINT64_T, source, id_tag,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rc != MPI_SUCCESS) break;
                rc = MPI_Recv(value_buffer, length, MPI_UINT64_T, source, value_tag,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rc != MPI_SUCCESS) break;
                for (int i = 0; i < length; ++i) {
                    recv[id_buffer[i]] = value_buffer[i];
                }
                if (rc != MPI_SUCCESS) break;
            }
            if (rc != MPI_SUCCESS) break;
        }
        free(id_buffer); free(value_buffer); free(counts);
        if (rc != MPI_SUCCESS) return rc;
    } else {
        int buffers_valid = 0;
        rc = MPI_Bcast(&buffers_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS || !buffers_valid) return rc == MPI_SUCCESS ? -1 : rc;
        for (int offset = 0; offset < count; offset += chunk) {
            int length = count-offset < chunk ? count-offset : chunk;
            rc = MPI_Ssend(ids+offset, length, MPI_UINT64_T, 0, id_tag, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) return rc;
            rc = MPI_Ssend(send+offset, length, MPI_UINT64_T, 0, value_tag, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) return rc;
        }
    }
    return MPI_Barrier(MPI_COMM_WORLD);
}

/* Integer counts establish the original source-major edge identities without
 * replicating edge arrays. The exclusive prefix follows global draw order. */
int b2mpi_topology_counts(const uint64_t *local, uint64_t *total,
                         uint64_t *prefix, int count) {
    int rank, rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rc != MPI_SUCCESS) return rc;
    rc = MPI_Allreduce(local, total, count, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) return rc;
    rc = MPI_Exscan(local, prefix, count, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) memset(prefix, 0, (size_t)count * sizeof(uint64_t));
    return rc;
}

/* Bounded construction batches; all counts/displacements are uint64 cells. */
int b2mpi_route_edges(const uint64_t *send, const int *counts,
                     uint64_t *receive, int capacity, int *received) {
    int size, rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rc != MPI_SUCCESS) return rc;
    int *in = calloc((size_t)size, sizeof(int));
    int *out_offsets = calloc((size_t)size, sizeof(int));
    int *in_offsets = calloc((size_t)size, sizeof(int));
    if (!in || !out_offsets || !in_offsets) {
        free(in); free(out_offsets); free(in_offsets); return -1;
    }
    rc = MPI_Alltoall(counts, 1, MPI_INT, in, 1, MPI_INT, MPI_COMM_WORLD);
    int sent = 0, got = 0;
    if (rc == MPI_SUCCESS) {
        for (int i = 0; i < size; ++i) {
            if (counts[i] < 0 || counts[i] % 3 || counts[i] > INT_MAX-sent ||
                in[i] < 0 || in[i] % 3 || in[i] > capacity-got) { rc = -1; break; }
            out_offsets[i] = sent; sent += counts[i];
            in_offsets[i] = got; got += in[i];
        }
    }
    if (rc == MPI_SUCCESS)
        rc = MPI_Alltoallv(send, counts, out_offsets, MPI_UINT64_T,
                          receive, in, in_offsets, MPI_UINT64_T, MPI_COMM_WORLD);
    *received = got;
    free(in); free(out_offsets); free(in_offsets);
    return rc;
}
