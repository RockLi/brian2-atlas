/* Bounded cluster connectivity probe: variable-size collectives and P2P ring. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size, ok = 1, all_ok, sum;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 1 || size > 64) { MPI_Abort(MPI_COMM_WORLD, 2); return 2; }
    int counts[64], offsets[64], send[64], received[2048], total = 0;
    for (int i = 0; i < size; ++i) { counts[i] = i; offsets[i] = total; total += i; }
    for (int i = 0; i < rank; ++i) send[i] = rank * 100 + i;
    double start = MPI_Wtime();
    for (int iteration = 0; iteration < 32; ++iteration) {
        MPI_Allgatherv(send, rank, MPI_INT, received, counts, offsets, MPI_INT, MPI_COMM_WORLD);
        for (int r = 0; r < size; ++r)
            for (int i = 0; i < r; ++i) ok &= received[offsets[r] + i] == r * 100 + i;
        int token = -1;
        MPI_Request requests[2];
        MPI_Irecv(&token, 1, MPI_INT, (rank + size - 1) % size, 7, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&rank, 1, MPI_INT, (rank + 1) % size, 7, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        ok &= token == (rank + size - 1) % size;
    }
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ok &= sum == size * (size - 1) / 2;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    char name[MPI_MAX_PROCESSOR_NAME] = {0};
    char names[64][MPI_MAX_PROCESSOR_NAME] = {{0}};
    int length;
    MPI_Get_processor_name(name, &length);
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("{\"schema\":\"b2-mpi-cluster-probe-v1\",\"ranks\":%d,\"sum\":%d,\"ok\":%s,\"seconds\":%.9f,\"hosts\":[", size, sum, all_ok ? "true" : "false", MPI_Wtime()-start);
        for (int r = 0; r < size; ++r) printf("%s\"%s\"", r ? "," : "", names[r]);
        puts("]}");
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
