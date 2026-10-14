#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern int b2mpi_collect(const uint64_t*,int,uint64_t*,int,int);
int main(int argc,char **argv) {
 MPI_Init(&argc,&argv);int rank,size,bad=0,total_bad=0;
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 if(size<1||size>64)MPI_Abort(MPI_COMM_WORLD,2);
 for(int round=0;round<32;round++) {
  uint64_t *results[64];
  for(int owner=0;owner<size;owner++) {
   uint64_t *send=rank==owner?malloc(16):(uint64_t*)(uintptr_t)8;
   results[owner]=rank==0?calloc(2,8):(uint64_t*)(uintptr_t)8;
   if(rank==owner){send[0]=(uint64_t)(round*10000+owner*100+1);send[1]=send[0]+1;}
   if(b2mpi_collect(send,rank==owner?2:0,results[owner],2,owner)!=MPI_SUCCESS)MPI_Abort(MPI_COMM_WORLD,3);
   if(rank==owner)free(send);
  }
  if(rank==0)for(int owner=0;owner<size;owner++) {
   uint64_t expected=(uint64_t)(round*10000+owner*100+1);
   if(results[owner][0]!=expected||results[owner][1]!=expected+1) {
    if(bad<5)printf("mismatch round=%d owner=%d actual=%llu,%llu expected=%llu\n",round,owner,(unsigned long long)results[owner][0],(unsigned long long)results[owner][1],(unsigned long long)expected);
    bad++;
   }
   free(results[owner]);
  }
  /* Uneven contiguous shards, including empty ranks and an empty output. */
  int totals[3]={0,2,size*2+1};
  for(int trial=0;trial<3;trial++) {
   int total=totals[trial],begin=total*rank/size,end=total*(rank+1)/size;
   uint64_t *send=end>begin?malloc((size_t)(end-begin)*8):(uint64_t*)(uintptr_t)8;
   uint64_t *recv=rank==0&&total?calloc((size_t)total,8):(uint64_t*)(uintptr_t)8;
   for(int i=begin;i<end;i++)send[i-begin]=(uint64_t)(round*1000000+i+1);
   if(b2mpi_collect(send,end-begin,recv,total,-1)!=MPI_SUCCESS)MPI_Abort(MPI_COMM_WORLD,4);
   if(end>begin)free(send);
   if(rank==0&&total) {
    for(int i=0;i<total;i++)if(recv[i]!=(uint64_t)(round*1000000+i+1))bad++;
    free(recv);
   }
  }
 }
 MPI_Allreduce(&bad,&total_bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 if(rank==0)printf("{\"schema\":\"mpi-collect-probe-v1\",\"ranks\":%d,\"bad\":%d}\n",size,total_bad);
 MPI_Finalize();return total_bad?1:0;
}
