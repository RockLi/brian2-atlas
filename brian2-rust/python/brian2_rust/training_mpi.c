/* Small native MPI ABI. Ordered rank summation makes a fixed rank layout
 * reproducible. Model arrays are replicated; numerical work is target-owned. */
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int b2_train_mpi_init(int* rank,int* size) {
 int rc=MPI_Init(NULL,NULL);if(rc!=MPI_SUCCESS)return rc;
 MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
 if((rc=MPI_Comm_rank(MPI_COMM_WORLD,rank))!=MPI_SUCCESS)return rc;
 return MPI_Comm_size(MPI_COMM_WORLD,size);
}
int b2_train_mpi_sum(double* values,uint64_t length) {
 if(length>INT_MAX)return -1;
 int rank,size,rc;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 double* receive=rank==0?malloc(length*(uint64_t)size*sizeof(double)):NULL;
 if(rank==0&&length&&!receive){MPI_Abort(MPI_COMM_WORLD,3);return -2;}
 rc=MPI_Gather(values,(int)length,MPI_DOUBLE,receive,(int)length,MPI_DOUBLE,0,MPI_COMM_WORLD);
 if(rc!=MPI_SUCCESS){free(receive);return rc;}
 if(rank==0)for(uint64_t i=0;i<length;i++){
  values[i]=receive[i];
  for(int r=1;r<size;r++)values[i]+=receive[(uint64_t)r*length+i];
 }
 free(receive);
 return MPI_Bcast(values,(int)length,MPI_DOUBLE,0,MPI_COMM_WORLD);
}
int b2_train_mpi_agree(const unsigned char* digest) {
 unsigned char root[32];memcpy(root,digest,32);
 int rc=MPI_Bcast(root,32,MPI_BYTE,0,MPI_COMM_WORLD);if(rc!=MPI_SUCCESS)return rc;
 int mismatch=memcmp(root,digest,32)!=0,total=0;
 rc=MPI_Allreduce(&mismatch,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 return rc==MPI_SUCCESS?(total?-1:0):rc;
}
int b2_train_mpi_finish(void){int rc=MPI_Barrier(MPI_COMM_WORLD);return rc==MPI_SUCCESS?MPI_Finalize():rc;}
void b2_train_mpi_abort(void){MPI_Abort(MPI_COMM_WORLD,2);}
