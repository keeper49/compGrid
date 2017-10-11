#include "stdio.h"
#define CUDA_ERR_CHECK(x) \
	do{ cudaError_t err = x; \
		if (err != cudaSuccess) { \
			fprintf(stderr, "Error \"%s\" at %s:%d \n", \
				cudaGetErrorString(err), __FILE__, __LINE__);\
		exit(0);} \
	} while(0)

#define DGX 3
#define DGY 2 
#define DBX 2
#define DBY 2 
#define DBZ 2
 
#define N (DBX*DBY*DBZ*DGX*DGY)

__global__ void gpu_kernel() {
	/*
	int block_idx, grid_dim;
	block_idx = blockIdx.x;	//номер блока по оси x
	grid_dim = gridDim.x;	//общее количество блоков по оси х
 	*/
	printf("\nThread:\n   block_x # %d\t   block_y # %d\t   block_z # %d\n   thread_x # %d\t   thread_y # %d\t   thread_z # %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main (void){

	dim3 grid(DGX, DGY);
	dim3 block(DBX,DBY,DBZ);
	
	int dev;
	cudaDeviceProp prop;
	
	CUDA_ERR_CHECK( cudaGetDevice( &dev ) );
	CUDA_ERR_CHECK( cudaGetDeviceProperties(&prop, dev) );

	printf("name\t\t\t%s\n", prop.name);
	printf("totalGlobalMem\t\t%zd\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock\t%zd\n", prop.sharedMemPerBlock);
	printf("regsPerBlock\t\t%d\n", prop.regsPerBlock);
	printf("warpSize\t\t%d\n", prop.warpSize);
	printf("memPitch\t\t%zd\n", prop.memPitch);
	printf("maxThreadsPerBlock\t%d\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim.x\t\t%d\n", prop.maxThreadsDim[0]);
	printf("maxThreadsDim.y\t\t%d\n", prop.maxThreadsDim[1]);
	printf("maxThreadsDim.z\t\t%d\n", prop.maxThreadsDim[2]);
	printf("maxGridSize.x\t\t%d\n", prop.maxGridSize[0]);
	printf("maxGridSize.y\t\t%d\n", prop.maxGridSize[1]);
	printf("maxGridSize.z\t\t%d\n", prop.maxGridSize[2]);
	printf("totalConstMem\t\t%zd\n", prop.totalConstMem);
	printf("major\t\t\t%d\n", prop.major);
	printf("minor\t\t\t%d\n", prop.minor);
	printf("clockRate\t\t%d\n", prop.clockRate);
	printf("textureAlignment\t%zd\n", prop.textureAlignment);
	printf("deviceOverlap\t\t%d\n", prop.deviceOverlap);
	printf("multiProcessorCount\t%d\n", prop.multiProcessorCount);
	printf("kernelExecTimeoutEnabled %d\n", prop.kernelExecTimeoutEnabled);
	printf("integrated\t\t%d\n", prop.integrated);
	printf("canMapHostMemory\t%d\n", prop.canMapHostMemory);
	printf("computeMode\t\t%d\n", prop.computeMode);
	printf("concurrentKernels\t%d\n", prop.concurrentKernels);
	printf("ECCEnabled\t\t%d\n", prop.ECCEnabled);
	printf("pciBusID\t\t%d\n", prop.pciBusID);
	printf("pciDeviceID\t\t%d\n", prop.pciDeviceID);
	printf("tccDriver\t\t%d\n", prop.tccDriver);

	printf("cudaComputeMode:\n");
	printf("cudaComputeModeDefault: %d\n", cudaComputeModeDefault);
	printf("cudaComputeModeExclusive: %d\n", cudaComputeModeExclusive);
	printf("cudaComputeModeProhibited: %d\n", cudaComputeModeProhibited);	

		
	//gpu_kernel<<<grid, block>>>();

	CUDA_ERR_CHECK( cudaGetLastError() );
	CUDA_ERR_CHECK( cudaDeviceSynchronize() );
	return 0;
}
