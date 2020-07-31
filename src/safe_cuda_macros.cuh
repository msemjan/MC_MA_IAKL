//#include "my_helper.h"

#ifndef CUDA_SAFE_CUDA_MACROS_CUH_
#define CUDA_SAFE_CUDA_MACROS_CUH_

#include <curand.h>
#include <cstdio>

// CUDA error checking macro
#define CUDAErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert( cudaError_t code
                     , const char *file
                     , int line
                     , bool abort = true )
{
    if (code != cudaSuccess){
        fprintf( stderr
               , "GPUassert: %s ; %s ; line %d\n"
               , cudaGetErrorString(code)
               , file
               , line);
        if (abort) exit(code);
    }
}

// cuRAND error checking macro
#define cuRAND_ErrChk(err) { if (err != CURAND_STATUS_SUCCESS) std::cout << curandGetErrorString(err) << "\n"; }

// cuRAND errors
char* curandGetErrorString( curandStatus_t );
const char* curanderr[14] =
{
    "No errors", 
    "Header file and linked library version do not match",
    "Generator not initialized", 
    "Memory allocation failed",
    "Generator is wrong type", 
    "Argument out of range",
    "Length requested is not a multiple of dimension",
    "GPU does not have double precision required by MRG32k3a",
    "Kernel launch failure", 
    "Pre-existing failure on library entry",
    "Initialization of CUDA failed", 
    "Architecture mismatch, GPU does not support requested feature",
    "Internal library error", 
    "Unknown error"
};

#endif // CUDA_SAFE_CUDA_MACROS_CUH_

//
//#ifndef MYTHRUSTYFUN
//#define MYTHRUSTYFUN
//// typedef float myType;
//
//template<typename myType>
//__inline__ __device__
//double warpReduceSum(myType val) {
//	for (int offset = warpSize / 2; offset > 0; offset /= 2)
//		val += __shfl_down(val, offset);
//	return val;
//}
//
//
//template<typename myType>
//__inline__ __device__
//double blockReduceSum(myType val) {
//
//	static __shared__ int shared[32]; // Shared mem for 32 partial sums
//	int lane = threadIdx.x % warpSize;
//	int wid = threadIdx.x / warpSize;
//
//	val = warpReduceSum(val);     // Each warp performs partial reduction
//
//	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory
//
//	__syncthreads();              // Wait for all partial reductions
//
//	//read from shared memory only if that warp existed
//	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
//
//	if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp
//
//	return val;
//}
//
//template<typename myType>
//__global__ void deviceReduceKernel(myType *in, myType* out, int N) {
//	myType sum = 0;
//	//reduce multiple elements per thread
//	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//		i < N;
//		i += blockDim.x * gridDim.x) {
//		sum += in[i];
//	}
//	sum = blockReduceSum(sum);
//	if (threadIdx.x == 0)
//		out[blockIdx.x] = sum;
//}
//
//template<typename myType>
//__global__ void deviceReduceWarpAtomicKernel(myType *in, myType* out, int N) {
//	myType sum = 0;
//	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//		i < N;
//		i += blockDim.x * gridDim.x) {
//		sum += in[i];
//	}
//	sum = warpReduceSum(sum);
//	if ((threadIdx.x & (warpSize - 1)) == 0)
//		atomicAdd(out, sum);
//}
//
//template<typename myType>
//void deviceReduce(myType *in, myType* out, int N) {
//	int threads = 512;
//	int blocks = min((N + threads - 1) / threads, 1024);
//
//	deviceReduceKernel << <blocks, threads >> >(in, out, N);
//	deviceReduceKernel << <1, 1024 >> >(out, out, blocks);
//}
//
//template< typename T >
//T* myCudaMalloc( int size )
//{
//    T* loc = nullptr;
//    const int space = size * sizeof( T );
//        
//    CUDAErrChk(cudaMalloc((void**)& loc, space));
//    return loc;
//}
//
//#endif
