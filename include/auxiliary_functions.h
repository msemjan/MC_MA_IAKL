
// C/C++ Includes
#include <vector>                   // containers for data
#include <fstream>                  // File IO
#include <iostream>                 // STD IO
#include <sstream>                  // Stringstreams
#include <ctime>                    // for time and random seed
#include <string>                   // C++ strings
#include <numeric>                  // std::accumulate
#include <algorithm>                // std::count_if

// CUDA Includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Helper functions
#include "safe_cuda_macros.cuh"      // Makros for checking CUDA errors
#include "Lattice.h"                 // Lattice class

// Include guard
#ifndef CUDA_MC_AUXILIARY_FUNCTIONS_H_
#define CUDA_MC_AUXILIARY_FUNCTIONS_H_

#define MAX_THREADS 65536 

/// Converts a std::vector to a string
template<typename T>
std::string vector_to_string(std::vector<T>& v){
    std::stringstream ss;
    bool first = true;
    for (auto it = v.begin(); it != v.end(); it++){
        if (!first)
            ss << ", ";
        ss << *it;
        first = false;
    }
    return ss.str();
}

/// Setup for RNG
template<typename State>
__global__ void setupKernel( State *state
                           , long int globalSeed )
{
    // Thread identification
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Each thread gets same seed, a different sequence number, no offset
	curand_init(globalSeed + id, id, 0, &state[id]);
}

/// Generates uniformly distributed random numbers of type T
template<typename State, typename T>
__global__ void generateUniformKernel( State *state
                                     , T *result
                                     , unsigned int n )
{
    // Thread identification
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Local copy of the state
    State localState = state[id];

    // Generating random numbers
	float4 v4 = curand_uniform4(&localState);

    // Copy the random numbers to output memory
    result[id            ] = (T)v4.w;
	result[id + 1 * n / 4] = (T)v4.x;
	result[id + 2 * n / 4] = (T)v4.y;
	result[id + 3 * n / 4] = (T)v4.z;

    // Copy state to global memory
    state[id] = localState;
}

/// Generates uniform numbers of type T and applies predicat p on them
template<typename State, typename T, typename Predicate>
__global__ void generateUniformKernel( State *state 
                                     , T *result 
                                     , Predicate p 
                                     , unsigned int n)
{
    // Thread identification
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Local copy of the state
	State localState = state[id];

    // Generate random numbers
	float4 v4 = curand_uniform4(&localState);

    // Apply predicate p and store random numbers to global memory
	result[id            ] = (T) p(v4.w);
	result[id + 1 * n / 4] = (T) p(v4.x);
	result[id + 2 * n / 4] = (T) p(v4.y);
	result[id + 3 * n / 4] = (T) p(v4.z);

    // Copy state to global memory
	state[id] = localState;
}

/// Generates uniformly distributed random floats
template<typename State>
__global__ void generateUniformFloatsKernel( State *state
                                           , float *result
                                           , unsigned int n )
{
    // Thread identification
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Local copy of the state
    State localState = state[id];
    
    for( int i = threadIdx.x + blockIdx.x * blockDim.x
       ; i < n
       ; i += blockDim.x * gridDim.x )
    {
        // Generate random numbers
        float4 v4 = curand_uniform4(&localState);

        // Apply predicate p and store random numbers to global memory
        result[i] = v4.w;
        result[i] = v4.x;
        result[i] = v4.y;
        result[i] = v4.z;
    }

    // Copy state to global memory
    state[id] = localState;
}

/// Generates uniformly distributed random floates and applies a lambda function
/// on each of them
template<typename State, typename Predicate>
__global__ void generateUniformFloatsKernel( State *state 
                                           , float *result 
                                           , Predicate p 
                                           , unsigned int n)
{
    // Thread identification
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Local copy of the state
	curandStatePhilox4_32_10_t localState = state[id];

    for( int i = threadIdx.x + blockIdx.x * blockDim.x
       ; i < n
       ; i += blockDim.x * gridDim.x )
    {
        // Generate random numbers
        float4 v4 = curand_uniform4(&localState);

        // Apply predicate p and store random numbers to global memory
        result[i] = p(v4.w);
        result[i] = p(v4.x);
        result[i] = p(v4.y);
        result[i] = p(v4.z);
    }
    // Copy state to global memory
	state[id] = localState;
}

/// Class that holds a curand randnom number generator - its states and random 
/// numbers. When the class goes out of scope, it automatically deallocates 
/// device memory.
template<unsigned int BLOCK_SIZE, typename Numbers, typename State>
class RNG{
    public:
        unsigned int n;      // Number of random numbers
        Numbers* d_rand;     // Pointer to array of random numbers
        State*   d_state;    // State of curand generator
        int DimBlockRNG, DimGridRNG;

        // Constructor - takes care of curand setup
        RNG( unsigned int n, long int globalSeed ){
            this->n           = n;
            int numSMs, devId;
            cudaGetDevice( &devId );
            cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId );
            this->DimBlockRNG = 256;
            this->DimGridRNG  = 32*numSMs;
            
            // Print kernel lunch configuration
            // std::cout << "DimBlockRNG(" << DimBlockRNG.x << ", "
            //           << DimBlockRNG.y << ", " << DimBlockRNG.z
            //           << ")\nDimGridRNG(" << DimGridRNG.x << ", "
            //           << DimGridRNG.y << ", " << DimGridRNG.z
            //           << ")\n" << std::endl;


            // Allocate memory
            CUDAErrChk(cudaMalloc( (void**)& d_rand, this->n*sizeof(Numbers) ));
            CUDAErrChk(cudaMalloc( (void**)& d_state, this->n*sizeof(State)  ));
           
            std::cout << "DimGrid: " << DimGridRNG << ", DimBlock: " << DimBlockRNG << std::endl;
            // Initiate the RNG 
            setupKernel<<<DimGridRNG, DimBlockRNG >>>( d_state, globalSeed );
            CUDAErrChk(cudaPeekAtLastError());
        }
        
        ~RNG(){
            // Deallocate memory
            CUDAErrChk(cudaFree( d_rand  ));
            CUDAErrChk(cudaFree( d_state ));
        }

        // Generates a new batch of n random numbers
        void generate(){
            generateUniformKernel<<<DimGridRNG
                                  , DimBlockRNG >>>
                                  ( d_state
                                  , d_rand
                                  , this->n);
            CUDAErrChk(cudaPeekAtLastError());
        }

        // Generates a new batch of n random numbers and applies predicate p
        // to all of them
        template<typename Predicate>
        void generate( Predicate p ){
            generateUniformKernel<<<DimGridRNG
                                  , DimBlockRNG >>>
                                  ( d_state
                                  , d_rand
                                  , p
                                  , this->n );
            CUDAErrChk(cudaPeekAtLastError());
        }

        // Returns a current batch of random numbers in a form of string
        std::string to_string(){
            // Copy data to host
            std::vector<Numbers> v( this->n );
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(Numbers) 
                                 , cudaMemcpyDeviceToHost )); 
            
            return vector_to_string(v);
        }

        // Copies random numbers from device to a given host memory 
        // std::vector
        void to_vector( std::vector<Numbers>& v ){
            if( v.size() < this->n ){
                std::cout << "Warning: Output vector is too small! Resizing..." 
                          << std::endl;
                v.resize( this->n );
            }

            // Copy data to host
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(Numbers) 
                                 , cudaMemcpyDeviceToHost)); 
        }

        // Returns a copy of current random numbers in a form of a std::vector 
        std::vector<Numbers> to_vector(){
            // Copy data to host
            std::vector<Numbers> v(this->n);
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(Numbers) 
                                 , cudaMemcpyDeviceToHost)); 

            return v;
        }
};

void init_lattice( Lattice* d_s, int globalSeed ){
    #ifdef DEBUG
    std::cout << "generating random configuration" << std::endl;
    #endif
    // Create generator 
    RNG<LBLOCKS, mType, generatorType> generator( RAND_N, globalSeed ); 
    
    // Generate +1 and -1 randomly
    generator.generate( [] __device__ ( mType x )
            {return (mType)(x <= 0.5 ? (mType)(-1.0) : (mType)(+1.0));} );

    // Initialization - Fill the lattice with random spins
    CUDAErrChk(cudaMemcpy( d_s->s1
                         , generator.d_rand
                         , N*sizeof(mType)
                         , cudaMemcpyDeviceToDevice ) );
    CUDAErrChk(cudaMemcpy( d_s->s2
                         , generator.d_rand + N
                         , N*sizeof(mType)
                         , cudaMemcpyDeviceToDevice ) );
    CUDAErrChk(cudaMemcpy( d_s->s3
                         , generator.d_rand + N * 2
                         , N*sizeof(mType)
                         , cudaMemcpyDeviceToDevice ) );
}


void config_to_file( std::string filename
                   , unsigned int latticeSize
                   , unsigned int numSweeps
                   , unsigned int numTemperatures )
{
    std::ofstream configFile( filename );

    configFile << latticeSize       << " "
               << numSweeps         << " "
               << numTemperatures   << " "
               << std::endl;

    configFile.close();
}

template<typename T>
void to_file( std::string filename, std::vector<T> &vec )
{
    std::ofstream file( filename );

    std::for_each( vec.begin()
                 , vec.end() 
                 , [&](T h){
                    file << h << " ";
                 });

    file.close();
}

#endif // CUDA_MC_AUXILIARY_FUNCTIONS_H_
