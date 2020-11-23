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
#include "config.h"

// Include guard
#ifndef MC_MA_AUXILIARY_FUNCTIONS_H_
#define MC_MA_AUXILIARY_FUNCTIONS_H_

struct UniSpins {
  __host__ __device__ __forceinline__ mType operator()(const float& x) const {
    return (mType)(x <= 0.5 ? -1.0 : +1.0);
  }
};

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

/// Class that holds a curand randnom number generator - its states and random 
/// numbers. When the class goes out of scope, it automatically deallocates 
/// device memory.
class RNG{
    public:
        unsigned int n;      // Number of random numbers
        unsigned long long seed;
        float* d_rand;     // Pointer to array of random numbers
        curandGenerator_t gen;

        // Constructor - takes care of curand setup
        RNG( unsigned int n, unsigned long long globalSeed ) 
            : n{n}
            , seed{globalSeed}
        {
            // Allocate memory
            CUDAErrChk(cudaMalloc( (void**)& d_rand, this->n*sizeof(float) ));
            
            // Create curand generator
            CURAND_CALL(curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_PHILOX4_32_10 ));

            // Set seed
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed( gen, seed ));
        }
        
        ~RNG(){
            // Deallocate memory
            if( this->d_rand ) CUDAErrChk(cudaFree( this->d_rand  ));
            if( this->gen    ) CURAND_CALL(curandDestroyGenerator(gen));
        }

        // Generates a new batch of n random numbers
        void generate(){
            CURAND_CALL(curandGenerateUniform(gen, d_rand, n));
        }

        // Returns a current batch of random numbers in a form of string
        std::string to_string(){
            // Copy data to host
            std::vector<float> v( this->n );
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(float) 
                                 , cudaMemcpyDeviceToHost )); 
            
            return vector_to_string(v);
        }

        // Copies random numbers from device to a given host memory 
        // std::vector
        void to_vector( std::vector<float>& v ){
            if( v.size() < this->n ){
                std::cout << "Warning: Output vector is too small! Resizing..." 
                          << std::endl;
                v.resize( this->n );
            }

            // Copy data to host
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(float) 
                                 , cudaMemcpyDeviceToHost)); 
        }

        // Returns a copy of current random numbers in a form of a std::vector 
        std::vector<float> to_vector(){
            // Copy data to host
            std::vector<float> v(this->n);
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(float) 
                                 , cudaMemcpyDeviceToHost)); 

            return v;
        }
};

/// Kernel that takes random numbers, transforms them with functors/lambdas/??
/// and then compies them into lattice
template<typename P1, typename P2, typename P3>
__global__ void init_lattice_kernel( Lattice* d_s
                                   , float* d_rand
                                   , P1 p1
                                   , P2 p2
                                   , P3 p3 ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

     d_s->s1[idx] = (mType) p1( d_rand[idx       ] );
     d_s->s2[idx] = (mType) p2( d_rand[idx +   N ] );
     d_s->s3[idx] = (mType) p3( d_rand[idx + 2*N ] );

}

void init_lattice( Lattice* d_s, float* d_rand ){
    #ifdef DEBUG
    std::cout << "generating random configuration" << std::endl;
    #endif
    
  // Lambda generating +1 and -1 randomly
  // auto uni_spin = [] __device__ ( mType x )
  //         {return (mType)(x <= 0.5 ? (mType)(-1.0) : (mType)(+1.0));};
  UniSpins uni_spin;

    // Initialization - Fill the lattice with random spins
    init_lattice_kernel<<<DimBlockLinear, DimGridLinear>>>
                                              ( d_s
                                              , d_rand
                                              , uni_spin
                                              , uni_spin
                                              , uni_spin );
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


/*
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
*/
