#ifndef MC_MA_KERNELS_CUH_
#define MC_MA_KERNELS_CUH_

// C/C++ imports
#include <algorithm>
#include <chrono>  // measuring of the time
#include <cmath>
#include <cstdio>  // fread, fwrite, fprintf
#include <cstdlib>
#include <ctime>     // time
#include <fstream>   // C++ type-safe files
#include <iomanip>   // std::put_time
#include <iomanip>   // std::setprecision
#include <iostream>  // cin, cout
#include <limits>    // std::numeric_limits
#include <numeric>   // accumulate
#include <random>
#include <sstream>  // for higher precision to_string
#include <string>   // for work with file names
#include <vector>   // data container

// CUB
#include <cub/cub.cuh>  // For parallel reductions

// CUDA specific imports
#include <cuda.h>    // CUDA header
#include <curand.h>  // Parallel Random Number Generators

//#define WIN_OS 1
#define LINUX_OS 1

// Auxiliary header files
#include "auxiliary_functions.h"  // Various Helper functions
#include "fileWrapper.h"
#include "safe_cuda_macros.cuh"
#include "systemSpecific.h"

// Simulation related imports
#include "Quantities.cuh"
#include "config.h"

/// Calculates the local energy of lattice
__global__ void energyCalculation(Lattice* d_s) {
    // Thread identification
    unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Shifts in x and y direction
    // unsigned short yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1
    // unsigned short xU = (x + 1 == L) ? 0 : (x + 1);        // x + 1
    unsigned short yD = (y==0)*(L-1) + (y!=0)*(y-1); // y - 1
    unsigned short xU = (x!=L-1)*(x+1);              // x + 1
    // printf( "x=%hu y=%hu xU=%hu yD=%hu\n", x, y, xU, yD );

    // Calculation of energy
    d_s->exchangeEnergy[x + L * y] = (-1
            * ( J1 * (eType) d_s->s1[x + L * y]
                            * ( (eType) d_s->s2[x + L * y]
                              + (eType) d_s->s3[x + L * y]
                              + (eType) (d_s->s3[x + yD * L]) )
              + J1 * (eType) d_s->s2[x + L * y]
                            * ( (eType) d_s->s3[x + L * y]
                              + (eType) (d_s->s3[xU + yD * L])
                              + (eType) (d_s->s1[xU + y * L]))));
    // d_s->exchangeEnergy[x + L * y] = (-1
    //         * (J1 * (eType) d_s->s1[x + L * y]
    //                 * ( (eType) d_s->s2[x + L * y]
    //                   + (eType) d_s->s3[x + L * y]
    //                   + (eType) (d_s->s3[x + yD * L]) )
    //                 +
    //                 J1 * (eType) d_s->s2[x + L * y]
    //                         * ((eType) d_s->s3[x + L * y]
    //                            + (eType) (d_s->s3[xU + yD * L])
    //                            + (eType) (d_s->s1[xU + y * L]))));
}

/// Tries to flip each spin of the sublattice 1
__global__ void update1( Lattice* s
                       , rngType* numbers
                       , unsigned int offset)
{
    unsigned short x, y, xD, yD;
    double p;
    eType sumNN;
    mType s1;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;
    // xD = (x - 1 == -1) ? (L - 1) : (x - 1); // x - 1
    // yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1
    xD = (x==0)*(L-1) + (x!=0)*(x-1);          // x - 1
    yD = (y==0)*(L-1) + (y!=0)*(y-1);          // y - 1

    s1    = s->s1[L * y  + x ];
    sumNN = s->s2[L * y  + x ] 
          + s->s2[L * y  + xD] 
          + s->s3[L * y  + x ]
          + s->s3[L * yD + x ];

    p = tex1Dfetch( boltz_tex, (s1 + 1) / 2 + 4 + sumNN );
    
    s->s1[L * y + x] *= 1 - 2*((mType)(numbers[offset + L*y+x]<p));
}

/// Tries to flip each spin of the sublattice 2
__global__ void update2( Lattice* s
                       , rngType* numbers
                       , unsigned int offset)
{
	unsigned short x, y, xU, yD;
	double p;
    mType s2;
    eType sumNN;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;
    // xU = (x + 1 == L) ? 0 : (x + 1); // x + 1
    // yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1
    xU = (x!=L-1)*(x+1);                     // x + 1
    yD = (y==0)*(L-1) + (y!=0)*(y-1);        // y - 1
    
    s2    = s->s2[L * y  + x ]; 
    sumNN = s->s1[L * y  + x ] 
          + s->s1[L * y  + xU] 
          + s->s3[L * y  + x ]
          + s->s3[L * yD + xU];

    p = tex1Dfetch( boltz_tex, (s2 + 1) / 2 + 4 + sumNN );
    
    s->s2[L * y + x] *= 1 - 2*((mType)(numbers[offset + L*y+x]<p));
}

/// Tries to flip each spin of the sublattice 3
__global__ void update3( Lattice* s
                       , rngType* numbers
                       , unsigned int offset)
{
	unsigned short x, y, xD, yU;
	double p;
    mType s3;
    eType sumNN;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;
    // xD = (x - 1 == -1) ? (L - 1) : (x - 1); // x - 1
    // yU = (y + 1 == L) ? 0 : (y + 1); // y + 1
    xD = (x==0)*(L-1) + (x!=0)*(x-1);           // x - 1
    yU = (y!=L-1)*(y+1);                        // y + 1

    s3    = s->s3[L * y  + x ]; 
    sumNN = s->s1[L * y  + x ] 
          + s->s1[L * yU + x ] 
          + s->s2[L * y  + x ]
          + s->s2[L * yU + xD];
    
    p = tex1Dfetch( boltz_tex, (s3 + 1) / 2 + 4 + sumNN );
    
    s->s3[L * y + x] *= 1 - 2*((mType)(numbers[offset + L*y+x]<p));
}
    
/// Updates all sublattices
void update( Lattice* s
           , rngType* numbers
           , unsigned int offset)
{
    update1<<<DimBlock,DimGrid>>>( s, numbers,   0 + offset );
    CUDAErrChk(cudaPeekAtLastError());

    update2<<<DimBlock,DimGrid>>>( s, numbers,   N + offset );
    CUDAErrChk(cudaPeekAtLastError());

    update3<<<DimBlock,DimGrid>>>( s, numbers, 2*N + offset );
    CUDAErrChk(cudaPeekAtLastError());
}

#endif
