/*
 * A port of my original C++ implementation of Metropolis algorithm
 * for Ising antiferromagnet on Kagome Lattice into CUDA.
 *
 * COMPILE WITH:
 * nvcc metropolis.cu -I$HOME/cub -lm -lcurand -o metropolis -arch=sm_75 --expt-extended-lambda
 * For debugging add -G -Xcompiler -rdynamic flags.
 *
 *  Created on: 14.10.2019
 *      Author: <marek.semjan@student.upjs.sk>
 */


// C/C++ imports
#include <fstream>							// C++ type-safe files
#include <iostream>                         // cin, cout
#include <sstream>                          // for higher precision to_string
#include <string>                           // for work with file names
#include <vector>                           // data container
#include <numeric>                          // accumulate
#include <algorithm>
#include <cmath>
#include <limits>                           // std::numeric_limits
#include <chrono>                           // measuring of the time
#include <iomanip>                  		// std::put_time
#include <ctime>                           	// time
#include <cstdio>                           // fread, fwrite, fprintf
#include <cstdlib>
#include <random>
#include <iomanip>                           // std::setprecision

// CUDA specific imports
#include <cub/cub.cuh>                       // For parallel reductions
#include <cuda.h>                            // CUDA header
#include <curand.h>                          // Parallel Random Number Generators

//#define WIN_OS 1
#define LINUX_OS 1

// Auxiliary header files
#include "safe_cuda_macros.cuh"
#include "auxiliary_functions.h"             // Various Helper functions
#include "systemSpecific.h"
#include "fileWrapper.h"
#include "Quantities.cuh"
#include "config.h"

// Texture memory
texture<float, cudaTextureType1D, cudaReadModeElementType> boltz_tex;

/*
o--x-->
|
y
|
V
     |          /    |          /    |         /
---- s1 ---- s2 ---- s1 ---- s2 ---- s1 ---- s2 ----
     |       /       |       /       |       / 
     |     /         |     /         |     / 
     |   /           |   /           |   /  
     | /             | /             | /                 
     s3              s3              s3
    /|              /|              /|              /
  /  |            /  |            /  |            /
     |          /    |          /    |          /       
---- s1 ---- s2 ---- s1 ---- s2 ---- s1 ---- s2 ----  
     |       /       |       /       |       / 
     |     /         |     /         |     / 
     |   /           |   /           |   /  
     | /             | /             | /                 
     s3              s3              s3
    /|              /|              /|              /
  /  |            /  |            /  |            /
     |          /    |          /    |          /       
---- s1 ---- s2 ---- s1 ---- s2 ---- s1 ---- s2 ----  
     |       /       |       /       |       / 
                    
                    Fig. 1
 Kagome lattice and it's sublattices s1, s2 and s3.
*/

// Prototypes of host functions


// Kernel prototypes
__global__ void energyCalculation( Lattice* d_s );
__global__ void update1( Lattice* s
                       , rngType* numbers
                       , unsigned int offset);
__global__ void update2( Lattice* s
                       , rngType* numbers
                       , unsigned int offset);
__global__ void update3( Lattice* s
                       , rngType* numbers
                       , unsigned int offset);

// Boltzman table - works for systems with spin number s = 1/2
std::vector<float> boltz( boltzL );
float *d_boltz;

// =========================================================================
//                                  Main
// =========================================================================
int main() {
    // Time for logging
    std::time_t logTime;
    std::time( &logTime );
    std::tm tm = *std::gmtime( &logTime );
    char buffer[80];

    // Reset device
    cudaDeviceReset();

    // Select device
    cudaSetDevice(0);

    // Randomize random numbers
    srand(time( NULL ));

    // Getting PID of the simulation
    mySys::myPidType pid = mySys::getPid();

    // Random number generator
    int globalSeed = (int)( rand() % std::numeric_limits<int>::max() );
    #ifdef DEBUG
    std::cout << "Calling constructor" << std::endl;
    #endif
    RNG<LBLOCKS, rngType, generatorType> generator( RAND_N, globalSeed ); 

    // Creating folder for files
    strftime( buffer, 80, "%F_%T", &tm );
    std::string dir        = "/media/semjan/DATA/IAKL_cuda/";
    std::string folderName = "Kagome_METRO_2D_" 
                           + std::to_string( L ) 
                           + "x" 
                           + std::to_string( L ) 
                           + "_TMIN_" 
                           + std::to_string( minTemperature ) 
                           + "_TMAX_" 
                           + std::to_string( maxTemperature ) 
                           + "_dT_" 
                           + std::to_string( deltaTemperature ) 
                           + "_MCS_" 
                           + std::to_string( numSweeps ) 
                           + "_J_" 
                           + std::to_string( J1 ) 
                           + "_F_" 
                           + std::to_string( field ) 
                           + "_Start_" 
                           + std::string( buffer ) 
                           + "PID" 
                           + std::to_string( pid ) 
                           + "SEED"
                           + std::to_string( globalSeed )
                           + "/";

    mySys::mkdir( dir + folderName );

    // Creating file names
    std::string logFileName      = dir + folderName + "log.txt";
    std::string meansFileName    = dir + folderName + "means.txt";
    std::string tsFileName       = dir + folderName + "ts.txt";
    std::string tempFileName     = dir + folderName + "temp.txt";
    std::string simFileName      = dir + folderName + "sim_config.txt";
    std::string latticeFilename  = dir + folderName + "lattice.txt";

    // Opening a log file
    std::ofstream logFile( logFileName.c_str() );

    // Logging something
    logFile << "   Monte Carlo - Metropolis in CUDA - Kagome lattice\n"
            << "======================================================\n"
            << "Lattice size:\t" 
            << L 
            << "x" 
            << L 
            << "\nJ:\t\t\t\t" 
            << J1 
            << "\nSweeps: \t\t" 
            << numSweeps 
            << "\nMinTemp:\t\t" 
            << minTemperature 
            << "\nMaxTemp:\t\t" 
            << maxTemperature 
            << "\ndTemp:\t\t\t" 
            << deltaTemperature 
            << "\nSEED:\t\t\t"
            << globalSeed
            << "\nMy PID:\t\t\t" 
            << pid 
            << "\nPrototyp v1.0\n"
            << "======================================================\n" 
            << "[" 
            << std::put_time( &tm, "%F %T" ) 
            << "] Simulation started..." 
            << std::endl;

    logFile.precision( 12 );

    // Lattice preparation
    Lattice s;				// Lattice
    Lattice* ss_host = &s;	// Pointer to the lattice
    Lattice* d_s;			// Device pointer to lattice

    // Allocate lattice
    CUDAErrChk(cudaMalloc( (void**)&d_s, sizeof(Lattice)) );

    // Prepare lattice
    init_lattice( d_s, globalSeed );

    #ifdef DEBUG
    // Copy init config of the lattice and save it
    std::cout << "Saving the initial lattice configuration" << std::endl;
    CUDAErrChk(cudaMemcpy( ss_host, d_s, sizeof(Lattice), cudaMemcpyDeviceToHost));
    to_file( ss_host, dir + folderName + "init_lattice.txt" );

    // Clean up
    // CUDAErrChk(cudaFree( d_s ));
    // return 0;
    #endif

    // Preparation of temperatures
    std::vector<double> temperature;
    // temperature.push_back(maxTemperature);
    for( int i = 0
       ; i < numTemp
       ; i++ )
    {
        // temperature.push_back( temperature[0] - i * deltaTemperature );
        temperature.push_back( maxTemperature * std::pow( deltaTemperature, i ) );
    }

    // Creating object for recording values of observable quantites
    Quantities q( numSweeps, temperature.size(), d_s );

    // Prepare array for Bolzman factors
    CUDAErrChk(cudaMalloc( (void**)& d_boltz, boltzL*sizeof(float)) );
    CUDAErrChk(cudaBindTexture( NULL
                              , boltz_tex
                              , d_boltz
                              , boltzL * sizeof(float) ));

    // Lunch specifications
    dim3 DimBlock(L / LBLOCKS, L / LBLOCKS, 1);
    dim3 DimGrid(LBLOCKS, LBLOCKS, 1);
    // dim3 DimBlock(1, 1, 1);
    // dim3 DimGrid(1, 1, 1);

    try {
        double beta;

        // Temperature loop
        for(int tempCounter = 0
           ; tempCounter < temperature.size()
           ; tempCounter++ )
        {
            beta = 1/temperature[tempCounter]; 

            // Generation of Boltzman factors
            for( int idx1 = 0; idx1 <= 2; idx1 += 2 ){
                for( int idx2 = 0; idx2 <= 8; idx2 += 2 ){
                    boltz[idx1 / 2 + idx2] =
                        exp( -beta * 2 * (idx1 - 1)*(J1*(idx2 - 4) + field) );
                }
            }

            // Copy Boltzman factors to device
            CUDAErrChk(cudaMemcpy( d_boltz
                                 , boltz.data()
                                 , boltzL * sizeof(float)
                                 , cudaMemcpyHostToDevice ));

            // Loop over sweeps - thermalization
            for( int sweep = 0; sweep < numThermalSweeps; sweep++ ){
// #ifdef DEBUG
//                 std::cout << "THERMALIZATION: S" << sweep << " T: " << tempCounter << std::endl;
// #endif
                // Generate random numbers
                generator.generate();

                // Lunch kernels :)
                update1<<<DimBlock,DimGrid>>>( d_s, generator.d_rand,   0 );
                CUDAErrChk(cudaPeekAtLastError());
                cudaDeviceSynchronize();
                update2<<<DimBlock,DimGrid>>>( d_s, generator.d_rand,   N );
                CUDAErrChk(cudaPeekAtLastError());
                cudaDeviceSynchronize();
                update3<<<DimBlock,DimGrid>>>( d_s, generator.d_rand, 2*N );
                CUDAErrChk(cudaPeekAtLastError());
                cudaDeviceSynchronize();
            }

            // Loop over sweeps - recording quantities
            for( int sweep = 0; sweep < numSweeps; sweep++ ){
// #ifdef DEBUG
//                 std::cout << "Sampling: S:" << sweep << " T: " << tempCounter << std::endl;
// #endif

                // Generate random numbers
                generator.generate();

                // Lunch kernels
                update1<<<DimBlock,DimGrid>>>( d_s, generator.d_rand,   0 );
                CUDAErrChk(cudaPeekAtLastError());
                cudaDeviceSynchronize();
                update2<<<DimBlock,DimGrid>>>( d_s, generator.d_rand,   N );
                CUDAErrChk(cudaPeekAtLastError());
                cudaDeviceSynchronize();
                update3<<<DimBlock,DimGrid>>>( d_s, generator.d_rand, 2*N );
                CUDAErrChk(cudaPeekAtLastError());
                cudaDeviceSynchronize();

                // Calculate energy
                energyCalculation<<<DimBlock, DimGrid>>>( d_s );

                // Calculate observables
                q.getObservables( d_s, tempCounter, sweep );
            }
            
            // Calculate means for current temperature
            q.means( tempCounter );

            // Logging stuff
            std::time( &logTime );
            tm = *std::gmtime( &logTime );
            logFile << std::put_time( &tm, "[%F %T] " ) 
                    << "Finished  another loop. Last beta = " 
                    << beta
                    << std::endl;
        
        } // end of temperature loop

    } catch (std::exception& e) {
        // Comrades made mistake and things went wrong, if we are here!
        std::cout << "Exception occured!\n" << e.what() << std::endl;
        std::cin.ignore();
        std::cin.ignore();
    }

    // Save simulation parameters to a file
    config_to_file( simFileName, L, numSweeps, numTemp ); 

    #ifdef SAVE_TEMPERATURES
    to_file( tempFileName, temperature );
    #endif

    #ifdef SAVE_TS
    // Save time series
    q.save_TS_to_file( tsFileName );
    #endif

    #ifdef SAVE_MEANS
    // Save means
    q.save_mean_to_file( meansFileName, temperature );
    #endif

    // Copy lattice to device
    CUDAErrChk(cudaMemcpy( ss_host, d_s, sizeof(Lattice), cudaMemcpyDeviceToHost));

    // Save lattice
    to_file( ss_host, latticeFilename );
	
    // Closing files
	logFile.close();
    
    // Clean up
    CUDAErrChk(cudaUnbindTexture( boltz_tex ));
    CUDAErrChk(cudaFree( d_s ));
    CUDAErrChk(cudaFree( d_boltz ));

	return 0;
}

// =========================================================================
//                       Functions and Kernels
// =========================================================================

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
            * (J1 * (eType) d_s->s1[x + L * y]
                    * ( (eType) d_s->s2[x + L * y]
                      + (eType) d_s->s3[x + L * y]
                      + (eType) (d_s->s3[x + yD * L]) )
                    +
                    J1 * (eType) d_s->s2[x + L * y]
                            * ((eType) d_s->s3[x + L * y]
                               + (eType) (d_s->s3[xU + yD * L])
                               + (eType) (d_s->s1[xU + y * L]))));
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