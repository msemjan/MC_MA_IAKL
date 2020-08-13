#ifndef QUANTITIES_CUH_
#define QUANTITIES_CUH_

// C/C++ imports
#include <fstream>							// C++ type-safe files
#include <iostream>                         // cin, cout
#include <sstream>                          // for higher precision to_string
#include <string>                           // for work with file names
#include <vector>                           // data container
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
#include "config.h"
#include "Lattice.h"

// Functor for squaring values with normalization to number of spins
template<typename T>
struct PerVolume 
{
    __host__ __device__ __forceinline__
        double operator()( const T &a ) const {
            return double( a / VOLUME );
        }
};

// Functor for squaring values
template<typename T>
struct Square
{
    __host__ __device__ __forceinline__
        double operator()( const T &a ) const {
            return double( a * a );
        }
};

// Functor for squaring values with normalization to number of spins
template<typename T>
struct SquarePerVolume
{
    __host__ __device__ __forceinline__
        double operator()( const T &a ) const {
            return double( a * a / VOLUME );
        }
};

class Quantities{
public:
    unsigned int numSweeps, numTemp;
    eType *d_energy, *d_mEnergy, *d_mEnergySq, *d_temp_storage_e; //, d_exchangeEnergy;
    mType *d_m1, *d_m2, *d_m3, *d_mm1, *d_mm2, *d_mm3, 
          *d_mm1Sq, *d_mm2Sq, *d_mm3Sq, *d_temp_storage_m, *d_temp_storage_s,
          *d_temp_storage_ee;
    size_t temp_storage_bytes_e, temp_storage_bytes_m, temp_storage_bytes_s, 
           temp_storage_bytes_ee;
    
    /// Constructor
    Quantities( unsigned int numSweeps, unsigned int numTemp, Lattice* d_s ){
        this->numSweeps = numSweeps;
        this->numTemp   = numTemp;
        
        // Allocation of memory
        CUDAErrChk(cudaMalloc( (void**)&d_energy,    numSweeps*numTemp*sizeof(eType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mEnergy,   numSweeps*numTemp*sizeof(eType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mEnergySq, numSweeps*numTemp*sizeof(eType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_m1,        numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_m2,        numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_m3,        numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mm1,       numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mm2,       numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mm3,       numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mm1Sq,     numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mm2Sq,     numSweeps*numTemp*sizeof(mType) ));
        CUDAErrChk(cudaMalloc( (void**)&d_mm3Sq,     numSweeps*numTemp*sizeof(mType) ));

        // CUB boilerplate
        d_temp_storage_e      = NULL;
        d_temp_storage_m      = NULL;
        d_temp_storage_s      = NULL;
        d_temp_storage_ee     = NULL;
        temp_storage_bytes_e  = 0;
        temp_storage_bytes_m  = 0;
        temp_storage_bytes_s  = 0;
        temp_storage_bytes_ee = 0;

        // CUB preparation
        cub::DeviceReduce::Sum( d_temp_storage_e
                              , temp_storage_bytes_e 
                              , d_energy
                              , d_mEnergy 
                              , numSweeps);
        CUDAErrChk(cudaMalloc( (void**)& d_temp_storage_e
                              , temp_storage_bytes_e));


        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m 
                              , d_m1
                              , d_mm1 
                              , numSweeps);
        CUDAErrChk(cudaMalloc( (void**)& d_temp_storage_m
                              , temp_storage_bytes_m));

        cub::DeviceReduce::Sum( d_temp_storage_s
                              , temp_storage_bytes_s 
                              , d_s->s1
                              , d_m1  
                              , N);
        CUDAErrChk(cudaMalloc( (void**)& d_temp_storage_s
                              , temp_storage_bytes_s));

        cub::DeviceReduce::Sum( d_temp_storage_ee
                              , temp_storage_bytes_ee 
                              , d_s->exchangeEnergy
                              , d_energy
                              , N);
        CUDAErrChk(cudaMalloc( (void**)& d_temp_storage_ee
                              , temp_storage_bytes_ee));

        #ifdef DEBUG
        std::cout << "Allocated " 
                  << temp_storage_bytes_e 
                  << " bytes at adress " 
                  << temp_storage_bytes_e 
                  << " for energy"
                  << std::endl;

        std::cout << "Allocated " 
                  << temp_storage_bytes_m 
                  << " bytes at adress " 
                  << temp_storage_bytes_m 
                  << " for magnetization"
                  << std::endl;
        #endif
    }

    /// Destructor
    ~Quantities(){
        // Deallocation of memory
        if( d_energy          ) CUDAErrChk(cudaFree( d_energy          ));
        if( d_m1              ) CUDAErrChk(cudaFree( d_m1              ));
        if( d_m2              ) CUDAErrChk(cudaFree( d_m2              ));
        if( d_m3              ) CUDAErrChk(cudaFree( d_m3              ));
        if( d_mEnergy         ) CUDAErrChk(cudaFree( d_mEnergy         ));
        if( d_mm1             ) CUDAErrChk(cudaFree( d_mm1             ));
        if( d_mm2             ) CUDAErrChk(cudaFree( d_mm2             ));
        if( d_mm3             ) CUDAErrChk(cudaFree( d_mm3             ));
        if( d_mEnergySq       ) CUDAErrChk(cudaFree( d_mEnergySq       ));
        if( d_mm1Sq           ) CUDAErrChk(cudaFree( d_mm1Sq           ));
        if( d_mm2Sq           ) CUDAErrChk(cudaFree( d_mm2Sq           ));
        if( d_mm3Sq           ) CUDAErrChk(cudaFree( d_mm3Sq           ));
        if( d_temp_storage_e  ) CUDAErrChk(cudaFree( d_temp_storage_e  ));
        if( d_temp_storage_m  ) CUDAErrChk(cudaFree( d_temp_storage_m  ));
        if( d_temp_storage_s  ) CUDAErrChk(cudaFree( d_temp_storage_s  ));
        if( d_temp_storage_ee ) CUDAErrChk(cudaFree( d_temp_storage_ee ));
    }

    /// Calculate sublattice magnetizations and energy from lattice configuration
    void getObservables( Lattice* s, unsigned int temp, unsigned int sweep ){
        // Calculate sublattice magnetizations and internal energy
        cub::DeviceReduce::Sum( d_temp_storage_s
                              , temp_storage_bytes_s 
                              , s->s1
                              , d_m1 + sweep + temp * numSweeps  
                              , N);

        cub::DeviceReduce::Sum( d_temp_storage_s
                              , temp_storage_bytes_s 
                              , s->s2
                              , d_m2 + sweep + temp * numSweeps  
                              , N);
        
        cub::DeviceReduce::Sum( d_temp_storage_s
                              , temp_storage_bytes_s 
                              , s->s3
                              , d_m3 + sweep + temp * numSweeps  
                              , N);

        cub::DeviceReduce::Sum( d_temp_storage_ee
                              , temp_storage_bytes_ee 
                              , s->exchangeEnergy
                              , d_energy + sweep + temp * numSweeps  
                              , N);

    }

    /// Calculates mean values of observables
    void means( unsigned int temp ){
        // Operators
        // PerVolume<mType> M_op;
        PerVolume<eType> E_op;
        SquarePerVolume<mType> sqM_op;
        SquarePerVolume<eType> sqE_op;

        // Create an iterator wrapper
        cub::TransformInputIterator< double
                                   , PerVolume<eType>
                                   , eType*>
                               it_e( d_energy + temp * numSweeps
                                   , E_op );
        
        // cub::TransformInputIterator< double
        //                            , PerVolume<mType>
        //                            , mType*>
        //                       it_m1( d_m1 + temp * numSweeps
        //                            , M_op );
        //
        // cub::TransformInputIterator< double
        //                            , PerVolume<mType>
        //                            , mType*>
        //                       it_m2( d_m2 + temp * numSweeps
        //                            , M_op );
        //
        // cub::TransformInputIterator< double
        //                            , PerVolume<mType>
        //                            , mType*>
        //                       it_m3( d_m3 + temp * numSweeps
        //                            , M_op );
        
        cub::TransformInputIterator< double
                                   , SquarePerVolume<eType>
                                   , eType*>
                             it_eSq( d_energy + temp * numSweeps
                                   , sqE_op );
        
        cub::TransformInputIterator< double
                                   , SquarePerVolume<mType>
                                   , mType*>
                            it_m1Sq( d_m1 + temp * numSweeps
                                   , sqM_op );
        
        cub::TransformInputIterator< double
                                   , SquarePerVolume<mType>
                                   , mType*>
                            it_m2Sq( d_m2 + temp * numSweeps
                                   , sqM_op );
        
        cub::TransformInputIterator< double
                                   , SquarePerVolume<mType>
                                   , mType*>
                            it_m3Sq( d_m3 + temp * numSweeps
                                   , sqM_op );

        // Reductions
        cub::DeviceReduce::Sum( d_temp_storage_e
                              , temp_storage_bytes_e
                              , it_e 
                              , d_mEnergy + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m
                              // , it_m1
                              , d_m1 + temp * numSweeps
                              , d_mm1 + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m
                              // , it_m2
                              , d_m2 + temp * numSweeps
                              , d_mm2 + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m
                              // , it_m3
                              , d_m3 + temp * numSweeps
                              , d_mm3 + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_e
                              , temp_storage_bytes_e
                              , it_eSq
                              , d_mEnergySq + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m
                              , it_m1Sq
                              , d_mm1Sq + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m
                              , it_m2Sq
                              , d_mm2Sq + temp
                              , numSweeps );
        
        cub::DeviceReduce::Sum( d_temp_storage_m
                              , temp_storage_bytes_m
                              , it_m3Sq
                              , d_mm3Sq + temp
                              , numSweeps );
    }

    /// Copies time series from device to host and save them to a given file
    void save_TS_to_file( std::string filename ){
        // Open a file for writing 
        std::ofstream ts_file( filename );

        // First line contains number of sweeps and number of temperatures
        // ts_file << numSweeps << " " << numTemp << std::endl;

        // Create host vectors
        std::vector<eType> energy( numSweeps*numTemp );
        std::vector<mType> m1( numSweeps*numTemp );
        std::vector<mType> m2( numSweeps*numTemp );
        std::vector<mType> m3( numSweeps*numTemp );

        // Copy data
        CUDAErrChk(cudaMemcpy( energy.data()
                             , d_energy
                             , numSweeps*numTemp*sizeof(eType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( m1.data()
                             , d_m1
                             , numSweeps*numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( m2.data()
                             , d_m2
                             , numSweeps*numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( m3.data()
                             , d_m3
                             , numSweeps*numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        
        ts_file.precision(30);

        // Save to the file 
        #ifdef CREATE_HEADERS
        // Create a header
        ts_file << "E "
                << "m1 "
                << "m2 "
                << "m3 "
                << std::endl;
        #endif

        for(int i = 0; i < energy.size(); i++){
            ts_file << energy[i] << " "
                    << m1[i]     << " "
                    << m2[i]     << " "
                    << m3[i]     << std::endl;
        }

        // std::for_each( energy.begin()
        //              , energy.end()
        //              , [&](eType h){
        //                 ts_file << h << " ";
        //               });
        //
        // ts_file << std::endl;
        //
        // std::for_each( m1.begin()
        //              , m1.end()
        //              , [&](mType h){
        //                 ts_file << h << " ";
        //               });
        //
        // ts_file << std::endl;
        //
        // std::for_each( m2.begin()
        //              , m2.end()
        //              , [&](mType h){
        //                 ts_file << h << " ";
        //               });
        //
        // ts_file << std::endl;
        //
        // std::for_each( m3.begin()
        //              , m3.end()
        //              , [&](mType h){
        //                 ts_file << h << " ";
        //               });

        ts_file << std::endl;
        ts_file.close();
    }

    /// Copies means to host and saves them to a given file
    void save_mean_to_file( std::string filename, std::vector<tType> &temperature ){
        // Open a file for writing 
        std::ofstream mean_file( filename );

        // First line contains number of temperatures
        // mean_file << numTemp << std::endl;

        // Create host vectors
        std::vector<eType> mEnergy(   numTemp );
        std::vector<eType> mEnergySq( numTemp );
        std::vector<mType> mm1(       numTemp );
        std::vector<mType> mm2(       numTemp );
        std::vector<mType> mm3(       numTemp );
        std::vector<mType> mm1Sq(     numTemp );
        std::vector<mType> mm2Sq(     numTemp );
        std::vector<mType> mm3Sq(     numTemp );

        // Copy data
        CUDAErrChk(cudaMemcpy( mEnergy.data()
                             , d_mEnergy
                             , numTemp*sizeof(eType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mEnergySq.data()
                             , d_mEnergySq
                             , numTemp*sizeof(eType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mm1.data()
                             , d_mm1
                             , numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mm2.data()
                             , d_mm2
                             , numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mm3.data()
                             , d_mm3
                             , numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mm1Sq.data()
                             , d_mm1Sq
                             , numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mm2Sq.data()
                             , d_mm2Sq
                             , numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        CUDAErrChk(cudaMemcpy( mm3Sq.data()
                             , d_mm3Sq
                             , numTemp*sizeof(mType)
                             , cudaMemcpyDeviceToHost ));
        
        mean_file.precision(30);

        #ifdef CREATE_HEADERS
        // Create a header
        mean_file << "T " 
                  << "E "
                  << "E2 "
                  << "m1 "
                  << "m2 "
                  << "m3 "
                  << "m1Sq "
                  << "m2Sq "
                  << "m3Sq\n";
        #endif

        // Save to the file 
        for(int i = 0; i < temperature.size(); i++){
            mean_file << temperature[i] << " " 
                      << mEnergy[i]   / numSweeps  << " "
                      << mEnergySq[i] / numSweeps  << " "
                      << mm1[i]       / numSweeps  << " "
                      << mm2[i]       / numSweeps  << " "
                      << mm3[i]       / numSweeps  << " "
                      << mm1Sq[i]     / numSweeps  << " "
                      << mm2Sq[i]     / numSweeps  << " "
                      << mm3Sq[i]     / numSweeps  << std::endl;
        }

        // std::for_each( temperature.begin()
        //              , temperature.end()
        //              , [&](tType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mEnergy.begin()
        //              , mEnergy.end()
        //              , [&](eType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // // Save to the file
        // std::for_each( mEnergySq.begin()
        //              , mEnergySq.end()
        //              , [&](eType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mm1.begin()
        //              , mm1.end()
        //              , [&](mType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mm2.begin()
        //              , mm2.end()
        //              , [&](mType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mm3.begin()
        //              , mm3.end()
        //              , [&](mType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mm1Sq.begin()
        //              , mm1Sq.end()
        //              , [&](mType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mm2Sq.begin()
        //              , mm2Sq.end()
        //              , [&](mType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;
        //
        // std::for_each( mm3Sq.begin()
        //              , mm3Sq.end()
        //              , [&](mType h){
        //                 mean_file << h << " ";
        //               });
        //
        // mean_file << std::endl;

        mean_file.close();
    }
};

#endif // QUANTITIES_CUH_
