/**
 * This file contains user-defined lunch parameters.
 *
 **/

// Include guard
#ifndef CUDA_MC_CONFIG_H_
#define CUDA_MC_CONFIG_H_

#define J1 -1                   // Interaction coupling constant
#define L 1024                  // Linear lattice size
#define N (L*L)                 // Number of sublattice spins
#define VOLUME (3 * N)          // Total number of spins
#define LBLOCKS 32              // Lenght of a block
#define RAND_N (64 * VOLUME)    // Number of random numbers
#define field 0                 // External magnetic field
// #define SAVE_TS 1
#define SAVE_TEMPERATURES 1
#define SAVE_MEANS 1
#define SAVE_LAST_CONFIG 1
// #define DEBUG 1
// #define SAVE_LATTICES 1
// #define CREATE_HEADERS 1

// Typedefs
typedef int mType;              // Magnetization and spins
typedef double eType;           // Energy
typedef double tType;           // Temperature
typedef float rngType;          // RNG generation - numbers
typedef curandStatePhilox4_32_10_t generatorType;
                                // RNG generation - generator
// NOTE: Values of const expressions
// (1<<18) =   262 144
// (1<<19) =   524 288
// (1<<20) = 1 048 576
// (1<<21) = 2 097 152
// (1<<22) = 4 194 304

// Parameters of the simulation
const unsigned int numThermalSweeps = 1<<18;   // Sweeps for thermalization
const unsigned int numSweeps        = 1<<18;       // Number of sweeps
const tType minTemperature          = 0.0;
const tType maxTemperature          = 3.0;
const tType deltaTemperature        = 0.7;
const size_t numTemp                = 1;
const unsigned int boltzL           = 2 * 5;   // # of unique Boltzman factors

// Lunch specifications
dim3 DimBlock(L / LBLOCKS, L / LBLOCKS, 1);
dim3 DimGrid(LBLOCKS, LBLOCKS, 1);
// dim3 DimBlock(1, 1, 1);
// dim3 DimGrid(1, 1, 1);

// Global variables
// Texture memory
texture<float, cudaTextureType1D, cudaReadModeElementType> boltz_tex;

// Boltzman table - works for systems with spin number s = 1/2
std::vector<float> boltz( boltzL );
float *d_boltz;

#endif // CUDA_MC_CONFIG_H_
