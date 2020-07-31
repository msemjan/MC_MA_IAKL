/**
 * This file contains user-defined lunch parameters.
 *
 **/

// Include guard
#ifndef CUDA_MC_CONFIG_H_
#define CUDA_MC_CONFIG_H_

#define J1 -1                   // Interaction coupling constant
#define L 64                    // Linear lattice size
#define N (L*L)                 // Number of sublattice spins
#define LBLOCKS 16              // Lenght of a block
#define RAND_N (3 * N)          // Number of random numbers
#define field 0                 // External magnetic field
#define SAVE_TS 1
#define SAVE_TEMPERATURES 1 
// #define SAVE_MEANS 1
// #define DEBUG 1

// Typedefs
typedef int mType;              // Magnetization and spins
typedef double eType;           // Energy
typedef double tType;           // Temperature
typedef float rngType;          // RNG generation - numbers
typedef curandStatePhilox4_32_10_t generatorType;
                                // RNG generation - generator
// NOTE: Values of const expressions
// (1<<18) =   252 144
// (1<<19) =   524 288
// (1<<20) = 1 048 576
// (1<<21) = 2 097 152
// (1<<22) = 4 194 304

// Parameters of the simulation
const unsigned int numThermalSweeps = 1<<19;   // Sweeps for thermalization
const unsigned int numSweeps = 1<<20;          // Number of sweeps
const tType minTemperature = 0.0;
const tType maxTemperature = 3.0;
const tType deltaTemperature = 0.7;
const size_t numTemp = 20;
const unsigned int boltzL = 2 * 5;              // # of unique Boltzman factors

#endif // CUDA_MC_CONFIG_H_