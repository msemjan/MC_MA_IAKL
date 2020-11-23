#ifndef CUDA_MC_LATTICE_H_
#define CUDA_MC_LATTICE_H_

#include <fstream>
#include <string>
#include "config.h"
#include <cuda.h>

// Class that stores the lattice
class Lattice {
public:
	mType s1[N];
	mType s2[N];
	mType s3[N];
    eType exchangeEnergy[N];
	__device__ __host__ Lattice() {
	}

};

void to_file( Lattice* s, std::string filename ){
    std::ofstream latticeFile( filename );

    for(int i = 0; i < N; i++){
        latticeFile << s->s1[i] << " "
                    << s->s2[i] << " "
                    << s->s3[i] << " "
                    #ifdef DEBUG
                    << s->exchangeEnergy[i] << " "
                    #endif
                    << std::endl;
    }       

    latticeFile.close();
}

/// Calculates table of Boltzman factors for all possible combinations of 
/// local spin values
void generate_Boltzman_factors( double beta ){
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
}

#endif // CUDA_MC_LATTICE_H_
