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

#endif // CUDA_MC_LATTICE_H_
