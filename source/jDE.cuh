#ifndef __jDE__
#define __jDE__

#include "helper.cuh"
#include "constants.cuh"

/* C++ includes */
#include <tuple>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>
#include <random>

class jDE {
protected:
  uint NP;
  uint n_dim;
  uint n_threads;
  uint n_blocks;

  float x_min;
  float x_max;

  /* device data */
  curandState * d_states;
  curandState * d_states2;

  uint * rseq;
  uint * fseq;
  float * F;
  float * CR;

public:
  jDE( uint, uint, float, float );
  ~jDE();

  uint iDivUp(uint a, uint b);
  void run(float *, float *);
  void update();
  void selection(float *, float *, float *, float *);
  void index_gen();
};

__global__ void updateK(curandState *, float *, float *);

__global__ void selectionK(float *, float *, float *, float *);

__global__ void DE(curandState *, float *, float *, float *, float *, uint *);

__global__ void mDE(curandState *, float *, float *, float *, float *, uint *);

__global__ void iGen(curandState *, uint *, uint *);

__global__ void setup_kernel(curandState *, uint);

__global__ void sk2(curandState *, uint);

#endif
