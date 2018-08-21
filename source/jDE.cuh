#ifndef __jDE__
#define __jDE__

#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "helper.cuh"
#include "constants.cuh"
#include <cstdio>

class jDE {
private:
  uint NP;
  uint n_dim;

  float * F;
  float * CR;

public:
  jDE( uint, uint );
  ~jDE();

  void run();
  void update();
  void selection();
};

__global__ void updateK(curandState *, float *, float *);

__global__ void selectionK(float *, float *, float *, float *, uint);

__global__ void DE(curandState *, float *, float *, float *, float *, uint *);
#endif
