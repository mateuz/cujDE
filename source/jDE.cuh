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
private:
  uint NP;
  uint n_dim;

  float x_min;
  float x_max;

  float * F;
  float * CR;

public:
  jDE( uint, uint, float, float );
  ~jDE();

  void run();
  void update();
  void selection();
  void index_gen();
};

__global__ void updateK(curandState *, float *, float *);

__global__ void selectionK(float *, float *, float *, float *, uint);

__global__ void DE(curandState *, float *, float *, float *, float *, uint *);

__global__ void iGen(curandState *, uint *, uint *);

__global__ void setup_kernel(curandState *);

#endif
