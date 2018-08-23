#include "F3.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F3::F3(uint _dim, uint _ps):Benchmarks()
{
  n_dim = _dim;
  ps = _ps;
  min = -600.0;
  max = +600.0;
  ID = 3;

  n_threads = 32;
  n_blocks = (ps%n_threads)? (ps/n_threads)+1 : (ps/n_threads);

  /* ---------------------------------------------- */
  /* Load a shift vector to test the bench function */
  std::string file_name = "data-files/shift_griewank.mat";
  std::string vec_name = "Shift - Griewank [-600.0, +600.0]";
  IO * io = new IO();
  std::ifstream file(file_name);
  if( not file.is_open() ){
    std::cout << "\"data-files/shift_griewank.mat\" could not be opened\n";
    exit(-1);
  }
  auto loaded_vec = io->load_vector<float>( vec_name, file ) ;
  file.close();
  /* ---------------------------------------------- */

  checkCudaErrors(cudaMemcpyToSymbol(shift, (void *) loaded_vec.data(), n_dim * sizeof(float)));
}

F3::~F3()
{
  /*empty*/
}

__global__ void computeK3(float * x, float * f){
  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  if( id_p < ps ){
    uint ndim = params.n_dim;
    uint id_d = id_p * ndim;
    float s1 = 0.0, s2 = 1.0, z;
    for(uint i = 0; i < ndim; i++){
       z = x[id_d + i] - shift[i];
       s1 += (z * z);
       s2 *= cosf(z/sqrtf(i+1));
    }
    s1 /= 4000.0;
    //printf("%u => %.20E\n", id_p, res);
    f[id_p] = (s1 - s2 + 1.0);
  }
}

void F3::compute(float * x, float * f){
  computeK3<<< n_blocks, n_threads >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
