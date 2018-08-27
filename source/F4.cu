#include "F4.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F4::F4(uint _dim, uint _ps):Benchmarks()
{
  n_dim = _dim;
  ps = _ps;
  min = -5.0;
  max = +5.0;
  ID = 4;

  n_threads = 32;
  n_blocks = (ps%n_threads)? (ps/n_threads)+1 : (ps/n_threads);

  /* ---------------------------------------------- */
  /* Load a shift vector to test the bench function */
  std::string file_name = "data-files/shift_rastrigin.mat";
  std::string vec_name = "Shift - Rastrigin [-5.0, +5.0]";
  IO * io = new IO();
  std::ifstream file(file_name);
  if( not file.is_open() ){
    std::cout << "\"data-files/shift_rastrigin.mat\" could not be opened\n";
    exit(-1);
  }
  auto loaded_vec = io->load_vector<float>( vec_name, file ) ;
  file.close();
  /* ---------------------------------------------- */

  checkCudaErrors(cudaMemcpyToSymbol(shift, (void *) loaded_vec.data(), n_dim * sizeof(float)));
}

F4::~F4()
{
  /*empty*/
}

__global__ void computeK4(float * x, float * f){
  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  if( id_p < ps ){
    uint ndim = params.n_dim;
    uint id_d = id_p * ndim;
    float s = 0.0, z;
    for(uint i = 0; i < ndim; i++){
       z = x[id_d + i] - shift[i];
       s += (z * z) - 10.0 * cospi(2.0 * z) + 10.0;
    }
    if( s <= 10e-08 )
      s = 0.0f;
    f[id_p] = s;
  }
}

void F4::compute(float * x, float * f){
  computeK4<<< n_blocks, n_threads >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
