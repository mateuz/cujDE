#include "F1.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F1::F1(uint _dim, uint _ps):Benchmarks()
{
  n_dim = _dim;
  ps = _ps;
  min = -100.0;
  max = +100.0;
  ID = 1;

  n_threads = 32;
  n_blocks = (ps%n_threads)? (ps/n_threads)+1 : (ps/n_threads);

  std::string file_name = "data-files/shift_sphere.mat";
  std::string vec_name = "Shift - Sphere [-100.0, +100.0]";
  IO * io = new IO();
  std::ifstream file(file_name);
  if(not file.is_open()){
    std::cout << "\"data-files/shift_sphere.mat\" could not be opened\n";
    exit(-1);
  }
  auto loaded_vec = io->load_vector<float>( vec_name, file ) ;
  file.close();

  checkCudaErrors(cudaMemcpyToSymbol(shift, (void *) loaded_vec.data(), n_dim * sizeof(float)));
}

F1::~F1()
{
  /* empty */
}

__global__ void computeK(float * x, float * f){
  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  if( id_p < ps ){
    uint ndim = params.n_dim;
    uint id_d = id_p * ndim;
    float res = 0.0, a;
    for(uint i = 0; i < ndim; i++){
      a = x[id_d + i] - shift[i];
      res += (a * a);
    }
    //printf("%u => %.2f\n", id_p, res);
    f[id_p] = res;
  }
}

void F1::compute(float * x, float * f){
  computeK<<< n_blocks, n_threads >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
