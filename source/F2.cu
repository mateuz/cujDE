#include "F2.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F2::F2(uint _dim, uint _ps):Benchmarks()
{
  n_dim = _dim;
  ps = _ps;
  min = -100.0;
  max = +100.0;
  ID = 2;

  n_threads = 32;
  n_blocks = (ps%n_threads)? (ps/n_threads)+1 : (ps/n_threads);

  /* ---------------------------------------------- */
  /* Load a shift vector to test the bench function */
  std::string file_name = "data-files/shift_rosenbrock.mat";
  std::string vec_name = "Shift - Rosenbrock [-100.0, +100.0]";
  IO * io = new IO();
  std::ifstream file(file_name);
  if( not file.is_open() ){
    std::cout << "\"data-files/shift_rosenbrock.mat\" could not be opened\n";
    exit(-1);
  }
  auto loaded_vec = io->load_vector<float>( vec_name, file ) ;
  file.close();
  /* ---------------------------------------------- */

  checkCudaErrors(cudaMemcpyToSymbol(shift, (void *) loaded_vec.data(), n_dim * sizeof(float)));
}

F2::~F2()
{
  /*empty*/
}


__global__ void computeK_F2_2(float * x, float * f){
  uint id_p, id_d, ndim, stride;

  id_p = blockIdx.x;
  id_d = threadIdx.x;
  ndim = params.n_dim;
  stride = id_p * ndim;

  float a, b, t1, t2;

  __shared__ float r[128];
  __shared__ float z[100];

  r[id_d] = 0.0f;

  if( id_d < ndim ){
    z[id_d] = x[stride + id_d] - shift[id_d] + 1.0f;
  }

  __syncthreads();

  if( id_d < (ndim-1) ){
    a = z[id_d];
    b = z[id_d+1];

    t1 = b - (a * a);
    t2 = a - 1.0;

    t1 *= t1;
    t2 *= t2;

    r[id_d] = (100.0 * t1) + t2;

    __syncthreads();

    /* Simple reduce sum */
    if( id_d < 64 )
      r[id_d] += r[id_d + 64];

    __syncthreads();

    if( id_d < 32 )
      r[id_d] += r[id_d + 32];

    __syncthreads();

    if( id_d < 16 )
      r[id_d] += r[id_d + 16];

    __syncthreads();

    if( id_d < 8 )
      r[id_d] += r[id_d + 8];

    __syncthreads();

    if( id_d < 4 )
      r[id_d] += r[id_d + 4];

    __syncthreads();

    if( id_d < 2 )
      r[id_d] += r[id_d + 2];

    __syncthreads();

    if( id_d == 0 )
      r[id_d] += r[id_d + 1];

    __syncthreads();

    if( id_d == 0 )
      f[id_p] = r[0];
  }
}

__global__ void computeK2(float * x, float * f){
  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  if( id_p < ps ){
    uint ndim = params.n_dim;
    uint id_d = id_p * ndim;
    float res = 0.0, a, b, t;
    for(uint i = 0; i < (ndim - 1); i++){
      a = x[id_d + i] - shift[i] + 1.00;
      b = x[id_d + i + 1] - shift[i + 1] + 1.00;

      t = (b - (a * a));
      res += (100.0 * t * t);
      t = (a - 1.00);
      res += (t * t);
    }
    if( res <= 10e-08 )
      res = 0.0f;

    f[id_p] = res;
  }
}

void F2::compute(float * x, float * f){
  //computeK2<<< n_blocks, n_threads >>>(x, f);
  computeK_F2_2<<< ps, 128 >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
