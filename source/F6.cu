#include "F6.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

/*
 * Shifted and Rotated Griewank's Function
 *
 * as defined in "Problem Definitions and Evaluation Criteria for the
 * CEC 2013 Special Session and Competition on Real-Parameter Optimization",
 * by Liang, J.J., Qu, B.-Y., Suganthan, P.N., Hernandez-Diaz, A.G.,
 * Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou,
 * China and Nanyang Technological University, Singapore, Technical Report,
 * v. 2012, p. 3-18, 2013.
*/

F6::F6(uint _dim, uint _ps):Benchmarks()
{
  n_dim = _dim;
  ps = _ps;
  min = -100.0;
  max = +100.0;
  ID = 6;

  n_threads = 32;
  n_blocks = (ps%n_threads)? (ps/n_threads)+1 : (ps/n_threads);

  /* ---------------------------------------------- */
  /* Load a shift vector to test the bench function */
  std::string file_name = "data-files/shift_griewank.mat";
  std::string vec_name = "Shift - Griewank";
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

  /* ---------------------------------------------- */
  /* Load a rotate matrix                           */
  if(!(n_dim==2 or n_dim==5 or n_dim==10 or n_dim==20 or n_dim==30 or n_dim==50 or n_dim==100)){
    printf("\nError: Rotation matrix are only defined for D = 2,5,10,20,30,50,100.\n");
    exit(-1);
  }

  file_name = "data-files/rot/M_D" + std::to_string(n_dim) + ".txt";
  vec_name = "M_D" + std::to_string(n_dim);
  file.open(file_name, std::ifstream::in);
  if( not file.is_open() ){
    std::cout << "Error opening rotation matrix file\n";
    exit(-1);
  }
  loaded_vec = io->load_vector<float>( vec_name, file ) ;
  file.close();
  /* ---------------------------------------------- */

  checkCudaErrors(cudaMemcpyToSymbol(m_rotation, (void *) loaded_vec.data(), n_dim * n_dim * sizeof(float)));

}

F6::~F6()
{
  /*empty*/
}

__global__ void computeK_F6(float * x, float * f){
  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  if( id_p < ps ){
    uint ndim = params.n_dim;
    uint id_d = id_p * ndim;
    uint i, j, k = ndim - 1;

    //The constant 600/100 is needed because on rotate operation
    //the value of a dimension can be higher than bounds;

    float z[100];
    //shift
    for( i = 0; i < ndim; i++ )
      z[i] = (x[id_d + i] - shift[i]) * 6.0;

    float z_rot[100];
    //rotation
    for( i = 0; i < ndim; i++ ){
      z_rot[i] = 0.0;
      for( j = 0; j < ndim; j++ )
        z_rot[i] += z[j] * m_rotation[i * ndim + j];
    }

    float s = 0.0, p = 1.0;
    for(uint i = 0; i < ndim; i++){
      z[i] = z_rot[i] * __powf(100.0, 1.0*i/k/2.0);

      s += z[i] * z[i];
      p *= __cosf( z[i] / __fsqrt_rn(1.0+i) );
    }
    s = 1.0 + s/4000.0 - p;

    if( s <= 10e-08 )
      s = 0.0;

    f[id_p] = s;
  }
}

__global__ void computeK2_F6(float * x, float * f){
  uint id_p, id_d, ndim, i, stride;

  id_p = blockIdx.x;
  id_d = threadIdx.x;
  //ps = params.ps;
  ndim = params.n_dim;
  stride = id_p * ndim;

  __shared__ float s[128];
  __shared__ float p[128];
  __shared__ float z[100];
  __shared__ float R[10000];
  __shared__ float z_rot[100];

  s[id_d] = 0.0f;
  p[id_d] = 1.0f;

  //load the z and R
  if( id_d < ndim ){
    z[id_d] = (x[stride+id_d] - shift[id_d]) * 6.0;
    //each dimension load your rotation column from rotation matrix
    for( i = 0; i < ndim; i++ ){
      R[(id_d*ndim)+i] = m_rotation[(id_d*ndim)+i];
    }
  }

  __syncthreads();

  if( id_d < ndim ){
    //apply rotation
    z_rot[id_d] = 0.0;
    for( i = 0; i < ndim; i++ ){
      z_rot[id_d] += z[i] * R[(id_d*ndim)+i];
    }

    __syncthreads();

    z[id_d] = z_rot[id_d] * __powf(100.0, 1.0*id_d/(ndim-1)/2.0);

    s[id_d] = (z[id_d]*z[id_d]);
    p[id_d] = __cosf( z[id_d] / __fsqrt_rn(1.0+id_d) );
  }

  __syncthreads();

  /*
  if( id_d == 0 ){
    float sum = 0.0, prod = 1.0;
    #pragma unroll
    for( i = 0; i < ndim; i++ ){
      prod *= p[i];
      sum  += s[i];
    }
    f[id_p] = 1.0 + sum/4000.0 - prod;
  }*/

  __syncthreads();

  /* Simple reduce sum */
  if( id_d < 64 && ndim == 100 ){
    s[id_d] += s[id_d + 64];
    p[id_d] *= p[id_d + 64];
  }

  __syncthreads();

  if( id_d < 32 ){
    s[id_d] += s[id_d + 32];
    p[id_d] *= p[id_d + 32];
  }

  __syncthreads();

  if( id_d < 16 ){
    s[id_d] += s[id_d + 16];
    p[id_d] *= p[id_d + 16];
  }

  __syncthreads();

  if( id_d < 8 ){
    s[id_d] += s[id_d + 8];
    p[id_d] *= p[id_d + 8];
  }

  __syncthreads();

  if( id_d < 4 ){
    s[id_d] += s[id_d + 4];
    p[id_d] *= p[id_d + 4];
  }

  __syncthreads();

  if( id_d < 2 ){
    s[id_d] += s[id_d + 2];
    p[id_d] *= p[id_d + 2];
  }

  __syncthreads();

  if( id_d == 0 ){
    s[id_d] += s[id_d + 1];
    p[id_d] *= p[id_d + 1];
  }

  __syncthreads();

  if( id_d == 0 ){
    f[id_p] = 1.0 + s[0]/4000.0 - p[0];
  }


}

void F6::compute(float * x, float * f){
  //computeK_F6<<< n_blocks, n_threads >>>(x, f);
  computeK2_F6<<< ps, 128 >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
