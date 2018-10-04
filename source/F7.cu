#include "F7.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

/*
 * Shifted and Rotated Rastrigin's Function
 *
 * as defined in "Problem Definitions and Evaluation Criteria for the
 * CEC 2013 Special Session and Competition on Real-Parameter Optimization",
 * by Liang, J.J., Qu, B.-Y., Suganthan, P.N., Hernandez-Diaz, A.G.,
 * Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou,
 * China and Nanyang Technological University, Singapore, Technical Report,
 * v. 2012, p. 3-18, 2013.
*/

F7::F7(uint _dim, uint _ps):Benchmarks()
{
  n_dim = _dim;
  ps = _ps;
  min = -100.0;
  max = +100.0;
  ID = 7;

  n_threads = 32;
  n_blocks = (ps%n_threads)? (ps/n_threads)+1 : (ps/n_threads);

  /* ---------------------------------------------- */
  /* Load a shift vector to test the bench function */
  std::string file_name = "data-files/shift_rastrigin.mat";
  std::string vec_name = "Shift - Rastrigin";
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

F7::~F7()
{
  /*empty*/
}

__global__ void computeK_F7(float * x, float * f){
  const float alpha = 10.0;
  const float beta  = 0.2;

  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  uint ndim = params.n_dim;
  int i, j;

  if( id_p < ps ){
    uint id_d = id_p * ndim;
    int k = ndim - 1;

    float z[100];
    float z_rot[100];
    float xosz[100];
    float c1, c2, xx;
    int sx;

    // shift func and multiply by 5.12/100
    for( i = 0; i < ndim; i++ )
      z[i] = (x[id_d + i] - shift[i]) * 0.0512;

    // rotatefunc (1)
    for( i = 0; i < ndim; i++ ){
      z_rot[i] = 0.0;
      for( j = 0; j < ndim; j++ )
        z_rot[i] += z[j] * m_rotation[i * ndim + j];
    }

    // oszfunc
    for( i = 0; i < ndim; i++ ){
      if( i == 0 || i == k ){
        if( z_rot[i] != 0 )
          xx = logf( fabsf(z_rot[i]) );

        if( z_rot[i] > 0.0 ){
          c1 = 10.0;
          c2 = 7.9;
        } else {
          c1 = 5.5;
          c2 = 3.1;
        }

        if( z_rot[i] > 0 )
          sx = 1;
        else if( z_rot[i] == 0 )
          sx = 0;
        else
          sx = -1;

        xosz[i] = sx*expf(xx+0.049*(sinf(c1*xx)+sinf(c2*xx)));
      } else {
        xosz[i] = z_rot[i];
      }

      //asyfunc
      if( xosz[i] > 0.0 )
        z_rot[i] = powf(xosz[i], 1.0 + beta * i / k * powf(xosz[i], 0.5));
    }

    // rotate func (2) (second rotation matrix)
    for( i = 0; i < ndim; i++){
      z[i] = 0.0;
      for( j = 0; j < ndim; j++ ){
        z[i] += z_rot[j] * m_rotation[i * ndim + j];
      }
      //pow(alpha, 1.0*i/(ndim - 1)/2);
      z[i] *= powf(alpha, 1.0*i/k/2);
    }

    // rotate func (3)
    for( i = 0; i < ndim; i++ ){
      z_rot[i] = 0.0;
      for( j = 0; j < ndim; j++ )
        z_rot[i] += z[j] * m_rotation[i * ndim + j];
    }

    float s = 0.0, p, u;
    // evaluation
    for( i = 0; i < ndim; i++ ){
      p = z_rot[i] * z_rot[i];
      u = cospif(2.0 * z_rot[i]);
      s += p - 10.0 * u + 10.0;
    }

    f[id_p] = s;
  }
}


__global__ void computeK2_F7(float * x, float * f){
  const float alpha = 10.0;
  const float beta  = 0.2;

  uint id_p, id_d, ndim, i, j, stride;

  id_p = blockIdx.x;
  id_d = threadIdx.x;
  //ps = params.ps;
  ndim = params.n_dim;
  stride = id_p * ndim;

  __shared__ float r[128];
  __shared__ float z[100];
  __shared__ float R[10000];
  __shared__ float z_rot[100];
  __shared__ float xosz[100];

  r[id_d] = 0.0;

  //every dimension load your value to shared memory
  if( id_d < ndim ){
    z[id_d] = (x[stride+id_d] - shift[id_d]) * 0.0512;
    //each dimension load your rotation column from rotation matrix
    for( i = 0; i < ndim; i++ ){
      R[(id_d*ndim)+i] = m_rotation[(id_d*ndim)+i];
    }
  }

  __syncthreads();

  // rotate func 1
  if( id_d < ndim ){
    float c1, c2, xx;
    int sx;

    //rotate 1
    z_rot[id_d] = 0.0;
    for( j = 0; j < ndim; j++ )
      z_rot[id_d] += z[j] * R[id_d * ndim + j];

    __syncthreads();

    if( id_d == 0 || id_d == (ndim-1) ){
      if( z_rot[id_d] != 0 )
        xx = logf( fabsf(z_rot[id_d]) );

      if( z_rot[id_d] > 0.0 ){
        c1 = 10.0;
        c2 = 7.9;
      } else {
        c1 = 5.5;
        c2 = 3.1;
      }

      if( z_rot[id_d] > 0 )
        sx = 1;
      else if( z_rot[id_d] == 0 )
        sx = 0;
      else
        sx = -1;

      xosz[id_d] = sx*expf(xx+0.049*(sinf(c1*xx)+sinf(c2*xx)));
    } else {
      xosz[id_d] = z_rot[id_d];
    }

    __syncthreads();
    //asyfunc
    if( xosz[id_d] > 0.0 )
      z_rot[id_d] = powf(xosz[id_d], 1.0+beta*id_d/(ndim-1)*powf(xosz[id_d], 0.5));

    __syncthreads();

    //rotate 2
    z[id_d] = 0.0;
    for( j = 0; j < ndim; j++ ){
      z[id_d] += z_rot[j] * R[id_d * ndim + j];
    }

    //pow(alpha, 1.0*i/(ndim - 1)/2);
    z[id_d] *= powf(alpha, 1.0*id_d/(ndim-1)/2);

    __syncthreads();

    //rotate 3
    z_rot[id_d] = 0.0;
    for( j = 0; j < ndim; j++ )
      z_rot[id_d] += z[j] * R[id_d * ndim + j];
  }

  __syncthreads();

  if( id_d < ndim ){
    float p, u;

    // evaluation
    p = z_rot[id_d];
    u = cospif(2.0*p);
    p *= p;

    r[id_d] = p - 10.0 * u + 10.0;

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

void F7::compute(float * x, float * f){
  //computeK_F7<<< n_blocks, n_threads >>>(x, f);
  computeK2_F7<<< ps, 128 >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
