#include "2D_AB.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F2DAB::F2DAB( uint _dim, uint _ps ):Benchmarks()
{
  //protein size
  n_dim = _dim;

  //number of individuals
  ps = _ps;

  min = -3.1415926535897932384626433832795029;
  max = +3.1415926535897932384626433832795029;

  ID = 1001;

  // get the next multiple of 32;
  n_threads = 32 * ceil((double) n_dim / 32.0);

  //one block per population member
  n_blocks = ps;

  // printf("nb: %d e nt: %d\n", n_blocks, n_threads);

  char s_2dab[60];
  memset(s_2dab, 0, sizeof(char) * 60);

  if( n_dim == 13 ){
    strcpy(s_2dab, "ABBABBABABBAB");
  } else if( n_dim == 21 ){
    strcpy(s_2dab, "BABABBABABBABBABABBAB");
  } else if( n_dim == 34 ){
    strcpy(s_2dab, "ABBABBABABBABBABABBABABBABBABABBAB");
  } else if( n_dim == 55 ){
    strcpy(s_2dab, "BABABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBAB");
  } else {
    std::cout << "error string size must be 13, 21, 34 or 55.\n";
    exit(-1);
  }

  checkCudaErrors(cudaMemcpyToSymbol(S_2DAB, (void *) s_2dab, 60 * sizeof(char)));
}

F2DAB::~F2DAB()
{
  /* empty */
}

__device__ float _C( uint i, uint j ){
  float c;

  if( S_2DAB[i] == 'A' && S_2DAB[j] == 'A' )
    c = 1.0;
  else if( S_2DAB[i] == 'B' && S_2DAB[j] == 'B' )
    c = 0.5;
  else
    c = -0.5;

  return c;
}

__global__ void computeK_2DAB(float * x, float * f){
  uint id_p = blockIdx.x;
  uint id_d = threadIdx.x;
  uint ndim = params.n_dim;

  uint stride = id_p * ndim;

  // if( id_p == 0 && id_d == 0 ){
  //   printf("Otimizando a string: %s\n", S_2DAB);
  //   printf("Nº de dimensões: %d\n", params.n_dim);
  //   printf("Nº de Indivíduos: %d\n", params.ps);
  //   printf("x in [%.3f, %.3f]\n", params.x_min, params.x_max);
  // }

  __shared__ AB_2D amino_acid[55];

  double d_x, d_y;

  if( id_d == 0 ){
    // printf("STRIDE for block %d is %d\n", id_p, stride);
    amino_acid[0].x = 0.0;
    amino_acid[0].y = 0.0;

    amino_acid[1].x = 1.0;
    amino_acid[1].y = 0.0;

    for( int i = 1; i < (ndim - 1); i++ ){
      d_x = amino_acid[i].x - amino_acid[i-1].x;
      d_y = amino_acid[i].y - amino_acid[i-1].y;

      amino_acid[i+1].x = amino_acid[i].x + d_x * cosf( x[stride + i - 1]) - d_y * sinf( x[stride + i - 1] );
      amino_acid[i+1].y = amino_acid[i].y + d_y * cosf( x[stride + i - 1]) + d_x * sinf( x[stride + i - 1] );
    }
  }

  __shared__ float v1[64], v2[64];

  v1[id_d] = 0.0;
  v2[id_d] = 0.0;

  __syncthreads();

  float C, D;
  if( id_d < (ndim - 2) ){
    v1[id_d] = (1.0 - cosf(x[stride + id_d])) / 4.0f;

    for( uint j = (id_d+2); j < ndim; j++ ){
      C = _C(id_d, j);

      d_x = amino_acid[id_d].x - amino_acid[j].x;
      d_y = amino_acid[id_d].y - amino_acid[j].y;

      D = sqrtf( (d_x * d_x) + (d_y * d_y) );
      v2[id_d] += 4.0 * ( 1/powf(D, 12) - C/powf(D, 6) );
    }
  }

  // Just apply a simple reduce sum to get the v1 and v2 sum

  if( id_d < 32 && ndim > 32 ){
    v1[id_d] += v1[id_d + 32];
    v2[id_d] += v2[id_d + 32];
  }

  __syncthreads();

  if( id_d < 16 ){
    v1[id_d] += v1[id_d + 16];
    v2[id_d] += v2[id_d + 16];
  }

  __syncthreads();

  if( id_d < 8 ){
    v1[id_d] += v1[id_d + 8];
    v2[id_d] += v2[id_d + 8];
  }

  __syncthreads();

  if( id_d < 4 ){
    v1[id_d] += v1[id_d + 4];
    v2[id_d] += v2[id_d + 4];
  }

  __syncthreads();

  if( id_d < 2 ){
    v1[id_d] += v1[id_d + 2];
    v2[id_d] += v2[id_d + 2];
  }

  __syncthreads();

  if( id_d == 0 ){
    v1[id_d] += v1[id_d + 1];
    v2[id_d] += v2[id_d + 1];

    f[id_p] = v1[0] + v2[0];
    // printf("[%d] %.3lf from %.3lf and %.3lf\n", id_p, f[id_p], v1[0], v2[0]);
  }


}

void F2DAB::compute(float * x, float * f){
  computeK_2DAB<<< n_blocks, n_threads >>>(x, f);
  // cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
