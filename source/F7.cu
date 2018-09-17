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

  file_name = "data-files/rot/M2_D" + std::to_string(n_dim) + ".txt";
  vec_name = "M2_D" + std::to_string(n_dim);
  file.open(file_name, std::ifstream::in);
  if( not file.is_open() ){
    std::cout << "Error opening rotation matrix file\n";
    exit(-1);
  }
  loaded_vec = io->load_vector<float>( vec_name, file ) ;
  file.close();
  /* ---------------------------------------------- */

  checkCudaErrors(cudaMalloc((void **)&M2, n_dim * n_dim * sizeof(float)));
  checkCudaErrors(cudaMemcpy(M2, (void*) loaded_vec.data(), n_dim*n_dim*sizeof(float), cudaMemcpyHostToDevice));

}

F7::~F7()
{
  /*empty*/
  checkCudaErrors(cudaFree(M2));
}

__global__ void computeK_F7(float * x, float * f, float * M2){
  const float alpha = 10.0;
  const float beta  = 0.2;
  const float c     = 0.0512;

  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  uint ndim = params.n_dim;
  int i, j;


  __shared__ float s_M2[10000];

  if( threadIdx.x == 0 ){
    for( i = 0; i < ndim; i++ )
      for( j = 0; j < ndim; j++ )
        s_M2[i*ndim+j] = M2[i*ndim+j];
  }

  __syncthreads();

  if( id_p < ps ){
    uint id_d = id_p * ndim;
    int k = ndim - 1;

    float z[100];
    float z_rot[100];
    float c1, c2, xx;
    int sx;

    // shift func and multiply by 5.12/100
    for( i = 0; i < ndim; i++ )
      z[i] = (x[id_d + i] - shift[i]) * c;

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

        z_rot[i] = sx*expf(xx+0.049*(sinf(c1*xx)+sinf(c2*xx)));
      } else {
        /* nothing to do */
      }
    }

    for( i = 0; i < ndim; i++ ){
      // asyfunc
      if( z_rot[i] > 0.0 ){
        z_rot[i] = powf(z_rot[i], 1.0 + beta * i / k * powf(z_rot[i], 0.5));
      }

    }

    // rotate func (2) (second rotation matrix)
    for( i = 0; i < ndim; i++){
      z[i] = 0.0;
      for( j = 0; j < ndim; j++ ){
        z[i] += z_rot[j] * s_M2[i * ndim + j];
      }
    }

    // pow(alpha, 1.0*i/(ndim - 1)/2);
    for( i = 0; i < ndim; i++ ){
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

void F7::compute(float * x, float * f){
  computeK_F7<<< n_blocks, n_threads >>>(x, f, M2);
  checkCudaErrors(cudaGetLastError());
}
