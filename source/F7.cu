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
  std::string vec_name = "Shift - Griewank";
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
  /* future implementation */
}

void F7::compute(float * x, float * f){
  computeK_F7<<< n_blocks, n_threads >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
