#include "jDE.cuh"

__constant__ __device__ float F_Lower = 0.10;
__constant__ __device__ float F_Upper = 0.90;
__constant__ __device__ float T = 0.10;

jDE::jDE( uint _s, uint _ndim ):
  NP(_s),
  n_dim(_ndim)
{
  checkCudaErrors(cudaMalloc((void **)&F,  NP * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&CR, NP * sizeof(float)));

  thrust::fill(thrust::device, F , F  + NP, 0.50);
  thrust::fill(thrust::device, CR, CR + NP, 0.90);
  config temp = { .x_min = -100.0, .x_max = +100.0 };
  temp.ps = NP;
  temp.n_dim = n_dim;
  checkCudaErrors(cudaMemcpyToSymbol(params, &temp, sizeof(config)));
}

jDE::~jDE()
{
  checkCudaErrors(cudaFree(F));
  checkCudaErrors(cudaFree(CR));
}

void jDE::update(){
  /* This function will call the update kernel (updateK) */
}

void jDE::run(){
  DE<<<1, 1>>>(NULL, NULL, NULL, NULL, NULL, NULL);
  cudaDeviceSynchronize();
}

/*
 * Update F and CR values accordly with jDE algorithm.
 *
 * F_Lower, F_Upper and T are constant variables declared in jDE.cuh
 */
__global__ void updateK(curandState * g_state, double * d_F, double * d_CR) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	curandState localState;
	localState = g_state[index];

	//(0, 1]
	double r1, r2, r3, r4;
	r1 = curand_uniform(&localState);
	r2 = curand_uniform(&localState);
	r3 = curand_uniform(&localState);
	r4 = curand_uniform(&localState);

	if (r2 < T)
		d_F[index] = F_Lower + (r1 * F_Upper);

	if (r4 < T)
		d_CR[index] = r3;

	g_state[index] = localState;
}

/*
 * Performs the selection step
 * In this case, aach thread is a individual
 * og -> Old genes, the previous generation offspring
 * ng -> New genes, the new generation offsprings
 * fog -> fitness of the old offspring
 * fng -> fitness of the new offspring
 * ndim -> number of dimensions used to copy the genes
 */
__global__ void selectionK(float * og, float * ng, float * fog, float * fng, uint n_dim){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
  if( fng[index] <= fog[index] ){
   memcpy(og + (n_dim * index), ng + (n_dim * index), n_dim * sizeof(float));
   fog[index] = fng[index];
 }
}

/*
 * Performs the DE/rand/1/bin operation
 * rng == global random state
 * fog == fitness of the old offspring
 * fng == fitness of the new offspring
 * F == mutation factor vector
 * CR == crossover probability vector
 */
__global__ void DE(curandState * rng, float * fog, float * fng, float * F, float * CR, uint * ind){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;

  printf("%d => %d %d %.2f %.2f\n", index, params.n_dim, params.ps, params.x_min, params.x_max);

  uint ps = params.ps;
  uint n_dim = params.n_dim;

  //uint n1, n2, n3, i;
  //n1 = ind[index];
  //n2 = ind[index + ps];
  //n3 = ind[index + ps + ps];

  //curandState random = rng[index];

  //float mF  = F[index];
  //float mCR = CR[index];

  //for( i = 0; i < n_dim; i++ ){
    //if( curand_uniform(&random) <= mCR || (i == n_dim - 1) ){
      /* empty for a while */
    //} else {
      /* empty for a while */
    //}
  //}

  //rng[index] = random;
}
