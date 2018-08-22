#include "jDE.cuh"

jDE::jDE( uint _s, uint _ndim, float _x_min, float _x_max ):
  NP(_s),
  n_dim(_ndim),
  x_min(_x_min),
  x_max(_x_max)
{
  checkCudaErrors(cudaMalloc((void **)&F,  NP * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&CR, NP * sizeof(float)));

  thrust::fill(thrust::device, F , F  + NP, 0.50);
  thrust::fill(thrust::device, CR, CR + NP, 0.90);

  Configuration conf;
  conf.x_min = x_min;
  conf.x_max = x_max;
  conf.ps = NP;
  conf.n_dim = n_dim;
  checkCudaErrors(cudaMemcpyToSymbol(params, &conf, sizeof(Configuration)));

  checkCudaErrors(cudaMalloc((void **)&rseq, NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&fseq, 3 * NP * sizeof(uint)));

  thrust::sequence(thrust::device, rseq, rseq + NP);

  checkCudaErrors(cudaMalloc((void **)&d_states, NP * sizeof(curandStateXORWOW_t)));

  std::random_device rd;
  unsigned int seed = rd();
  setup_kernel<<<2, 10>>>(d_states, seed);
  checkCudaErrors(cudaGetLastError());
}

jDE::~jDE()
{
  checkCudaErrors(cudaFree(F));
  checkCudaErrors(cudaFree(CR));
  checkCudaErrors(cudaFree(rseq));
  checkCudaErrors(cudaFree(fseq));
}

void jDE::update(){
  updateK<<<1, 20>>>(d_states, F, CR);
  checkCudaErrors(cudaGetLastError());
}

void jDE::run(){
  DE<<<1, 20>>>(d_states, NULL, NULL, F, CR, fseq);
  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();
}

void jDE::index_gen(){
  iGen<<<2,10>>>(d_states, rseq, fseq);
  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();
}

/*
 * Update F and CR values accordly with jDE algorithm.
 *
 * F_Lower, F_Upper and T are constant variables declared
 * on constants header
 */
__global__ void updateK(curandState * g_state, float * d_F, float * d_CR) {
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
__global__ void selectionK(float * og, float * ng, float * fog, float * fng){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
  if( fng[index] <= fog[index] ){
    uint ndim = params.n_dim;
    memcpy(og + (ndim * index), ng + (ndim * index), ndim * sizeof(float));
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
__global__ void DE(curandState * rng, float * fog, float * fng, float * F, float * CR, uint * fseq){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;

  if(index == 0){
    printf("%d => ndim(%d) ps(%d)\nx_min: %+.2f x_max: %+.2f\nF_Lower: %.2f F_Upper: %.2f T: %.2f\n",
      index, params.n_dim, params.ps,
      params.x_min, params.x_max,
      F_Lower, F_Upper, T
    );
  }

  uint ps = params.ps;
  uint n_dim = params.n_dim;

  uint n1, n2, n3, i;
  n1 = fseq[index];
  n2 = fseq[index + ps];
  n3 = fseq[index + ps + ps];

  float mF  = F[index];
  float mCR = CR[index];

  printf("[%-2d] %-2d | %-2d | %-2d F: %.2f CR: %.2f\n", index, n1, n2, n3, mF, mCR);
  curandState random = rng[index];

  for( i = 0; i < n_dim; i++ ){
    if( curand_uniform(&random) <= mCR || (i == n_dim - 1) ){
      /* empty for a while */
    } else {
      /* empty for a while */
    }
  }
  rng[index] = random;
}

/*
 * Generate 3 different indexs to DE/rand/1/bin.
 * @TODO:
 *  + rseq on constant memory;
 */
__global__ void iGen(curandState * g_state, uint * rseq, uint * fseq){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;

  curandState s = g_state[index];

  uint n1, n2, n3, ps = params.ps;

  n1 = curand(&s) % ps;
  if( rseq[n1] == index )
    n1 = (n1 + 1) % ps;

  n2 = ( curand(&s) % ((int)ps/3) ) + 1;
  if( rseq[(n1 + n2) % ps] == index )
    n2 = (n2 + 1) % ps;

  n3 = ( curand(&s) % ((int)ps/3) ) + 1;
  if( rseq[(n1 + n2 + n3) % ps] == index )
    n3 = (n3 + 1 ) % ps;

  fseq[index] = rseq[n1];
  fseq[index+ps] = rseq[(n1+n2)%ps];
  fseq[index+ps+ps] = rseq[(n1+n2+n3)%ps];

  g_state[index] = s;

  printf("[%-2d] %-2d | %-2d | %-2d\n", index, rseq[n1], rseq[(n1+n2)%ps], rseq[(n1+n2+n3)%ps]);
}

/* Each thread gets same seed, a different sequence number, no offset */
__global__ void setup_kernel(curandState * random, uint seed){
	uint index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < params.ps)
		curand_init(seed, index, 0, &random[index]);
}
