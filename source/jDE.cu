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

void jDE::index_gen(){
  thrust::device_vector<uint> rseq(NP);
  thrust::device_vector<uint> fseq(NP * 3);
  thrust::sequence(thrust::device, rseq.begin(), rseq.end());

  curandState * d_states;
  checkCudaErrors(cudaMalloc((void **)&d_states, NP * sizeof(curandStateXORWOW_t)));
  setup_kernel<<<2, 10>>>(d_states);
  checkCudaErrors(cudaGetLastError());
  iGen<<<2,10>>>(d_states, thrust::raw_pointer_cast(&rseq[0]), thrust::raw_pointer_cast(&fseq[0]));
  checkCudaErrors(cudaGetLastError());
}

/*
 * Update F and CR values accordly with jDE algorithm.
 *
 * F_Lower, F_Upper and T are constant variables declared on constants header
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

  printf("%d => ndim(%d) ps(%d)\nx_min: %+.2f x_max: %+.2f\nF_Lower: %.2f F_Upper: %.2f T: %.2f\n",
    index, params.n_dim, params.ps,
    params.x_min, params.x_max,
    F_Lower, F_Upper, T
  );

  /*
  uint ps = params.ps;
  uint n_dim = params.n_dim;

  uint n1, n2, n3, i;
  n1 = ind[index];
  n2 = ind[index + ps];
  n3 = ind[index + ps + ps];

  curandState random = rng[index];

  float mF  = F[index];
  float mCR = CR[index];

  for( i = 0; i < n_dim; i++ ){
    if( curand_uniform(&random) <= mCR || (i == n_dim - 1) ){
      /* empty for a while * /
    } else {
      /* empty for a while * /
    }
  }

  rng[index] = random;
  */
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

  printf("[%-2.d] %-2.d | %-2.d | %-2.d n(%-2.d %-2.d %-2.d)\n", index, rseq[n1], rseq[(n1+n2)%ps], rseq[(n1+n2+n3)%ps], n1, n2, n3);
}

/* Each thread gets same seed, a different sequence number, no offset */
__global__ void setup_kernel(curandState * random){
	uint index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < params.ps)
		curand_init(1234, index, 0, &random[index]);
}
