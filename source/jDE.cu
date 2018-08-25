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
  thrust::fill(thrust::device, CR, CR + NP, 0.30);

  Configuration conf;
  conf.x_min = x_min;
  conf.x_max = x_max;
  conf.ps = NP;
  conf.n_dim = n_dim;

  checkCudaErrors(cudaMemcpyToSymbol(params, &conf, sizeof(Configuration)));
  checkCudaErrors(cudaMalloc((void **)&rseq, NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&fseq, 3 * NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_states, NP * sizeof(curandStateXORWOW_t)));
  thrust::sequence(thrust::device, rseq, rseq + NP);

  n_threads = 32;
  n_blocks = iDivUp(NP, n_threads);

  //printf("nThreads(): %u nBlocks(): %u\n", nt, nb);

  std::random_device rd;
  unsigned int seed = rd();
  setup_kernel<<<n_blocks, n_threads>>>(d_states, seed);
  checkCudaErrors(cudaGetLastError());
}

jDE::~jDE()
{
  checkCudaErrors(cudaFree(F));
  checkCudaErrors(cudaFree(CR));
  checkCudaErrors(cudaFree(rseq));
  checkCudaErrors(cudaFree(fseq));
}

uint jDE::iDivUp(uint a, uint b)
{
  return (a%b)? (a/b)+1 : a/b;
}

void jDE::update(){
  updateK<<<n_blocks, n_threads>>>(d_states, F, CR);
  checkCudaErrors(cudaGetLastError());
}

/*
 * fog == fitness of the old offspring
 * fng == fitness of the new offspring
 */
void jDE::run(float * og, float * ng){
  DE<<<n_blocks, n_threads>>>(d_states, og, ng, F, CR, fseq);
  checkCudaErrors(cudaGetLastError());
}

void jDE::index_gen(){
  iGen<<<n_blocks, n_threads>>>(d_states, rseq, fseq);
  checkCudaErrors(cudaGetLastError());
}

void jDE::selection(float * og, float * ng, float * fog, float * fng){
  selectionK<<<n_blocks, n_threads>>>(og, ng, fog, fng);
  checkCudaErrors(cudaGetLastError());
}

/*
 * Update F and CR values accordly with jDE algorithm.
 *
 * F_Lower, F_Upper and T are constant variables declared
 * on constants header
 */
__global__ void updateK(curandState * g_state, float * d_F, float * d_CR) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

  uint ps = params.ps;

  if( index < ps ){
  	curandState localState;
  	localState = g_state[index];

  	//(0, 1]
  	float r1, r2, r3, r4;
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
}

/*
 * Performs the selection step
 * In this case, each thread is a individual
 * og -> Old genes, the previous generation offspring
 * ng -> New genes, the new generation offsprings
 * fog -> fitness of the old offspring
 * fng -> fitness of the new offspring
 * ndim -> number of dimensions used to copy the genes
 */
__global__ void selectionK(float * og, float * ng, float * fog, float * fng){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
  uint ps = params.ps;

  if( index < ps ){
    if( fng[index] <= fog[index] ){
      uint ndim = params.n_dim;
      memcpy(og + (ndim * index), ng + (ndim * index), ndim * sizeof(float));
      fog[index] = fng[index];
   }
  }
}

/*
 * Performs the DE/rand/1/bin operation
 * 1 thread == 1 individual
 * rng == global random state
 * fog == fitness of the old offspring
 * fng == fitness of the new offspring
 * F == mutation factor vector
 * CR == crossover probability vector
 */
__global__ void DE(curandState * rng, float * og, float * ng, float * F, float * CR, uint * fseq){
  uint i, index, ps, n_dim;
  index = threadIdx.x + blockDim.x * blockIdx.x;
  ps = params.ps;

  if(index < ps){
    uint n1, n2, n3, p1, p2, p3, p4;
    n_dim = params.n_dim;

    float mF  = F[index];
    float mCR = CR[index];

    curandState random = rng[index];

    n1 = fseq[index];
    n2 = fseq[index + ps];
    n3 = fseq[index + ps + ps];

    //do n1 = curand(&random)%ps; while (n1 == index);
    //do n2 = curand(&random)%ps; while (n2 == index || n2 == n1 );
    //do n3 = curand(&random)%ps; while (n3 == index || n3 == n1 || n3 == n2);

    p1 = index * n_dim;
    p2 = n3 * n_dim;
    p3 = n2 * n_dim;
    p4 = n1 * n_dim;
    //printf("[%u] %u %u %u => %u %u %u %u\n", index, n1, n2, n3, p4, p3, p2, p1);
    for( i = 0; i < n_dim; i++ ){
      if( curand_uniform(&random) <= mCR || (i == n_dim - 1) ){
        /* Get three mutually different indexs */
        ng[p1 + i] = og[p2 + i] + mF * (og[p3 + i] - og[p4 + i]);

        /* Check bounds */
        ng[p1 + i] = max(params.x_min, ng[p1 + i]);
        ng[p1 + i] = min(params.x_max, ng[p1 + i]);
      } else {
        ng[p1 + i] = og[p1 + i];
      }
    }
    rng[index] = random;
  }
}

/*
 * Generate 3 different indexs to DE/rand/1/bin.
 * @TODO:
 *  + rseq on constant memory;
 */
__global__ void iGen(curandState * g_state, uint * rseq, uint * fseq){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;

  uint ps = params.ps;
  if( index < ps ){
    curandState s = g_state[index];

    uint n1, n2, n3;

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
    //printf("[%-3d] %-3d | %-3d | %-3d\n", index, rseq[n1], rseq[(n1+n2)%ps], rseq[(n1+n2+n3)%ps]);
  }
}

/* Each thread gets same seed, a different sequence number, no offset */
__global__ void setup_kernel(curandState * random, uint seed){
	uint index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < params.ps)
		curand_init(seed, index, 0, &random[index]);
}
