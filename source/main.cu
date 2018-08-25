/* C++ includes */

#include <iostream>
#include <chrono>
#include <functional>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

/* C includes */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "Benchmarks.cuh"
#include "F1.cuh"
#include "F2.cuh"
#include "F3.cuh"
#include "F4.cuh"
#include "jDE.cuh"

struct prg
{
  float a, b;

  __host__ __device__ prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

  __host__ __device__ float operator()(const unsigned int n) const
  {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);

    return dist(rng);
  }
};

void show_params(
	uint n_runs,
	uint NP,
	uint n_evals,
	uint n_dim,
	std::string FuncObj
){
	printf(" | Number of Executions:                    %d\n", n_runs);
	printf(" | Population Size:                         %d\n", NP);
	printf(" | Number of Dimensions:                    %d\n", n_dim);
	printf(" | Number of Function Evaluations:          %d\n", n_evals);
	printf(" | Optimization Function:                   %s\n", FuncObj.c_str());
	printf(" +==============================================================+ \n");
	printf(" | Number of Threads                        %d\n", 32);
	printf(" | Number of Blocks                         %d\n", (NP%32)? (NP/32)+1 : NP/32);
}

std::string toString(uint id){
  switch( id ){
    case 1:
      return "Shifted Sphere";
    case 2:
      return "Shifted Rosenbrock";
    case 3:
      return "Shifted Griewank";
    case 4:
      return "Shifted Rastringin";
    default:
      return "Unknown";
  }
}

Benchmarks * getFunction(uint id, uint n_dim, uint ps){
  Benchmarks * n;

  if( id == 1 ){
    n = new F1(n_dim, ps);
    return n;
  }

  if( id == 2 ){
    n = new F2(n_dim, ps);
    return n;
  }

  if( id == 3 ){
    n = new F3(n_dim, ps);
    return n;
  }

  if( id == 4 ){
    n = new F4(n_dim, ps);
    return n;
  }

  return NULL;
}

int main(int argc, char * argv[]){
	srand(time(NULL));
	uint n_runs, NP, n_evals, n_dim, f_id;

	try{
		po::options_description config("Opções");
		config.add_options()
			("runs,r"    , po::value<uint>(&n_runs)->default_value(1)    , "Number of Executions"          )
			("pop_size,p", po::value<uint>(&NP)->default_value(20)       , "Population Size"               )
			("dim,d"     , po::value<uint>(&n_dim)->default_value(10)    , "Number of Dimensions"          )
			("func_obj,o", po::value<uint>(&f_id)->default_value(1)      , "Function to Optimize"          )
			("max_eval,e", po::value<uint>(&n_evals)->default_value(10e5), "Number of Function Evaluations")
			("help,h", "Mostrar texto de Ajuda");

		po::options_description cmdline_options;
		cmdline_options.add(config);
		po::variables_map vm;
		store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
		notify(vm);
		if (vm.count("help")) {
			std::cout << cmdline_options << "\n";
			return 0;
		}
	}catch(std::exception& e){
		std::cout << e.what() << "\n";
		return 1;
	}

	printf(" +==============================================================+ \n");
	printf(" |                      EXECUTION PARAMETERS                    | \n");
	printf(" +==============================================================+ \n");
	show_params(n_runs, NP, n_evals, n_dim, toString(f_id));
	printf(" +==============================================================+ \n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	thrust::device_vector<float> d_og(n_dim * NP);
	thrust::device_vector<float> d_ng(n_dim * NP);
	thrust::device_vector<float> d_fog(NP, 0.0);
	thrust::device_vector<float> d_fng(NP, 0.0);

	thrust::host_vector<float> h_og(n_dim * NP);
	thrust::host_vector<float> h_ng(n_dim * NP);
	thrust::host_vector<float> h_fog(NP);
	thrust::host_vector<float> h_fng(NP);

	float * p_og  = thrust::raw_pointer_cast(d_og.data());
	float * p_ng  = thrust::raw_pointer_cast(d_ng.data());
	float * p_fog = thrust::raw_pointer_cast(d_fog.data());
	float * p_fng = thrust::raw_pointer_cast(d_fng.data());

	thrust::device_vector<float>::iterator it;

	Benchmarks * B = NULL;
  B = getFunction(f_id, n_dim, NP);

  if( B == NULL ){
     printf("Unknown function! Exiting...\n");
     exit(EXIT_FAILURE);
  }
	float x_min = B->getMin();
	float x_max = B->getMax();

	float time  = 0.00;

	std::vector< std::pair<float, float> > stats;
	for( int i = 1; i <= n_runs; i++ ){
    jDE * jde = new jDE(NP, n_dim, x_min, x_max);

    cudaEventRecord(start);

    // Randomly initiate the population
		thrust::counting_iterator<uint> isb(0);
		thrust::transform(isb, isb + (n_dim * NP), d_og.begin(), prg(x_min, x_max));
		/* Starts a Run */
		B->compute(p_og, p_fog);
		for( uint evals = 0; evals < n_evals; evals += NP ){
			jde->index_gen();
			jde->run(p_og, p_ng);
			B->compute(p_ng, p_fng);
			jde->selection(p_og, p_ng, p_fog, p_fng);
			//jde->update();
	  }
		cudaEventRecord(stop);
    cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		/* End a Run */
		it = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
		printf(" | Execution: %-2d Overall Best: %+.8lf Time(ms): %.8f\n", i, static_cast<float>(*it), time);
		stats.push_back(std::make_pair(static_cast<float>(*it), time));
	}
	/* Statistics of the Runs */
	double FO_mean  = 0.0f, FO_std  = 0.0f;
	double T_mean   = 0.0f, T_std   = 0.0f;
	for( auto it = stats.begin(); it != stats.end(); it++){
		FO_mean += it->first;
		T_mean  += it->second;
	}
	FO_mean /= n_runs;
	T_mean  /= n_runs;
	for( auto it = stats.begin(); it != stats.end(); it++){
		FO_std += (( it->first - FO_mean )*( it->first  - FO_mean ));
		T_std  += (( it->second - T_mean )*( it->second - T_mean  ));
	}
	FO_std /= n_runs;
	FO_std = sqrt(FO_std);
	T_std /= n_runs;
	T_std = sqrt(T_std);
	printf(" +==============================================================+ \n");
	printf(" |                     EXPERIMENTS RESULTS                      | \n");
	printf(" +==============================================================+ \n");
	printf(" | Objective Function:\n");
	printf(" | \t mean:         %+.20E\n", FO_mean);
	printf(" | \t std:          %+.20E\n", FO_std);
	printf(" | Execution Time (ms): \n");
	printf(" | \t mean:         %+.3lf\n", T_mean);
	printf(" | \t std:          %+.3lf\n", T_std);
	printf(" +==============================================================+ \n");

	/*
	printf("================\n");
	h_fog = d_fog;
	for( int i = 0; i < NP; i++ )
		printf("%.3f ", h_fog[i]);
	printf("\n");
	printf("================\n");
	h_og = d_og;
	for( int i = 0; i < NP*n_dim; i++ )
		printf("%.3f ", h_og[i]);
	printf("\n");
	printf("================\n");
	*/
	return 0;
}
