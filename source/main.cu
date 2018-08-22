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

int main(int argc, char * argv[]){
	srand(time(NULL));
	uint n_runs, NP, n_evals, n_dim, f_id;
	std::string FuncObj;

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
	show_params(n_runs, NP, n_evals, n_dim, FuncObj);
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

	float x_min = -10.0;
	float x_max = +10.0;
	float time  = 0.00;

	for( int i = 1; i <= n_runs; i++ ){
		/* Randomly initiate the population */
		thrust::counting_iterator<uint> isb(0);
		thrust::transform(isb, isb + (n_dim * NP), d_og.begin(), prg(x_min, x_max));
		jDE * jde = new jDE(NP, n_dim, x_min, x_max);

		cudaEventRecord(start);
		for( uint evals = 0; evals < n_evals; evals += NP ){
			jde->index_gen();
			jde->run(p_og, p_ng);
			jde->selection(p_og, p_ng, p_fog, p_fng);
			jde->update();
		}
		cudaEventRecord(stop);
    cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf(" | Execution: %-2d Overall Best: %+.10E Time(ms): %.5f\n", i, 0.00, time);
	}

	double FO_mean  = 0.0f, FO_std  = 0.0f;
	double T_mean   = 0.0f, T_std   = 0.0f;
	printf(" +==============================================================+ \n");
	printf(" |                     EXPERIMENTS RESULTS                      | \n");
	printf(" +==============================================================+ \n");
	printf(" | Objective Function:\n");
	printf(" | \t mean:         %+.20E\n", FO_mean);
	printf(" | \t std:          %+.20E\n", FO_std);
	printf(" | Execution Time: \n");
	printf(" | \t mean:         %+.3lf\n", T_mean);
	printf(" | \t std:          %+.3lf\n", T_std);
	printf(" +==============================================================+ \n");

	/*
	printf("================\n");
	h_og = d_og;
	for( int i = 0; i < NP*n_dim; i++ )
		printf("%.3f ", h_og[i]);
	printf("\n");
	printf("================\n");
	h_ng = d_ng;
	for( int i = 0; i < NP*n_dim; i++ )
		printf("%.3f ", h_ng[i]);
	printf("\n");
	printf("================\n");*/
	return 0;
}
