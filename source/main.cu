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

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "jDE.cuh"

void show_params(
	uint n_runs,
	uint NP,
	uint n_evals,
	uint n_dim,
	float F,
	float CR,
	std::string FuncObj
){
	printf(" | Number of Executions:                    %d\n", n_runs);
	printf(" | Population Size:                         %d\n", NP);
	printf(" | Number of Dimensions:                    %d\n", n_dim);
	printf(" | Number of Function Evaluations:          %d\n", n_evals);
	printf(" | Mutation Factor:                         %.2f\n", F);
	printf(" | Crossover Probability:                   %.2f\n", CR);
	printf(" | Optimization Function:                   %s\n", FuncObj.c_str());
}

int main(int argc, char * argv[]){
	uint n_runs, NP, n_evals, n_dim, f_id;
	float F, CR;
	std::string FuncObj;

	try{
		po::options_description config("Opções");
		config.add_options()
			("runs,r"    , po::value<uint>(&n_runs)->default_value(5)    , "Number of Executions"          )
			("pop_size,p", po::value<uint>(&NP)->default_value(20)       , "Population Size"               )
			("dim,d"     , po::value<uint>(&n_dim)->default_value(10)    , "Number of Dimensions"          )
			("func_obj,o", po::value<uint>(&f_id)->default_value(1)      , "Function to Optimize"          )
			("max_eval,e", po::value<uint>(&n_evals)->default_value(10e5), "Number of Function Evaluations")
			("mf,f"      , po::value<float>(&F)->default_value(0.5)      , "Mutation Factor [0, 2]"        )
			("cr,c"      , po::value<float>(&CR)->default_value(0.9)     , "Crossover Probability [0, 1]"  )
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
	show_params(n_runs, NP, n_evals, n_dim, F, CR, FuncObj);
	printf(" +==============================================================+ \n");

	for( int i = 1; i <= n_runs; i++ ){
		printf(" | Execution: %-2d Overall Best: %+.10E Time (s): %.3lf\n",
		 i, 0.00, 0.00);
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

	jDE * jde = new jDE(20, 10);
	jde->run();
	return 0;
}
