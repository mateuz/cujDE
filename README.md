# A Self-Adaptive Differential Evolution (jDE) on GPU using CUDA

###### A CUDA-Based implementation of a self-adaptive differential evolution namely jDE [1, 2]. Differential Evolution is one of the most used algorithm for optimization in the continuous domain, but is highly sensitive to parameters. The self-adaptive aims to remove the responsability of the designer to define the parameters values, given more robustness to the algorithm. The jDE algorithm uses a self-adapting mechanism on the control parameters F (mutation scale factor) and CR (crossover probability), changing his values during the run. This implementation uses DE/rand/1/bin strategy to create mutant vectors.

***

##### Compile

```sh
$ cd repo
$ make
```
##### Execute

```sh
$ cd repo
$ ./GjDE-demo or make run
```

#### Clean up

```sh
$ make clean
```

### TODO

    - Empty list for a while
    
***

[1] J. Brest, V. Zumer and M. S. Maucec, "Self-Adaptive Differential Evolution Algorithm in Constrained Real-Parameter Optimization," 2006 IEEE International Conference on Evolutionary Computation, Vancouver, BC, 2006, pp. 215-222. doi: 10.1109/CEC.2006.1688311, [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1688311&isnumber=35623)

[2] [CUDA is a parallel computing platform and programming model developed by NVIDIA for GPGPU](https://developer.nvidia.com/cuda-zone)
