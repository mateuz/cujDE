# A GPU-Based jDE Algorithm

###### Population-based search algorithms, such as the Differential Evolution approach, evolve a pool of candidate solutions during the optimization process and are suitable for massively parallel architectures promoted by the use of GPUs. The repository contains a GPU-based implementation of a self-adaptive Differential Evolution employing the jDE mechanism. 

***
##### Requirements

- ##### [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with 9.2)

- ##### GPU Compute Capability (tested with versions 5.2 and 6.1)

- ##### [Boost C++ Libraries - Program Options](https://www.boost.org/) (tested with 1.58.0)

##### Compile

```sh
$ cd repo
$ make
```

##### Parameters Setting

```
$ "runs, r"      - Number of Executions
$ "pop_size, p"  - Population Size
$ "dim, d"       - Number of Dimensions {2, 5, 10, 20, 30, 50, 100}
$ "func_obj, o"  - Function to Optimize [1, 7]
$ "max_eval, e"  - Number of Function Evaluations
$ "help, h"      - Show this help
```

##### Functions Available

<table style="undefined;table-layout: fixed; width: 550px">
<colgroup>
<col style="width: 76px">
<col style="width: 156px">
<col style="width: 80px">
<col style="width: 73px">
<col style="width: 165px">
</colgroup>
  <tr>
    <th>Function</th>
    <th>Name</th>
    <th>Shifted</th>
    <th>Rotated</th>
    <th>Search Space</th>
  </tr>
  <tr>
    <td>1</td>
    <td>S. Sphere</td>
    <td>Yes</td>
    <td>No</td>
    <td>[-100, +100]</td>
  </tr>
  <tr>
    <td>2</td>
    <td>S. Rosenbrock</td>
    <td>Yes</td>
    <td>No</td>
    <td>[-100, +100]</td>
  </tr>
  <tr>
    <td>3</td>
    <td>S. Griewank</td>
    <td>Yes</td>
    <td>No</td>
    <td>[-600, +600]</td>
  </tr>
  <tr>
    <td>4</td>
    <td>S. Rastrigin</td>
    <td>Yes</td>
    <td>No</td>
    <td>[-5, +5]</td>
  </tr>
  <tr>
    <td>5</td>
    <td>S. R. Rosenbrock</td>
    <td>Yes</td>
    <td>Yes</td>
    <td>[-100, +100]</td>
  </tr>
  <tr>
    <td>6</td>
    <td>S. R. Griewank</td>
    <td>Yes</td>
    <td>Yes</td>
    <td>[-600, +600]</td>
  </tr>
  <tr>
    <td>7</td>
    <td>S. R. Rastrigin</td>
    <td>Yes</td>
    <td>Yes</td>
    <td>[-5, +5]</td>
  </tr>
</table>
##### Execute

```sh
$ cd repo
$ ./demo <parameter setting> or make run (with default parameters)
```

##### Clean up

```sh
$ make clean
```

##### TODO

- Empty list for a while

***

[1] J. Brest, V. Zumer and M. S. Maucec, "Self-Adaptive Differential Evolution Algorithm in Constrained Real-Parameter Optimization," 2006 IEEE International Conference on Evolutionary Computation, Vancouver, BC, 2006, pp. 215-222. doi: 10.1109/CEC.2006.1688311, [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1688311&isnumber=35623)

[2] [CUDA is a parallel computing platform and programming model developed by NVIDIA for GPGPU](https://developer.nvidia.com/cuda-zone)
