#ifndef _F5_H
#define _F5_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F5 : public Benchmarks
{
public:
  F5( uint, uint );
  ~F5();

  void compute(float * x, float * fitness);
};

__global__ void computeK_F5(float * x, float * f);

#endif
