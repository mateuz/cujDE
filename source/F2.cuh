#ifndef _F2_H
#define _F2_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F2 : public Benchmarks
{
public:
  F2( uint, uint );
  ~F2();

  void compute(float * x, float * fitness);
};

__global__ void computeK2(float * x, float * f);

#endif
