#ifndef _F3_H
#define _F3_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F3 : public Benchmarks
{
public:
  F3( uint, uint );
  ~F3();

  void compute(float * x, float * fitness);
};

__global__ void computeK(float * x, float * f);

#endif
