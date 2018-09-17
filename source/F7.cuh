#ifndef _F7_H
#define _F7_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F7 : public Benchmarks
{
public:
  F7( uint, uint );
  ~F7();

  void compute(float * x, float * fitness);
};

__global__ void computeK_F7(float * x, float * f);

#endif
