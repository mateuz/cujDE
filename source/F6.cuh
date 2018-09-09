#ifndef _F6_H
#define _F6_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F6 : public Benchmarks
{
public:
  F6( uint, uint );
  ~F6();

  void compute(float * x, float * fitness);
};

__global__ void computeK_F6(float * x, float * f);

#endif
