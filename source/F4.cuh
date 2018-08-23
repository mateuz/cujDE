#ifndef _F4_H
#define _F4_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F4 : public Benchmarks
{
public:
  F4( uint, uint );
  ~F4();

  void compute(float * x, float * fitness);
};

__global__ void computeK4(float * x, float * f);

#endif
