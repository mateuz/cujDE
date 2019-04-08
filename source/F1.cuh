#ifndef _F1_H
#define _F1_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F1 : public Benchmarks
{
public:
  F1( uint, uint );
  ~F1();

  void compute(float * x, float * fitness);
};

__global__ void computeK_F1(float * x, float * f);

__global__ void computeK2_F1(float * x, float * f);

#endif
