#ifndef _2DAB_H
#define _2DAB_H

#include "Benchmarks.cuh"
#include "helper.cuh"

typedef struct {
  double x, y;
} AB_2D;

class F2DAB : public Benchmarks
{
private:
  /* empty */

public:
  F2DAB( uint, uint );
  ~F2DAB();

  void compute(float * x, float * fitness);

};

__device__ float _C( uint, uint );

__global__ void computeK_2DAB(float * x, float * f);

#endif
