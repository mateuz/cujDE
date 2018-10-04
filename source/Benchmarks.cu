#include "Benchmarks.cuh"

Benchmarks::Benchmarks()
{
  min = -100.0;
  max = +100.0;
  n_threads = 1;
  n_blocks = 1;
  n_dim = 100;
}

Benchmarks::~Benchmarks()
{
  /* empty */
}

float Benchmarks::getMin(){
  return min;
}

float Benchmarks::getMax(){
  return max;
}

uint Benchmarks::getID(){
  return ID;
}

void Benchmarks::setMin( float _min ){
  min = _min;
}

void Benchmarks::setMax( float _max ){
  max = _max;
}

void Benchmarks::setThreads( uint _n){
  n_threads = _n;
}

void Benchmarks::setBlocks( uint _n ){
  n_blocks = _n;
}

uint Benchmarks::getThreads(){
  return n_threads;
}

uint Benchmarks::getBlocks(){
  return n_blocks;
}
