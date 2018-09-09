#include "constants.cuh"

__constant__ float shift[100];
__constant__ float m_rotation[10000];
__constant__ Configuration params;
__constant__ float F_Lower = 0.10;
__constant__ float F_Upper = 0.90;
__constant__ float T = 0.10;
