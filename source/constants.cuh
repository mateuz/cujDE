#ifndef _CONSTANTS_H
#define _CONSTANTS_H

typedef struct {
    float x_min;
    float x_max;
    uint n_dim;
    uint ps;
} config;

extern __constant__ float shift[1000];
extern __constant__ config params;

#endif
