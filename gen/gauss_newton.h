#ifndef _GAUSS_NEWTON_H_
#define _GAUSS_NEWTON_H_
#include "util.h"

void gauss_newton(Problem &p, double *x, double xtol, int max_it, double rcond=1e-14); 

#endif
