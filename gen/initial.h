#ifndef _INITIAL_H_
#define _INITIAL_H_

#include "ceres/ceres.h"

void fill_initial(double *x, int argc, const char **argv, ceres::Problem &problem);

#endif
