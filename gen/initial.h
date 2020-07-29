#ifndef _INITIAL_H_
#define _INITIAL_H_

#include "ceres/ceres.h"

void fill_initial(double *x, ceres::Problem &problem, std::string fname="");

#endif
