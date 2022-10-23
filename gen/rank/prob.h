#ifndef _RANK_PROB_H_
#define _RANK_PROB_H_

#include "ceres/ceres.h"
extern int N;
extern int M;
#ifdef CX
#define MULT 2
#else
#define MULT 1
#endif
extern int BLOCKS;
extern int *BBOUND;
ceres::ResidualBlockId AddToProblem(ceres::Problem &p, double *x, int eqi);
void SetParameterBlockOrdering(ceres::ParameterBlockOrdering &pbo, double *x);
class Eq : public ceres::CostFunction {
    public: double alpha = 1.0;
};

#endif
