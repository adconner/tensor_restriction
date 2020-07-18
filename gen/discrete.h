#ifndef _DISCRETE_H_
#define _DISCRETE_H_
#include "util.h"

enum DiscreteAttempt {
  DA_ZERO,
  DA_PM_ONE,
  DA_E3
};
void greedy_discrete(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    int &successes, DiscreteAttempt da = DA_ZERO, int trylimit = 10);
void greedy_discrete_pairs(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    const int trylimit = 10);

#endif
