#ifndef _L2REG_H_
#define _L2REG_H_
#include "util.h"

void l2_reg_search(Problem &p, double *x, const Solver::Options & opts);
void l2_reg_refine(Problem &p, double *x, const Solver::Options & opts);

#endif
