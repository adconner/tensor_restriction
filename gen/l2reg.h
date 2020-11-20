#ifndef _L2REG_H_
#define _L2REG_H_
#include "util.h"

enum MyTerminationType {
  CONTINUE,
  SOLUTION,
  BORDER_LIKELY,
  NO_SOLUTION,
  UNKNOWN
};

MyTerminationType l2_reg_search(Problem &problem, double *x, double target_relative_decrease, double ftol);
double minimize_max_abs(Problem &problem, double *x, double eps=1e-3, double step_mult = 0.95, double relftol = 1e-4);
void sparsify(Problem &problem, double *x, double B, double relftol);

#endif
