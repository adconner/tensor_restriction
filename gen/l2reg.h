#ifndef _L2REG_H_
#define _L2REG_H_
#include "problem.h"
#include "util.h"

enum MyTerminationType {
  CONTINUE,
  CONTINUE_RESET,
  SOLUTION,
  BORDER_LIKELY,
  NO_SOLUTION,
  UNKNOWN
};

MyTerminationType l2_reg_search(MyProblem &problem,
    double target_relative_decrease, double ftol, bool stop_on_br = true);
double minimize_max_abs(MyProblem &problem, double eps=1e-3, 
    double step_mult = 0.95, double relftol = 1e-4);
void sparsify(MyProblem &problem, double B, double relftol);

#endif
