#ifndef _PROBLEM_H_
#define _PROBLEM_H_

#include "ceres/ceres.h"
#include <vector>

#include "prob.h"

using namespace std;
using namespace ceres;

struct MyProblem {
  Problem p;
  vector<double> x;
  vector<bool> variable_mask;

  MyProblem(Problem::Options popts, int xlen) : p(popts), 
    x(xlen*MULT), variable_mask(xlen,true) { }
};

#endif
