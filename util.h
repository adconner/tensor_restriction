#ifndef _UTIL_H_
#define _UTIL_H_
#include <ceres/ceres.h>
#include <string>
#include <complex>
#include <deque>
#include <random>
#include "opts.h"
#include "problem.h"

using namespace std;
using namespace ceres;
typedef complex<double> cx;

extern mt19937 rng;

void logsol(const vector<double> &x, string fname);
void set_value_variable(MyProblem &p, int i);
void set_value_constant(MyProblem &p, int i);
double max_abs(const MyProblem &p);

#endif
