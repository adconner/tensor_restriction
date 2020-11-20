#ifndef _UTIL_H_
#define _UTIL_H_
#include <ceres/ceres.h>
#include <string>
#include <complex>
#include <deque>
#include <random>
#include "opts.h"

using namespace std;
using namespace ceres;
typedef complex<double> cx;

extern mt19937 rng;

void logsol(double *x, string fname);

#endif
