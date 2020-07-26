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

extern bool print_lines;
extern mt19937 rng;

void logsol(double *x, string fname);

class SolvedCallback : public IterationCallback {
  public:
    CallbackReturnType operator()(const IterationSummary& summary);
};

class AvoidBorderRankCallback : public IterationCallback {
  public:
    AvoidBorderRankCallback(double *_x);
    CallbackReturnType operator()(const IterationSummary& summary);
    deque<double> max_rats;
    double *x;
    double ma_last;
};

class RecordCallback : public IterationCallback {
  public:
    RecordCallback(double *_x, ostream &_out);
    CallbackReturnType operator()(const IterationSummary& summary);
    double *x;
    ostream &out;
};

class PrintCallback : public IterationCallback {
  public:
    PrintCallback(double *_x);
    CallbackReturnType operator()(const IterationSummary& summary); 
    double *x;
    double ma_last;
};

#endif
