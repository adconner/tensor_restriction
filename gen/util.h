#ifndef _UTIL_H_
#define _UTIL_H_
#include <ceres/ceres.h>
#include <string>
#include <complex>
#include "opts.h"

using namespace std;
using namespace ceres;
typedef complex<double> cx;

extern double sqalpha;
extern bool print_lines;

void logsol(double *x, string fname);

class SolvedCallback : public IterationCallback {
  public:
    CallbackReturnType operator()(const IterationSummary& summary);
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

class Equal : public SizedCostFunction<MULT,MULT> {
  public:
    Equal(double _a, cx _x0);
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const; 
    double a;
    cx x0;
};

class LinearCombination : public SizedCostFunction<MULT,MULT,MULT> {
  public:
    LinearCombination(double _a, double _b);
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const; 
    double a,b;
};

#endif
