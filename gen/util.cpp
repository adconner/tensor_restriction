#include <iterator>
#include <random>
#include <fstream>
#include <memory>
#include <limits>
#include <tuple>
#include "util.h"

// control variables
double sqalpha; // square root of l2 regularization coefficient
bool print_lines;

void logsol(double *x, string fname) {
  if (tostdout) {
    cout.precision(numeric_limits<double>::max_digits10);
    cout << fname << " ";
    copy(x,x+MULT*N,ostream_iterator<double>(cout," "));
    cout << endl;
  } else {
    ofstream out(fname);
    out.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(out,"\n"));
  }
}

CallbackReturnType SolvedCallback::operator()(const IterationSummary& summary) {
  return summary.cost < 1e-29 ? SOLVER_TERMINATE_SUCCESSFULLY : SOLVER_CONTINUE;
}

RecordCallback::RecordCallback(double *_x, ostream &_out) : x(_x), out(_out) {}
CallbackReturnType RecordCallback::operator()(const IterationSummary& summary) {
  copy(x,x+MULT*N,ostream_iterator<double>(out," ")); out << endl;
  return SOLVER_CONTINUE;
}

PrintCallback::PrintCallback(double *_x) : x(_x) {}
CallbackReturnType PrintCallback::operator()(const IterationSummary& summary) {
  if (print_lines) {
    double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) {return max(std::abs(a),std::abs(b));} ); 
    cout << summary.iteration << " " << summary.cost << " " << ma <<
      " " << summary.relative_decrease << endl;
    /* if (ma > 4 && summary.iteration >= 10) */
    /*   return SOLVER_ABORT; */
  }
  return SOLVER_CONTINUE;
}

bool L2Regularization::Evaluate(const double* const* x,
                    double* residuals,
                    double** jacobians) const {
  residuals[0] = sqalpha * x[0][0];
  if (MULT == 2) residuals[1] = sqalpha * x[0][1];
  if (jacobians) {
    if (jacobians[0]) {
      jacobians[0][0] = sqalpha;
      if (MULT == 2) {
        jacobians[0][1] = 0;
        jacobians[0][2] = 0;
        jacobians[0][3] = sqalpha;
      }
    }
  }
  return true;
}

Equal::Equal(double _a, cx _x0) : a(_a), x0(_x0) {}
bool Equal::Evaluate(const double* const* x,
                    double* residuals,
                    double** jacobians) const {
  residuals[0] = a*(x[0][0]-x0.real());
  if (MULT == 2) residuals[1] = a*(x[0][1]-x0.imag());
  if (jacobians && jacobians[0]) {
    jacobians[0][0] = a;
    if (MULT == 2) {
      jacobians[0][1] = 0;
      jacobians[0][2] = 0;
      jacobians[0][3] = a;
    }
  }
  return true;
}

LinearCombination::LinearCombination(double _a, double _b) : a(_a), b(_b) {}
bool LinearCombination::Evaluate(const double* const* x,
                    double* residuals,
                    double** jacobians) const {
  residuals[0] = a*x[0][0] + b*x[1][0];
  if (MULT == 2) residuals[1] = a*x[0][1] + b*x[1][1];
  if (jacobians) {
    if (jacobians[0]) {
      jacobians[0][0] = a;
      if (MULT == 2) {
        jacobians[0][1] = 0;
        jacobians[0][2] = 0;
        jacobians[0][3] = a;
      }
    }
    if (jacobians[1]) {
      jacobians[1][0] = b;
      if (MULT == 2) {
        jacobians[1][1] = 0;
        jacobians[1][2] = 0;
        jacobians[1][3] = b;
      }
    }
  }
  return true;
}
