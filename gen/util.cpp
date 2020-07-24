#include <iterator>
#include <random>
#include <fstream>
#include <memory>
#include <limits>
#include <tuple>
#include "util.h"

// control variables
bool print_lines;

void logsol(double *x, string fname) {
  if (tostdout) {
    cout.precision(numeric_limits<double>::max_digits10);
    cout << "OUTPUT " << fname << " ";
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

PrintCallback::PrintCallback(double *_x) : x(_x), ma_last(1.0) {}
CallbackReturnType PrintCallback::operator()(const IterationSummary& summary) {
  if (print_lines) {
    double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) {return max(std::abs(a),std::abs(b));} ); 
    cout << summary.iteration << " " << summary.cost << " " << ma <<
      " " << summary.relative_decrease << " " <<
      ((ma - ma_last) / ma) / (summary.cost_change / summary.cost)
      << endl;
    ma_last = ma;
    /* if (ma > 4 && summary.iteration >= 10) */
    /*   return SOLVER_ABORT; */
  }
  return SOLVER_CONTINUE;
}

