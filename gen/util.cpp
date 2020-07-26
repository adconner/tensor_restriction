#include <algorithm>
#include <iterator>
#include <fstream>
#include <memory>
#include <limits>
#include <tuple>
#include "util.h"

// control variables
bool print_lines;
mt19937 rng;

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
  return summary.cost < 1e-26 ? SOLVER_TERMINATE_SUCCESSFULLY : SOLVER_CONTINUE;
}

AvoidBorderRankCallback::AvoidBorderRankCallback(double *_x) : x(_x), ma_last(1.0) {}
CallbackReturnType AvoidBorderRankCallback::operator()(const IterationSummary& summary) {
  const int hist = 20;
  const double maxrat_lower = 5e-2;
  const double maxrat_rel_var_upper = 0.06;

  double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) {return max(std::abs(a),std::abs(b));} ); 
  double maxrat = ((ma - ma_last) / ma) / (summary.cost_change / summary.cost);
  ma_last = ma;

  max_rats.push_front(maxrat);
  if (max_rats.size() > hist) {
    max_rats.pop_back();
  }
  if (summary.iteration > hist && all_of(max_rats.begin(),max_rats.end(),[](double mr){return isnormal(mr);})) {

    double mravg = 0;
    for (double mr: max_rats) {
      mravg += mr;
    }
    mravg /= max_rats.size();
    double mrvar = 0;
    for (double mr:max_rats) {
      mrvar += (mr -mravg) * (mr-mravg);
    }
    mrvar = std::sqrt(mrvar / max_rats.size());
    if (ma*mravg >= maxrat_lower && mrvar/mravg <= maxrat_rel_var_upper) {
      return SOLVER_ABORT;
    }
  }
  return SOLVER_CONTINUE;
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
      << " " << summary.step_norm << endl;
    ma_last = ma;
    /* if (ma > 4 && summary.iteration >= 10) */
    /*   return SOLVER_ABORT; */
  }
  return SOLVER_CONTINUE;
}

