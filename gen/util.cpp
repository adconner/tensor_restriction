#include <algorithm>
#include <iterator>
#include <fstream>
#include <memory>
#include <limits>
#include <tuple>
#include "util.h"

// control variables
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

