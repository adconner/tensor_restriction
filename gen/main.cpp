#include <random>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include <tuple>
#include "ceres/ceres.h"
#include "glog/logging.h"

#include "opts.h"
#include "util.h"
#include "initial.h"
#include "l2reg.h"

using namespace ceres;
using namespace std;

int main(int argc, const char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);
  random_device rd;
  rng.seed(rd());
  /* rng.seed(8); */

  double x[MULT*N];

  Problem::Options popts;
  popts.enable_fast_removal = true;
  Problem problem(popts);
  for (int i=0; i<M; ++i) {
    AddToProblem(problem,x,i);
  }

  fill_initial(x,problem,argc>1 ? argv[1] : "");

  MyTerminationType term = l2_reg_search(problem, x, 1e-2, 1e-4);

  if (term == SOLUTION) {
    double ma = minimize_max_abs(problem, x, 1e-3, 0.8, 1e-4);
    sparsify(problem, x, 1.0, 1e-4);
    cout << "ma " << ma << endl;
  }

  logsol(x,"out.txt");
  return 0;
}
