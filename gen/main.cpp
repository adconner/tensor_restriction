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
#include "problem.h"
#include "discrete.h"

using namespace ceres;
using namespace std;

int main(int argc, const char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);
  random_device rd;
  rng.seed(rd());
  /* rng.seed(8); */

  Problem::Options popts;
  popts.enable_fast_removal = true;
  MyProblem p(popts,N);
  for (int i=0; i<M; ++i) {
    AddToProblem(p.p,p.x.data(),i);
  }

  fill_initial(p,argc>1 ? argv[1] : "");

  MyTerminationType term = l2_reg_search(p, 1e-2, 1e-4);
  cout << term << endl;

  /* if (term == SOLUTION || term == BORDER_LIKELY) { */
  /*   logsol(p,"out.txt"); */
  /*   return 0; */
  /* } */
  /* return 1; */

  /* if (term == SOLUTION) { */
  /*   logsol(x,"out_dense.txt"); */
  /*   double ma = minimize_max_abs(p, x, 1e-3, 0.8, 1e-4); */
  /*   sparsify(p, x, 1.0, 1e-4); */
  /*   sparsify(p, x, 1.0, 1e-4); */
  /*   cout << "ma " << ma << endl; */
  /*   logsol(p,"out.txt"); */
  /*   return 0; */
  /* } */
  /* return 1; */

  if (term == SOLUTION) {
    logsol(p,"out_dense.txt");
    minimize_max_abs(p, 1e-1);
    /* sparsify(p, x, 1.0, 1e-4); */
    int successes = 0;
    greedy_discrete(p, successes, DA_ZERO, N);
    greedy_discrete(p, successes, DA_E3, N);
    /* greedy_discrete_pairs(p, N*100); */
    double ma = minimize_max_abs(p);
    cout << "ma " << ma << endl;
    logsol(p,"out.txt");
    return 0;
  }
  return 1;

  logsol(p,"out.txt");
  return 0;
}
