#include <random>
#include <fstream>
#include <iterator>
#include <limits>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "prob.h"

using namespace ceres;
using namespace std;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double x[N];
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> uni(0,1);
  generate_n(x,N,[&] {return uni(gen);});

  Problem problem;
  AddToProblem(problem,x);

  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  /* options.max_num_iterations = 10000; */
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;

  /* options.minimizer_type = TRUST_REGION; */
  // trust region options
  /* options.use_inner_iterations = true; */
  options.use_nonmonotonic_steps = true;
  /* options.trust_region_strategy_type = // */

  // linear solver options
  /* options.dynamic_sparsity = true; // since solutions are typically sparse? */

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  cout << summary.FullReport() << "\n";
  ofstream out("out.txt");
  out.precision(numeric_limits<double>::max_digits10);
  copy(x,x+N,ostream_iterator<double>(out,"\n"));
  return 0;
}
