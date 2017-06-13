#include <random>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "prob.h"

using namespace ceres;
using namespace std;

class SolvedCallback : public IterationCallback {
  public:
    SolvedCallback(double solved) : _solved(solved) {}
    CallbackReturnType operator()(const IterationSummary& summary) {
      return summary.cost < _solved ? SOLVER_TERMINATE_SUCCESSFULLY : SOLVER_CONTINUE;
    }
  private:
    double _solved;
};

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
  auto solved = make_unique<SolvedCallback>(1e-3);
  options.callbacks.push_back(solved.get());
  /* options.max_num_iterations = 10000; */
  /* options.max_num_iterations = 1000; */
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;

  /* options.minimizer_type = LINE_SEARCH; */
  // trust region options
  options.trust_region_strategy_type = LEVENBERG_MARQUARDT;
  options.use_nonmonotonic_steps = true;
  /* options.use_inner_iterations = true; */

  // line search options
  /* options.line_search_direction_type = NONLINEAR_CONJUGATE_GRADIENT; */
  /* options.line_search_type = ARMIJO; */
  /* options.nonlinear_conjugate_gradient_type = POLAK_RIBIERE; */
  /* options.nonlinear_conjugate_gradient_type = HESTENES_STIEFEL; */

  // linear solver options
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.dynamic_sparsity = true; // since solutions are typically sparse?
  /* options.use_postordering = true; */

  /* options.linear_solver_type = DENSE_QR; */
  /* options.dense_linear_algebra_library_type = LAPACK; */

  /* options.linear_solver_type = CGNR; */
  /* options.linear_solver_type = ITERATIVE_SCHUR; */

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  cout << summary.FullReport() << "\n";
  ofstream out("out.txt");
  out.precision(numeric_limits<double>::max_digits10);
  copy(x,x+N,ostream_iterator<double>(out,"\n"));
  return 0;
}
