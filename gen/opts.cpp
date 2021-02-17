#include "opts.h"
using namespace ceres;

void solver_opts(Solver::Options &options) {
  options.num_threads = 1;

  /* options.minimizer_type = LINE_SEARCH; */
  // trust region options
  options.trust_region_strategy_type = LEVENBERG_MARQUARDT;
  /* options.use_nonmonotonic_steps = true; */
  /* options.use_inner_iterations = true; */

  // line search options
  /* options.line_search_direction_type = BFGS; */
  /* options.line_search_type = ARMIJO; */
  /* options.nonlinear_conjugate_gradient_type = POLAK_RIBIERE; */
  /* options.nonlinear_conjugate_gradient_type = HESTENES_STIEFEL; */

  // linear solver options
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  /* options.dynamic_sparsity = true; // since solutions are typically sparse? */
  /* options.use_postordering = true; */

  /* options.linear_solver_type = DENSE_NORMAL_CHOLESKY; */
  /* options.linear_solver_type = DENSE_QR; */
  /* options.linear_solver_type = CGNR; */
  /* options.linear_solver_type = ITERATIVE_SCHUR; */

  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.dense_linear_algebra_library_type = LAPACK;

  options.function_tolerance = ftol;
  options.parameter_tolerance = ptol;
  options.gradient_tolerance = gtol;
}

