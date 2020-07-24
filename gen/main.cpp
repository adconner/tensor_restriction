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
#include "discrete.h"
#include "initial.h"
#include "l2reg.h"

using namespace ceres;
using namespace std;

void solver_opts(Solver::Options &options) {
  options.num_threads = 1;
  options.num_linear_solver_threads = 1;

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


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);

  double x[MULT*N];

  Problem::Options popts;
  popts.enable_fast_removal = true;
  Problem problem(popts);
  for (int i=0; i<M; ++i) {
    AddToProblem(problem,x,i);
  }
  Problem::EvaluateOptions eopts;
  problem.GetResidualBlocks(&eopts.residual_blocks);
  // save residuals we care about before adding other regularization
  ceres::ParameterBlockOrdering pbo;
  SetParameterBlockOrdering(pbo, x);

  fill_initial(x,argc,argv,problem);

  /* int maxi = 42*25; */
  /* for (int i=maxi; i < N; ++i) { */
  /*   for (int j=0; j<MULT; ++j) { */
  /*     x[MULT*i+j] = 0; */
  /*   } */
  /*   problem.SetParameterBlockConstant(x+MULT*i); */
  /* } */

  Solver::Options options;
  solver_opts(options);
  options.callbacks.push_back(new SolvedCallback);
  options.callbacks.push_back(new AvoidBorderRankCallback(x));
  if (verbose) {
    options.update_state_every_iteration = true;
    options.callbacks.push_back(new PrintCallback(x));
  }
  print_lines = verbose;

  if (l2_reg_always || (l2_reg_random_start && argc == 1)) {
    l2_reg_search(problem, x, options);

    if (log_rough) logsol(x,"out_rough.txt");
    double cost; problem.Evaluate(eopts,&cost,0,0,0);
    if (cost > abort_worse) {
      if (verbose) cout << "cost " << cost << 
        "rough solution worse than abort_worse. Aborting" << endl;
      return 2;
    } else if (verbose) {
      cout << "rough solution better than abort_worse. Fine tuning solution..." << endl;
    }
  }

  options.minimizer_type = TRUST_REGION;
  options.max_num_iterations = iterations_fine;
  Solver::Summary summary;
  double cost; problem.Evaluate(eopts,&cost,0,0,0);
  if (cost > solved_fine) {
    if (record_iterations) {
      options.update_state_every_iteration = true;
      auto record = make_unique<RecordCallback>(x,cerr);
      options.callbacks.push_back(record.get());
      Solve(options, &problem, &summary);
      options.callbacks.pop_back();
    } else {
      Solve(options, &problem, &summary);
    }
  }
  /* cout << summary.FullReport() << endl; */

  logsol(x,"out_dense.txt");

  if (attemptsparse) {
    if (summary.final_cost > attempt_sparse_thresh) {
      if (verbose) {
        cout << "solution worse than attempt_sparse_thresh, not sparsifying" << endl;
        cout << summary.FullReport() << "\n";
      }
      return 1;
    }
    if (verbose) cout << "solution better than attempt_sparse_thresh, sparsifying..." << endl;

    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_discrete;
    options.function_tolerance = ftol_discrete;
    print_lines = false;
    options.linear_solver_type = DENSE_NORMAL_CHOLESKY;

    int successes = 0;
    greedy_discrete(problem,x,options,successes,DA_ZERO,N);
    /* greedy_discrete_pairs(problem,x,options,eopts,N); */
    /* greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE,N); */

    l2_reg_refine(problem,x,options);
    greedy_discrete_lines(problem,x,options,N-successes,6);
    greedy_discrete_lines(problem,x,options,N-successes,12);
    greedy_discrete(problem,x,options,successes,DA_E3,N-successes);

    /* for (int refine=1; refine<=1; ++refine) { */
    /*   options.max_num_iterations *= 2; */
    /*   options.function_tolerance *= options.function_tolerance; */
    /*   greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N/10+1); */
    /*   greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE,N/10+1); */
    /*   /1* greedy_discrete_pairs(problem,x,options,eopts,N/10+1); *1/ */
    /* } */
    /* greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N/10+1); */
    /* greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE,N/10+1); */

    logsol(x,"out.txt");
  }
  return 0;
}
