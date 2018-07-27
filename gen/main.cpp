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
  /* options.linear_solver_type = SPARSE_NORMAL_CHOLESKY; */
  /* options.dynamic_sparsity = true; // since solutions are typically sparse? */
  /* options.use_postordering = true; */

  options.linear_solver_type = DENSE_NORMAL_CHOLESKY;
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

  Problem problem;
  ceres::ParameterBlockOrdering pbo;
  AddToProblem(problem,pbo,x);
  Problem::EvaluateOptions eopts;
  problem.GetResidualBlocks(&eopts.residual_blocks);
  // save residuals we care about before adding other regularization

  fill_initial(x,argc,argv,problem);

  Solver::Options options;
  solver_opts(options);
  options.callbacks.push_back(new SolvedCallback);
  if (verbose) {
    options.update_state_every_iteration = true;
    options.callbacks.push_back(new PrintCallback(x));
  }
  print_lines = verbose;

  if (l2_reg_always || (l2_reg_random_start && argc == 1)) {
    vector<ResidualBlockId> rids;
    for (int i=0; i<N; ++i) {
      rids.push_back(problem.AddResidualBlock(new L2Regularization, NULL, &x[MULT*i]));
    }
    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_rough;
    options.function_tolerance = ftol_rough;
    sqalpha = std::sqrt(alphastart);
    for (int i=l2_reg_steps; i>0; --i, sqalpha *= std::sqrt(l2_reg_decay)) {
      Solver::Summary summary;
      if (verbose) {
        cout << "l2 regularization coefficient " << (sqalpha * sqalpha) << endl;
        cout.flush();
      }
      Solve(options, &problem, &summary);
      /* if (verbose) cout << summary.FullReport() << "\n"; */
    }
    for (auto rid : rids) {
      problem.RemoveResidualBlock(rid);
    }
    if (verbose) cout << "rough solving..." << endl;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    options.function_tolerance = ftol;

    /* vector<double> rs; problem.Evaluate(eopts,0,&rs,0,0); */
    /* int bad = count_if(rs.begin(),rs.end(), */
    /*       [](double r){return std::abs(r) > abort_worse;}); */
    /* if (0 < bad && bad <= 2) { */
    /*   logsol(x,"out_almost.txt"); */
    /*   return 0; */
    /* } */

    if (log_rough) logsol(x,"out_rough.txt");
    double cost; problem.Evaluate(eopts,&cost,0,0,0);
    if (cost > abort_worse) {
      if (verbose) cout << summary.FullReport() << endl << 
        "cost " << cost << " not good enough to refine. Aborting" << endl;
      return 2;
    } else if (verbose) {
      cout << "fine tuning solution..." << endl;
    }
  }

  options.minimizer_type = TRUST_REGION;
  options.max_num_iterations = iterations_fine;
  Solver::Summary summary;
  if (record_iterations) {
    options.update_state_every_iteration = true;
    auto record = make_unique<RecordCallback>(x,cerr);
    options.callbacks.push_back(record.get());
    Solve(options, &problem, &summary);
    options.callbacks.pop_back();
  } else {
    Solve(options, &problem, &summary);
  }
  /* cout << summary.FullReport() << endl; */

  logsol(x,"out_dense.txt");

  if (attemptsparse) {
    if (summary.final_cost > attempt_sparse_thresh) {
      if (verbose) {
        cout << "accuracy fail, not sparsifying" << endl;
        cout << summary.FullReport() << "\n";
      }
      return 1;
    }
    if (verbose) cout << "solution seems good, sparsifying..." << endl;
    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_discrete;
    options.function_tolerance = ftol_discrete;
    print_lines = false;

    int successes = 0;
    greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N);
    greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE_ZERO,N/2+1);
    for (int refine=1; refine<=1; ++refine) {
      options.max_num_iterations *= 2;
      options.function_tolerance *= options.function_tolerance;
      greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N/10+1);
      greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE_ZERO,N/10+1);
      /* greedy_discrete_pairs(problem,x,options,eopts,N/10+1); */
    }
    /* greedy_discrete_pairs(problem,x,options,eopts,N/10+1); */
    /* greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N/10+1); */
    /* greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE_ZERO,N/10+1); */

    logsol(x,"out.txt");
  }
  return 0;
}
