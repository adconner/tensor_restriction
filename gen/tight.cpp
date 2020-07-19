#include <random>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include <tuple>
#include "ceres/ceres.h"
#include "glog/logging.h"

#include "ClpSimplex.hpp"
#include "CoinHelperFunctions.hpp"
#include "CoinTime.hpp"
#include "CoinBuild.hpp"
#include "CoinModel.hpp"

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

  options.trust_region_strategy_type = LEVENBERG_MARQUARDT;

  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  /* options.linear_solver_type = DENSE_NORMAL_CHOLESKY; */

  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.dense_linear_algebra_library_type = LAPACK;

  options.function_tolerance = ftol_rough;
  options.parameter_tolerance = ptol;
  options.gradient_tolerance = gtol;
}

bool done(const set<int> &solved) {
  ClpSimplex model; 
  model.resize(0,TDIM);
  for (int i=0; i<TDIM; ++i) {
    model.setColumnLower(i,-COIN_DBL_MAX);
    model.setColumnUpper(i,COIN_DBL_MAX);
  }
  CoinBuild build;
  vector<int> ixs;
  vector<double> ts;
  for (int i=0; i<M; ++i) {
    if (!solved.count(i)) {
      ixs.clear(); ts.clear();
      for (int j=0; j<TDIM; ++j) {
        if (tight[i][j] != 0) {
          ixs.push_back(j);
          ts.push_back(tight[i][j]);
        }
      }
      build.addRow(ixs.size(),ixs.data(),ts.data(),1.0,COIN_DBL_MAX);
    }
  }
  model.addRows(build);
  model.primal();
  return !model.isProvenPrimalInfeasible();
}

bool for_each_solvable(const set<int> &solved, int start, function<bool(int)> f) {
  // want nonzero vector with dot product with solved to be nonnegative
  // nonzero can be asserted by saying sums of dot products is positive
  ClpSimplex model; 
  model.resize(0,TDIM);
  for (int i=0; i<TDIM; ++i) {
    model.setColumnLower(i,-COIN_DBL_MAX);
    model.setColumnUpper(i,COIN_DBL_MAX);
  }
  /* CoinBuild build; */
  vector<int> ixs;
  vector<double> ts;
  for (int i : solved) {
    ixs.clear(); ts.clear();
    for (int j=0; j<TDIM; ++j) {
      if (tight[i][j] != 0) {
        ixs.push_back(j);
        ts.push_back(tight[i][j]);
      }
    }
    /* build.addRow(ixs.size(),ixs.data(),ts.data(),-COIN_DBL_MAX,0.0); */
    model.addRow(ixs.size(),ixs.data(),ts.data(),-COIN_DBL_MAX,0.0);
  }
  for (int i=0; i<start; ++i) {
    if (!solved.count(i)) {
      ixs.clear(); ts.clear();
      for (int j=0; j<TDIM; ++j) {
        if (tight[i][j] != 0) {
          ixs.push_back(j);
          ts.push_back(tight[i][j]);
        }
      }
      /* build.addRow(ixs.size(),ixs.data(),ts.data(),1.0,COIN_DBL_MAX); */
      model.addRow(ixs.size(),ixs.data(),ts.data(),1.0,COIN_DBL_MAX);
    }
  }
  /* model.addRows(build); */

  cout << start << endl;
  model.primal();
  assert(model.isProvenOptimal());

  for (int i=start; i<M; ++i) {
    if (!solved.count(i)) {
      ixs.clear(); ts.clear();
      for (int j=0; j<TDIM; ++j) {
        if (tight[i][j] != 0) {
          ixs.push_back(j);
          ts.push_back(tight[i][j]);
        }
      }
      model.addRow(ixs.size(),ixs.data(),ts.data(),COIN_DBL_MIN,0.0);

      model.primal();
      if (model.isProvenOptimal() && f(i)) {
        return true;
      }

      model.setRowLower(model.getNumRows()-1,1.0);
      model.setRowUpper(model.getNumRows()-1,COIN_DBL_MAX);
      model.primal();
      if (model.isProvenPrimalInfeasible()) {
        return false;
      }
    }
  }
  return true;
}



bool dfs(Problem &problem, double *x, const Solver::Options &options,
    set<int> &solved, int start) {
  return for_each_solvable(solved, start, [&](int i) {
    if (dfs(problem,x,options,solved,i+1)) {
      return true;
    }

    vector<double> sav(x,x+N*MULT);
    auto rid = AddToProblem(problem,x,i);
    // TODO set weight small
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    double cost; problem.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0);
    if (cost > solved_fine) { // solve fail
      copy(sav.begin(),sav.end(),x);
      problem.RemoveResidualBlock(rid);
      return false;
    }
    l2_reg_refine(problem,x,options);

    solved.insert(i);
    if (dfs(problem,x,options,solved,i+1)) {
      return true;
    }
    solved.erase(i);
    copy(sav.begin(),sav.end(),x); // this is unnecessary but should be smaller values
    problem.RemoveResidualBlock(rid);
    return false;
  });
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);

  double x[MULT*N];

  Problem problem;

  set<int> solved; // set inital solved and add equations
#ifdef TIGHT
  for (int i=0; i<M; ++i) {
    bool nz = false;
    for (int j=0; j<TDIM; ++j) {
      if (tight[i][j] != 0) {
        nz = true;
        break;
      }
    }
    if (!nz) {
      solved.insert(i);
    }
  }
#else
  for (int i=0; i<M; ++i) solved.insert(i); // this case irrelevant, but do something
#endif

  for (auto i : solved) {
    AddToProblem(problem,x,i);
  }

  fill_initial(x,argc,argv,problem);

  /* int maxi = 42*25; */
  /* for (int i=MULT*maxi; i < MULT*N; ++i) { */
  /*   for (int j=0; j<MULT; ++j) { */
  /*     x[MULT*i+j] = 0; */
  /*   } */
  /*   problem.SetParameterBlockConstant(x+MULT*i); */
  /* } */

  Solver::Options options;
  solver_opts(options);
  options.callbacks.push_back(new SolvedCallback);
  if (verbose) {
    options.update_state_every_iteration = true;
    options.callbacks.push_back(new PrintCallback(x));
  }
  print_lines = verbose;
  l2_reg_search(problem, x, options);
  print_lines = false;

  if (dfs(problem,x,options,solved,0)) {
    cout << endl << endl << "SOLUTION FOUND" << endl << endl << endl;
  } else {
    cout << endl << endl << "SOLUTION NOT FOUND" << endl << endl << endl;
    return 1;
  }

  if (attemptsparse) {
    if (verbose) cout << "sparsifying solution..." << endl;

    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_discrete;
    options.function_tolerance = ftol_discrete;
    options.linear_solver_type = DENSE_NORMAL_CHOLESKY;

    int successes = 0;
    greedy_discrete(problem,x,options,Problem::EvaluateOptions(),successes,DA_ZERO,N);
    greedy_discrete(problem,x,options,Problem::EvaluateOptions(),successes,DA_PM_ONE,N);
    logsol(x,"out.txt");
  }
  return 0;
}
