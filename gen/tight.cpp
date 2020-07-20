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

bool dfs(Problem &problem, double *x, const Solver::Options &options,
    set<int> &solved, ClpSimplex &model, int i);

bool solve_and_continue(Problem &problem, double *x, const Solver::Options &options,
    set<int> &solved, ClpSimplex &model, int i, bool cursolved) {
  vector<int> ixs;
  vector<double> ts;
  for (int j=0; j<TDIM; ++j) {
    if (tight[i][j] != 0) {
      ixs.push_back(j);
      ts.push_back(tight[i][j]);
    }
  }
  if (cursolved) {
    model.addRow(ixs.size(),ixs.data(),ts.data(),-COIN_DBL_MAX,0.0);
  } else {
    model.addRow(ixs.size(),ixs.data(),ts.data(),1.0,COIN_DBL_MAX);
  }

  model.primal();
  if (!model.isProvenOptimal()) {
    int row = model.getNumRows() - 1;
    model.deleteRows(1,&row);
    return false;
  }

  vector<int> solvedadded;
  vector<ResidualBlockId> eqs;
  if (cursolved) {
    solvedadded.push_back(i);
    solved.insert(i);
    eqs.push_back(AddToProblem(problem,x,i));
  }
  assert(solved.size() == problem.NumResiduals());

  for (int j=i+1; j<M; ++j) {
    if (!solved.count(j)) {
      ixs.clear(); ts.clear();
      for (int k=0; k<TDIM; ++k) {
        if (tight[j][k] != 0) {
          ixs.push_back(k);
          ts.push_back(tight[j][k]);
        }
      }
      model.addRow(ixs.size(),ixs.data(),ts.data(),1.0,COIN_DBL_MAX);
      model.primal();
      if (!model.isProvenOptimal()) {
        // cannot be positive, must try to solve
        solvedadded.push_back(j);
        solved.insert(j);
        eqs.push_back(AddToProblem(problem,x,j));
      }
      // no matter what, can delete row, either it is superfulous or not needed
      int row = model.getNumRows() - 1;
      model.deleteRows(1,&row);
    }
  }

  if (eqs.size()) {
    vector<double> sav(x,x+N*MULT);

    char *a = "";
    fill_initial(x,1,&a,problem);

    l2_reg_search(problem, x, options);
    /* Solver::Summary summary; */
    /* Solve(options, &problem, &summary); */

    double cost; problem.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0);
    if (cost < solved_fine) { // solve success
      cout << "tight: solve success" << endl;
      /* l2_reg_refine(problem,x,options); */
      if (dfs(problem,x,options,solved,model,i+1)) {
        return true;
      }
    } else {
      cout << "tight: solve fail" << endl;
    }
    copy(sav.begin(),sav.end(),x);

    for (auto rid : eqs) {
      problem.RemoveResidualBlock(rid);
    }
    for (auto j : solvedadded) {
      solved.erase(j);
    }
  } else {
    if (dfs(problem,x,options,solved,model,i+1)) {
      return true;
    }
  }
  int row = model.getNumRows() - 1;
  model.deleteRows(1,&row);
  return false;
}

bool dfs(Problem &problem, double *x, const Solver::Options &options,
    set<int> &solved, ClpSimplex &model, int i) {
  while (i < M && solved.count(i)) {
    i++;
  }
  if (i == M) {
    return true;
  }
  cout << "tight: solved "; 
  /* copy(solved.begin(),solved.end(),ostream_iterator<double>(cout," ")); */
  cout << solved.size();
  cout << " i: " << i;
  cout << endl;
  if (solve_and_continue(problem,x,options,solved,model,i,false)) {
    return true;
  }
  cout << "tight: return solved "; 
  /* copy(solved.begin(),solved.end(),ostream_iterator<double>(cout," ")); */
  cout << solved.size();
  cout << " i: " << i;
  cout << endl;
  if (solve_and_continue(problem,x,options,solved,model,i,true)) {
    return true;
  }
  return false;
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);

  double x[MULT*N];
  Problem::Options popts;
  popts.enable_fast_removal = true;
  Problem problem(popts);
  set<int> solved;
  /* for (int i=0; i<M; ++i) { */
  /*   if (all_of(tight[i],tight[i]+TDIM,[](int e){return e==0;})) { */
  /*     solved.insert(i); */
  /*     AddToProblem(problem,x,i); */
  /*   } */
  /* } */
  fill_initial(x,argc,argv,problem);

  Solver::Options options;
  solver_opts(options);
  options.callbacks.push_back(new SolvedCallback);
  if (verbose) {
    options.update_state_every_iteration = true;
    options.callbacks.push_back(new PrintCallback(x));
  }
  print_lines = verbose;

  /* l2_reg_search(problem, x, options); */
  /* Solver::Summary summary; */
  /* Solve(options, &problem, &summary); */

  ClpSimplex model; 
  model.setLogLevel(0);
  model.resize(0,TDIM);
  for (int i=0; i<TDIM; ++i) {
    model.setColumnLower(i,-COIN_DBL_MAX);
    model.setColumnUpper(i,COIN_DBL_MAX);
  }
  if (dfs(problem,x,options,solved,model,0)) {
    cout << endl << endl << "SOLUTION FOUND" << endl;
    cout << "solved: ";
    copy(solved.begin(),solved.end(),ostream_iterator<int>(cout," "));
    cout << endl << endl << endl;
    logsol(x,"out_dense.txt");
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
