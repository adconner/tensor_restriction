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

const bool verbose = true;
const bool tostdout = false;
const bool attemptsparse = true; 
const bool force = true;

const double ftol = 1e-13;
const double gtol = 1e-13;
const double ptol = 1e-13;

const int num_relax = 30;
const double alphastart = 0.02;
const double ftol_rough = 1e-4;
const double abort_worse = 1e-3;

const double solved_fine = 1e-25;
const double attempt_sparse_thresh = 1e-5;

// parameters for finding an initial solution
const int iterations_rough = 200;
const int iterations_fine = 2500;

// parameters for descretization
const double checkpoint = 0.5;
const int iterations_checkpoint = 5;
const int iterations_tiny = 60;

// control variables
double sqalpha; // square root of forcing coeffient
double solved; // value to stop optimization at if reached
double checkpoint_ok; // stop iteration if have not converged to at least here
double checkpoint_iter; // checkpoint iteration number

void solver_opts(Solver::Options &options) {
  options.num_threads = 1;
  options.num_linear_solver_threads = 1;

  /* options.minimizer_type = LINE_SEARCH; */
  // trust region options
  options.trust_region_strategy_type = LEVENBERG_MARQUARDT;
  options.use_nonmonotonic_steps = true;
  options.use_inner_iterations = true;

  // line search options
  /* options.line_search_direction_type = BFGS; */
  /* options.line_search_type = ARMIJO; */
  /* options.nonlinear_conjugate_gradient_type = POLAK_RIBIERE; */
  /* options.nonlinear_conjugate_gradient_type = HESTENES_STIEFEL; */

  // linear solver options
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  /* options.dynamic_sparsity = true; // since solutions are typically sparse? */
  /* options.use_postordering = true; */

  /* options.linear_solver_type = DENSE_QR; */
  /* options.linear_solver_type = CGNR; */
  /* options.linear_solver_type = ITERATIVE_SCHUR; */

  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.dense_linear_algebra_library_type = LAPACK;

  options.function_tolerance = ftol;
  options.parameter_tolerance = ptol;
  options.gradient_tolerance = gtol;
}

class SolvedCallback : public IterationCallback {
  public:
    CallbackReturnType operator()(const IterationSummary& summary) {
      if (checkpoint_iter >= 0 
          && summary.iteration >= checkpoint_iter 
          && summary.cost > checkpoint_ok) return SOLVER_ABORT;
      return summary.cost < solved ? SOLVER_TERMINATE_SUCCESSFULLY : SOLVER_CONTINUE;
    }
};

class NoBorderRank : public SizedCostFunction<MULT,MULT> {
  public:
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = sqalpha * x[0][0];
      if (MULT == 2) residuals[1] = sqalpha * x[0][1];
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = sqalpha;
          if (MULT == 2) {
            jacobians[0][1] = 0;
            jacobians[0][2] = 0;
            jacobians[0][3] = sqalpha;
          }
        }
      }
      return true;
    }
};

class Zero : public SizedCostFunction<MULT,MULT> {
  public:
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = x[0][0];
      if (MULT == 2) residuals[1] = x[0][1];
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = 1;
          if (MULT == 2) {
            jacobians[0][1] = 0;
            jacobians[0][2] = 0;
            jacobians[0][3] = 1;
          }
        }
      }
      return true;
    }
};

void greedy_discrete(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    const int faillimit = -1) {
  while (true) {
    vector<pair<double,int> > vals(N);
    for (int i=0; i<N; ++i) {
      for (int j=0; j<MULT; ++j) {
        vals[i].first += x[i*MULT + j] * x[i*MULT+j];
      }
      vals[i].second = i*MULT;
    }
    sort(vals.begin(),vals.end());
    int fails = faillimit;
    for (int i=0; i<N; ++i) {
      if (!p.IsParameterBlockConstant(x+vals[i].second)) {
        vector<double> sav(x,x+N*MULT);
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          cout << "cost " << icost << " attempting to zero "; 
          cout << "x[" << vals[i].second << "]";
          if (MULT == 2) cout << ", x[" << vals[i].second + 1 << "]";
          cout << "...";
          cout.flush();
        }
        auto rid = p.AddResidualBlock(new Zero, NULL, x+vals[i].second);
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        p.RemoveResidualBlock(rid);
        if (summary.final_cost <= std::max(icost,solved)) {
          for (int j=0; j<MULT; ++j) x[vals[i].second+j] = 0;
          p.SetParameterBlockConstant(x+vals[i].second);
          Solve(opts,&p,&summary);
          cout << " success" << endl;
          goto found;
        }
        if (verbose) cout << " fail" << endl;
        p.SetParameterBlockVariable(x+vals[i].second);
        copy(sav.begin(),sav.end(),x);
        if (faillimit > 0 && fails-- == 0) break;
      }
    }
    break;
    found:;
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double x[MULT*N];
  if (argc == 1) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> uni(0,1);
    generate_n(x,MULT*N,[&] {return uni(gen);});
  } else {
    ifstream in(argv[1]);
    for (int i=0; i<MULT*N; ++i)
      in >> x[i];
  }

  Problem problem;
  ceres::ParameterBlockOrdering pbo;
  AddToProblem(problem,pbo,x);
  Problem::EvaluateOptions eopts;
  problem.GetResidualBlocks(&eopts.residual_blocks);
  // save residuals we care about before adding other regularization

  Solver::Options options; solver_opts(options);
  auto solvedstop = make_unique<SolvedCallback>();
  options.callbacks.push_back(solvedstop.get());

  if (force) {
    vector<ResidualBlockId> rids;
    for (int i=0; i<N; ++i) {
      rids.push_back(problem.AddResidualBlock(new NoBorderRank, NULL, &x[MULT*i]));
    }
    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_rough;
    options.function_tolerance = ftol_rough;
    checkpoint_iter = -1;
    /* options.minimizer_progress_to_stdout=true; */
    // dogleg for positive alpha?
    for (int i=num_relax; i>=0; --i) {
      sqalpha = std::sqrt(alphastart * i/(double) num_relax);
      Solver::Summary summary;
      if (verbose) {
        cout << "forcing coefficient " << (sqalpha * sqalpha) << " cost ";
        cout.flush();
      }
      Solve(options, &problem, &summary);
      /* if (verbose) cout << summary.FullReport() << "\n"; */
      double cost; problem.Evaluate(eopts,&cost,0,0,0);
      if (verbose) cout << cost << endl;
    }
    for (auto rid : rids) {
      problem.RemoveResidualBlock(rid);
    }
  }
  options.function_tolerance = ftol;
  double cost; problem.Evaluate(eopts,&cost,0,0,0);
  if (cost > abort_worse) {
    if (verbose) cout << "cost " << cost << " not good enough to refine. Aborting" << endl;
    return 2;
  }

  options.minimizer_type = TRUST_REGION;
  options.max_num_iterations = iterations_fine;
  solved = solved_fine;
  checkpoint_iter = -1;
  if (verbose) options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  /* cout << summary.FullReport() << endl; */

  if (tostdout) {
    cout.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(cout," "));
    cout << endl;
  } else {
    ofstream out("out_dense.txt");
    out.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(out,"\n"));
  }

  if (summary.final_cost > attempt_sparse_thresh) {
    if (verbose) {
      cout << "accuracy fail, not sparsifying" << endl;
      cout << summary.FullReport() << "\n";
    }
    return 1;
  }
  if (!attemptsparse) return 0;
  if (verbose) cout << "solution seems good, sparsifying..." << endl;
  options.minimizer_progress_to_stdout = false;
  options.minimizer_type = TRUST_REGION;
  options.max_num_iterations = iterations_rough;
  checkpoint_iter = iterations_checkpoint;
  checkpoint_ok = checkpoint;

  greedy_discrete(problem,x,options,eopts,10);

  if (tostdout) {
    cout.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(cout," "));
    cout << endl;
  } else {
    ofstream out("out.txt");
    out.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(out,"\n"));
  }
  return 0;
}
