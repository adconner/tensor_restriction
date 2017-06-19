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

const double ftol = 1e-13;
const double gtol = 1e-13;
const double ptol = 1e-13;

const int num_relax = 30;
const double alphastart = 0.02;
const double ftol_rough = 1e-4;

const double solved_fine = 1e-7;
const double attempt_sparse_thresh = 5e-5;

// parameters for finding an initial solution
const int iterations_trust_rough = 200;
const int iterations_trust_fine = 2500;
const int iterations_line_rough = 700;
const int iterations_line_fine = 10000;

// parameters for descretization
const double checkpoint = 0.5;
const int iterations_trust_checkpoint = 5;
const int iterations_line_checkpoint = 30;
const int iterations_trust_tiny = 30;
const int iterations_line_tiny = 100;

// control variables
double sqalpha; // square root of forcing coeffient
double solved; // value to stop optimization at if reached
double checkpoint_ok; // stop iteration if have not converged to at least here
double checkpoint_iter; // checkpoint iteration number

void solver_opts(Solver::Options &options) {
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;

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
  options.dynamic_sparsity = true; // since solutions are typically sparse?
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

class NoBorderRank : public SizedCostFunction<1,1> {
  public:
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = sqalpha * x[0][0];
      if (jacobians && jacobians[0]) {
        jacobians[0][0] = sqalpha;
      }
      return true;
    }
};

// choose a pretty coarse rational approximation by default
pair<int,int> rational_approximation(double e, double eps) {
  double arg = e;
  bool minus = e < 0;
  if (minus) e = -e;
  int h=int(e),hp=1,k=1,kp=0;
  e -= int(e); e = 1/e;
  while (std::abs(arg-h/(double)k) > eps) {
    int ai = int(e);
    e -= ai; e = 1/e;
    tie(h,hp) = make_pair(h*ai+hp,h);
    tie(k,kp) = make_pair(k*ai+kp,k);
  }
  return make_pair(minus ? -h : h,k);
}

double double_rational_approximation(double e, double eps = 1e-2) {
  int h,k; tie(h,k) = rational_approximation(e,eps);
  return h / (double)k;
}

enum DiscreteMode {
  DM_ZERO, DM_INTEGER, DM_RATIONAL
};

void greedy_discrete(const Solver::Options & opts, Problem &p, double *x, 
    double solved = 1e-2, DiscreteMode dm=DM_ZERO) {
  const int n = p.NumParameters();
  while (true) {
    vector<tuple<double,double,int> > vals(n);
    for (int i=0; i<n; ++i) {
      double approx;
      switch (dm) {
        case DM_ZERO: approx = 0; break;
        case DM_INTEGER: approx = std::round(x[i]); break;
        case DM_RATIONAL: approx = double_rational_approximation(x[i]); break;
      }
      vals[i] = make_tuple(std::abs(x[i]-approx),approx,i);
    }
    sort(vals.begin(),vals.end());
    for (int i=0; i<n; ++i) {
      if (dm != DM_ZERO && get<0>(vals[i]) > 0.5) break;
      if (!p.IsParameterBlockConstant(x+get<2>(vals[i]))) {
        vector<double> sav(x,x+n);
        x[get<2>(vals[i])] = get<1>(vals[i]);
        if (verbose) {
          cout << "setting x[" << get<2>(vals[i]) << "] = " 
            << x[get<2>(vals[i])] << "... ";
          cout.flush();
        }
        p.SetParameterBlockConstant(x+get<2>(vals[i]));
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        if (summary.final_cost <= attempt_sparse_thresh) {
          if (verbose) cout << "success" << endl;
          goto found;
        }
        /* cout << "fail" << endl << summary.BriefReport() << endl; */
        if (verbose) cout << "fail" << endl;
        p.SetParameterBlockVariable(x+get<2>(vals[i]));
        copy(sav.begin(),sav.end(),x);
      }
    }
    break;
    found:;
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double x[N];
  if (argc == 1) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> uni(0,1);
    generate_n(x,N,[&] {return uni(gen);});
  } else {
    ifstream in(argv[1]);
    for (int i=0; i<N; ++i)
      in >> x[i];
  }

  Problem problem;
  AddToProblem(problem,x);
  Problem::EvaluateOptions eopts;
  problem.GetResidualBlocks(&eopts.residual_blocks);
  // save residuals we care about before adding other regularization

  Solver::Options options; solver_opts(options);
  auto solvedstop = make_unique<SolvedCallback>();
  options.callbacks.push_back(solvedstop.get());

  if (argc == 1) {
    for (int i=0; i<N; ++i) {
      problem.AddResidualBlock(new NoBorderRank, NULL, &x[i]);
    }
    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_trust_rough;
    options.function_tolerance = ftol_rough;
    checkpoint_iter = -1;
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
  }
  sqalpha = 0; // TODO should remove forcing terms?
  options.function_tolerance = ftol;

  // should also cancel if cost not small enough

  options.minimizer_type = TRUST_REGION;
  options.max_num_iterations = iterations_trust_fine;
  solved = solved_fine;
  checkpoint_iter = -1;
  if (verbose) options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  options.minimizer_type = LINE_SEARCH;
  options.max_num_iterations = iterations_line_fine;
  Solve(options, &problem, &summary);

  {
    ofstream out("out_dense.txt");
    out.precision(numeric_limits<double>::max_digits10);
    copy(x,x+N,ostream_iterator<double>(out,"\n"));
  }

  if (summary.final_cost > attempt_sparse_thresh) {
    cout << "accuracy fail, not sparsifying" << endl;
    cout << summary.FullReport() << "\n";
    return 0;
  }

  if (verbose) cout << "solution seems good, sparsifying..." << endl;
  options.minimizer_progress_to_stdout = false;

  options.minimizer_type = LINE_SEARCH;
  options.max_num_iterations = iterations_line_tiny;
  checkpoint_iter = iterations_line_checkpoint;
  checkpoint_ok = checkpoint;

  /* options.minimizer_type = TRUST_REGION; */
  /* options.max_num_iterations = iterations_trust_tiny; */
  /* checkpoint_iter = iterations_trust_checkpoint; */
  /* checkpoint_ok = checkpoint; */

  greedy_discrete(options,problem,x,solved,DM_ZERO);
  greedy_discrete(options,problem,x,solved,DM_INTEGER);
  greedy_discrete(options,problem,x,solved,DM_RATIONAL);

  ofstream out("out.txt");
  out.precision(numeric_limits<double>::max_digits10);
  copy(x,x+N,ostream_iterator<double>(out,"\n"));
  return 0;
}
