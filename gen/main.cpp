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
const bool l2_reg_always = false;
const bool l2_reg_random_start = true;
const bool record_iterations = false;

const double ftol = 1e-13;
const double gtol = 1e-13;
const double ptol = 1e-13;

const int l2_reg_steps = 30;
const double l2_reg_decay = 0.95;
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

class RecordCallback : public IterationCallback {
  public:
    RecordCallback(double *_x, ostream &_out) : x(_x), out(_out) {}
    CallbackReturnType operator()(const IterationSummary& summary) {
      copy(x,x+MULT*N,ostream_iterator<double>(out," ")); out << endl;
      return SOLVER_CONTINUE;
    }
    double *x;
    ostream &out;
};

class L2Regularization : public SizedCostFunction<MULT,MULT> {
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
    Zero(double _a) : a(_a) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = a*x[0][0];
      if (MULT == 2) residuals[1] = a*x[0][1];
      if (jacobians && jacobians[0]) {
        jacobians[0][0] = a;
        if (MULT == 2) {
          jacobians[0][1] = 0;
          jacobians[0][2] = 0;
          jacobians[0][3] = a;
        }
      }
      return true;
    }
    double a;
};

void logsol(double *x, string fname) {
  if (tostdout) {
    cout.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(cout," "));
    cout << endl;
  } else {
    ofstream out(fname);
    out.precision(numeric_limits<double>::max_digits10);
    copy(x,x+MULT*N,ostream_iterator<double>(out,"\n"));
  }
}

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
        auto rid = p.AddResidualBlock(new Zero(1), NULL, x+vals[i].second);
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        p.RemoveResidualBlock(rid);
        if (summary.final_cost <= std::max(icost,solved)) {
          for (int j=0; j<MULT; ++j) x[vals[i].second+j] = 0;
          p.SetParameterBlockConstant(x+vals[i].second);
          Solve(opts,&p,&summary);
          cout << " success" << endl;
          logsol(x,"out_partial.txt");
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

class LinearCombination : public SizedCostFunction<MULT,MULT,MULT> {
  public:
    LinearCombination(double _a, double _b) : a(_a), b(_b) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = a*x[0][0] + b*x[1][0];
      if (MULT == 2) residuals[1] = a*x[0][1] + b*x[1][1];
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = a;
          if (MULT == 2) {
            jacobians[0][1] = 0;
            jacobians[0][2] = 0;
            jacobians[0][3] = a;
          }
        }
        if (jacobians[1]) {
          jacobians[1][0] = b;
          if (MULT == 2) {
            jacobians[1][1] = 0;
            jacobians[1][2] = 0;
            jacobians[1][3] = b;
          }
        }
      }
      return true;
    }
    double a,b;
};

void greedy_discrete_pairs(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    const int faillimit = -1) {
  set<pair<int,int> > fixed;
  while (true) {
    vector<pair<double,pair<int,int> > > vals(N * (N-1) / 2);
    int ix = 0;
    for (int i=0; i<N; ++i) {
      for (int j=i+1; j<N; ++j) {
        vals[ix].second.first = i*MULT;
        vals[ix].second.second = j*MULT;
        for (int k=0; k<MULT; ++k) {
          double dx = x[i*MULT + k] - x[j*MULT + k];
          vals[ix].first += dx*dx;
        }
        ix++;
      }
    }
    sort(vals.begin(),vals.end());
    int fails = faillimit;
    for (int i=0; i<vals.size(); ++i) {
      if (!p.IsParameterBlockConstant(x+vals[i].second.first) &&
          !p.IsParameterBlockConstant(x+vals[i].second.second) && 
          !fixed.count(vals[i].second)) {
        vector<double> sav(x,x+N*MULT);
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          cout << "cost " << icost << " attempting to set "; 
          cout << "x[" << vals[i].second.first << "] = x[" 
            << vals[i].second.second << "]...";
          cout.flush();
        }
        auto rid = p.AddResidualBlock(new LinearCombination(1,-1), NULL, 
            {x+vals[i].second.first, x+vals[i].second.second});
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        if (summary.final_cost <= std::max(icost,solved)) {
          fixed.insert(vals[i].second);
          cout << " success" << endl;
          logsol(x,"out_partial.txt");
          goto found;
        }
        if (verbose) cout << " fail" << endl;
        p.RemoveResidualBlock(rid);
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
    normal_distribution<> dist(0,1);
    generate_n(x,MULT*N,[&] {return dist(gen);});
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

  if (l2_reg_always || (l2_reg_random_start && argc == 1)) {
    vector<ResidualBlockId> rids;
    for (int i=0; i<N; ++i) {
      rids.push_back(problem.AddResidualBlock(new L2Regularization, NULL, &x[MULT*i]));
    }
    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = iterations_rough;
    options.function_tolerance = ftol_rough;
    if (verbose) options.minimizer_progress_to_stdout = true;
    checkpoint_iter = -1;
    /* options.minimizer_progress_to_stdout=true; */
    // dogleg for positive alpha?
    sqalpha = std::sqrt(alphastart);
    for (int i=l2_reg_steps; i>=0; --i, sqalpha *= std::sqrt(l2_reg_decay)) {
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
    cout << "rough solving..." << endl;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    options.function_tolerance = ftol;
    double cost; problem.Evaluate(eopts,&cost,0,0,0);
    if (cost > abort_worse) {
      if (verbose) cout << summary.FullReport() << endl << 
        "cost " << cost << " not good enough to refine. Aborting" << endl;
      return 2;
    }
  }

  options.minimizer_type = TRUST_REGION;
  options.max_num_iterations = iterations_fine;
  solved = solved_fine;
  checkpoint_iter = -1;
  if (verbose) options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  if (record_iterations) {
    options.update_state_every_iteration = true;
    auto record = make_unique<RecordCallback>(x,cerr);
    options.callbacks.push_back(record.get());
    Solve(options, &problem, &summary);
    options.callbacks.pop_back();
    options.update_state_every_iteration = false;
  } else {
    Solve(options, &problem, &summary);
  }
  /* cout << summary.FullReport() << endl; */

  logsol(x,"out_dense.txt");

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
  /* options.max_num_iterations = iterations_tiny; */
  checkpoint_iter = iterations_checkpoint;
  checkpoint_ok = checkpoint;

  greedy_discrete(problem,x,options,eopts,10);
  greedy_discrete_pairs(problem,x,options,eopts,10);

  logsol(x,"out.txt");

  return 0;
}
