#include <random>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include <complex>
#include <tuple>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "prob.h"

using namespace ceres;
using namespace std;
typedef complex<double> cx;

const bool verbose = true;
const bool tostdout = false;
const bool attemptsparse = true; 
const bool l2_reg_always = false;
const bool l2_reg_random_start = true;
const bool record_iterations = false;
const bool log_rough = !tostdout;

const double ftol = 1e-13;
const double gtol = 1e-30;
const double ptol = 1e-30;

const int l2_reg_steps = 3;
const double l2_reg_decay = 0.60;
const double alphastart = 0.01;
const double ftol_rough = 1e-3;
const double abort_worse = 1e-3;

const double solved_fine = 1e-25;
const double attempt_sparse_thresh = 1e-12;

// parameters for finding an initial solution
const int iterations_rough = 200;
const int iterations_fine = 2500;

// parameters for descretization
const int iterations_discrete = 60;
const double ftol_discrete = 0.4; // ~ solved_fine ^ (1/iterations_discrete)

// control variables
double sqalpha; // square root of l2 regularization coefficient
bool print_lines;

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

class SolvedCallback : public IterationCallback {
  public:
    CallbackReturnType operator()(const IterationSummary& summary) {
      return summary.cost < 1e-29 ? SOLVER_TERMINATE_SUCCESSFULLY : SOLVER_CONTINUE;
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

class PrintCallback : public IterationCallback {
  public:
    PrintCallback(double *_x) : x(_x) {}
    CallbackReturnType operator()(const IterationSummary& summary) {
      if (print_lines) {
        double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) {return max(std::abs(a),std::abs(b));} ); 
        cout << summary.iteration << " " << summary.cost << " " << ma <<
          " " << summary.relative_decrease << endl;
        /* if (ma > 4 && summary.iteration >= 10) */
        /*   return SOLVER_ABORT; */
      }
      return SOLVER_CONTINUE;
    }
    double *x;
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

class Equal : public SizedCostFunction<MULT,MULT> {
  public:
    Equal(double _a, cx _x0) : a(_a), x0(_x0) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = a*(x[0][0]-x0.real());
      if (MULT == 2) residuals[1] = a*(x[0][1]-x0.imag());
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
    cx x0;
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

enum DiscreteAttempt {
  DA_ZERO,
  DA_PM_ONE,
  DA_PM_ONE_ZERO,
  DA_INTEGER };
void greedy_discrete(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    int &successes, DiscreteAttempt da = DA_ZERO, const int faillimit = -1) {
  vector<int> counts(N);
  /* const double fail_penalty = 0.05; */
  const double fail_penalty = 0.2;
  while (true) {
    vector<tuple<double,cx,int> > vals(N);
    for (int i=0; i<N; ++i) {
      cx target;
      switch (da) {
        case DA_ZERO: target = 0.0; break;
        case DA_PM_ONE: target = x[i*MULT] >= 0 ? 1.0 : -1.0; break;
        case DA_PM_ONE_ZERO: target = std::abs(x[i*MULT]) < 1e-2 ? 0.0 : (x[i*MULT] >= 0 ? 1.0 : -1.0); break;
        case DA_INTEGER: target = std::round(x[i*MULT]); break;
      }
      cx cur = MULT == 1 ? cx(x[i]) : cx(x[i*MULT],x[i*MULT+1]);
      get<0>(vals[i]) = std::abs(cur - target) + fail_penalty * counts[i];
      get<1>(vals[i]) = target;
      get<2>(vals[i]) = i;
    }
    sort(vals.begin(),vals.end(),[](const auto &a,const auto &b) {
        return get<0>(a) < get<0>(b);
    });
    int fails = faillimit;
    vector<double> sav(x,x+N*MULT);
    for (int i=0; i<N; ++i) {
      if (!p.IsParameterBlockConstant(x+get<2>(vals[i]))) {
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          cout << "successes " << successes
            << " rem " << (faillimit == -1 ? N-i : fails)
            << " lfails " << counts[get<2>(vals[i])]
            << " setting " << "x[" << get<2>(vals[i]) << "] = "
            << get<1>(vals[i]).real();
          if (MULT == 2) cout << ", x[" << get<2>(vals[i]) + 1 << "] = "
            << get<1>(vals[i]).imag();
          cout << "...";
          cout.flush();
        }
        x[get<2>(vals[i])*MULT] = get<1>(vals[i]).real();
        if (MULT == 2) x[get<2>(vals[i])*MULT + 1] = get<1>(vals[i]).imag();
        p.SetParameterBlockConstant(x+get<2>(vals[i]));
        double scost; p.Evaluate(eopts,&scost,0,0,0);
        if (scost < std::max(icost,solved_fine)) {
          if (verbose) cout << " success free " << endl;
          successes++;
          goto found;
        } else {
          Solver::Summary summary;
          Solve(opts,&p,&summary);
          if (summary.final_cost <= std::max(icost,solved_fine)) { // improved or good enough
            if (verbose) cout << " success " << summary.iterations.size() - 1
                << " iterations" << endl;
            logsol(x,"out_partial_sparse.txt");
            successes++;
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
              << summary.final_cost << endl;
          counts[get<2>(vals[i])]++;
          p.SetParameterBlockVariable(x+get<2>(vals[i]));
          copy(sav.begin(),sav.end(),x);
          if (faillimit > 0 && fails-- == 0) break;
        }
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
    vector<double> sav(x,x+N*MULT);
    for (int i=0; i<vals.size(); ++i) {
      if (!p.IsParameterBlockConstant(x+vals[i].second.first) &&
          !p.IsParameterBlockConstant(x+vals[i].second.second) && 
          !fixed.count(vals[i].second)) {
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          cout << "cost " << icost << " attempting to set "; 
          cout << "x[" << vals[i].second.first << "] = x[" 
            << vals[i].second.second << "]...";
          cout.flush();
        }
        auto rid = p.AddResidualBlock(new LinearCombination(1.0,-1.0), NULL, 
            {x+vals[i].second.first, x+vals[i].second.second});
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        if (summary.final_cost <= std::max(icost,solved_fine)) {
          fixed.insert(vals[i].second);
          if (verbose) cout << " success " << summary.iterations.size() - 1
              << " iterations" << endl;
          logsol(x,"out_partial_sparse.txt");
          goto found;
        }
        if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
          << summary.final_cost << endl;
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
  cout.precision(numeric_limits<double>::digits10);

  double x[MULT*N];
  if (argc == 1) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0,0.4);
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
    greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N/2+1);
    greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE_ZERO,N/10+1);
    for (int refine=1; refine<=2; ++refine) {
      options.max_num_iterations *= 2;
      options.function_tolerance *= options.function_tolerance;
      greedy_discrete(problem,x,options,eopts,successes,DA_ZERO,N/10+1);
      greedy_discrete(problem,x,options,eopts,successes,DA_PM_ONE_ZERO,N/10+1);
    }
    /* greedy_discrete_pairs(problem,x,options,eopts,10); */

    logsol(x,"out.txt");
  }
  return 0;
}
