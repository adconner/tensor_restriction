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

const int num_relax = 50;
double alphastart = 0.01;
double sqalpha;
double solved = 1e-4;

class SolvedCallback : public IterationCallback {
  public:
    CallbackReturnType operator()(const IterationSummary& summary) {
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
pair<int,int> rational_approximation(double e, double eps = 1e-2) {
  bool minus = e < 0;
  if (minus) e = -e;
  vector<int> a;
  while (true) {
    a.push_back(int(e));
    e -= int(e);
    if (e < eps) break;
    e = 1/e;
  }
  int h=a[0],hp=1,k=1,kp=0;
  for (int i=1; i<a.size(); ++i) {
    tie(h,hp) = make_pair(h*a[i]+hp,h);
    tie(k,kp) = make_pair(k*a[i]+kp,k);
  }
  return make_pair(h,k);
}

double double_rational_approximation(double e, double eps = 1e-2) {
  int h,k; tie(h,k) = rational_approximation(e);
  return h / (double)k;
}

void solver_opts(Solver::Options &options) {
  /* options.minimizer_progress_to_stdout = true; */
  options.max_num_iterations = 200;
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
      if (!p.IsParameterBlockConstant(x+get<2>(vals[i]))) {
        vector<double> sav(x,x+n);
        x[get<2>(vals[i])] = get<1>(vals[i]);
        cout << "setting x[" << get<2>(vals[i]) << "] = " << x[get<2>(vals[i])] << "... ";
        cout.flush();
        p.SetParameterBlockConstant(x+get<2>(vals[i]));
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        if (summary.final_cost <= solved) {
          cout << "success" << endl;
          goto found;
        }
        cout << "fail" << endl << summary.BriefReport() << endl;
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
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> uni(0,1);
  generate_n(x,N,[&] {return uni(gen);});

  Problem problem;
  AddToProblem(problem,x);
  Problem::EvaluateOptions eopts;
  problem.GetResidualBlocks(&eopts.residual_blocks);
  // these are the residuals we care about

  for (int i=0; i<N; ++i) {
    problem.AddResidualBlock(new NoBorderRank, NULL, &x[i]);
  }

  Solver::Options options;
  auto solvedstop = make_unique<SolvedCallback>();
  options.callbacks.push_back(solvedstop.get());
  solver_opts(options);

  for (int i=num_relax; i>=0; --i) {
    sqalpha = std::sqrt(alphastart * i/(double) num_relax);
    Solver::Summary summary;
    cout << "forcing coefficient " << (sqalpha * sqalpha) << " cost ";
    cout.flush();
    Solve(options, &problem, &summary);
    /* cout << summary.BriefReport() << "\n"; */
    /* cout << summary.FullReport() << "\n"; */
    double cost; problem.Evaluate(eopts,&cost,0,0,0);
    cout << cost << endl;
  }
  sqalpha = 0;
  solved = 1e-9;
  options.max_num_iterations = 500;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  if (summary.final_cost > solved) {
    cout << "accuracy fail, not sparsifying" << endl;
    cout << summary.FullReport() << "\n";
  } else {
    cout << "solution seems good, sparsifying..." << endl;
    options.max_num_iterations = 25;
    greedy_discrete(options,problem,x,solved,DM_ZERO);
    greedy_discrete(options,problem,x,solved,DM_INTEGER);
    greedy_discrete(options,problem,x,solved,DM_RATIONAL);
  }

  ofstream out("out.txt");
  out.precision(numeric_limits<double>::max_digits10);
  copy(x,x+N,ostream_iterator<double>(out,"\n"));
  return 0;
}
