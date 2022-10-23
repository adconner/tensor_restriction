#include <random>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include <tuple>
#include <algorithm>
#include "ceres/ceres.h"
#include "glog/logging.h"

#include "util/eqs.h"

bool restofile = true;

bool find_brank = true;
string brank_out = "outbr.txt";
double ftol_br = 1e-5;
double ptol_br;
double gtol_br;
int max_steps_br = 600;
double solved_thresh_br = 5e-3;

bool find_rank = false;
string rank_out = "outr.txt";
double ftol_r = 1e-3;
double ptol_r;
double gtol_r;
int max_steps_r = 150;
double solved_thresh_r = 1e-13;

using namespace ceres;
using namespace std;

void solver_opts(Solver::Options &options) {
  options.minimizer_progress_to_stdout = false;
  /* options.num_threads = 1; */

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

  if (max_steps_r) options.max_num_iterations = max_steps_br;
  if (ftol_r) options.function_tolerance = ftol_br;
  if (ptol_r) options.parameter_tolerance = ptol_br;
  if (gtol_r) options.gradient_tolerance = gtol_br;
}

// cheap corresponds to looking only for rank decomposition
void solver_opts_cheap(Solver::Options &options) {
  solver_opts(options);
  if (max_steps_r) options.max_num_iterations = max_steps_r;
  if (ftol_r) options.function_tolerance = ftol_r;
  if (ptol_r) options.parameter_tolerance = ptol_r;
  if (gtol_r) options.gradient_tolerance = gtol_r;
}

void logsol(const vector<F> &x, string fname) {
  ofstream out(fname);
  out.precision(numeric_limits<double>::max_digits10);
  for (F e : x) {
    out << e.real() << "\n" << e.imag() << "\n";
  }
}

struct Dat {
  Dat(F *_T, int _a, int _b, int _c, Problem::Options popts) : 
    problem(popts), T(_T), a(_a), b(_b), c(_c), r(-1) {}
  Problem problem;
  F *T;
  int a;
  int b; 
  int c;
  int r;
  vector<F> x;
};

void change_r(Dat &d, int r) {
  if (r != d.r) {
    d.r = r;
    MakeProblem(d.problem,d.x,d.T,d.a,d.b,d.c,r);
  }
}

normal_distribution<> dist(0,0.4);
mt19937 gen;
void seed_random() {
  typename mt19937::result_type seeds[mt19937::state_size];
  random_device rd;
  generate(begin(seeds), end(seeds), ref(rd));
  seed_seq ss(begin(seeds), end(seeds));
  gen.seed(ss);
}

enum TRY_RES { NEITHER, BRLOWER, RLOWER };
TRY_RES get_res(double fcost) {
  if (fcost < solved_thresh_r) return RLOWER;
  if (fcost < solved_thresh_br) return BRLOWER;
  return NEITHER;
}

TRY_RES try1(Dat &d, int r, bool cheap = false) {
  static Solver::Options opts, opts_c;
  static bool set_opts = true;
  if (set_opts) {
    solver_opts(opts);
    solver_opts_cheap(opts_c);
    set_opts = false;
  }
  change_r(d,r);
  generate_n(d.x.begin(),(d.a+d.b+d.c)*d.r,[&] {return F(dist(gen),dist(gen));});
  Solver::Summary summary;
  /* cout << d.problem.NumResidualBlocks() << endl; */
  Solve(cheap ? opts_c : opts, &d.problem, &summary);
  /* cout << summary.FullReport() << endl; */
  return get_res(summary.final_cost);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);
  seed_random();

  int a,b,c; cin >> a >> b >> c;
  vector<F> T(a*b*c);
  for (int i=0; i<T.size(); ++i) {
    cin >> T[i];
  }

  if (argc >= 2) {
    brank_out = string(argv[1]);
    find_brank = brank_out != "no";
  }
  if (argc >= 3) {
    rank_out = string(argv[2]);
    find_rank = rank_out != "no";
  }

  int brlower = argc >= 4 ? stoi(argv[3]) : 0;
  int rupper = argc >= 5 ? stoi(argv[4]) : min(min(a*b,a*c), b*c);

  Problem::Options popts;
  popts.enable_fast_removal = true;
  Dat d(T.data(),a,b,c,popts);

  vector<double> brp(rupper+1);
  vector<double> rp(rupper+1);
  for (int i=0; i < max(max(a,b),c); ++i) {
    rp[i] = brp[i] = 0.1;
  }
  brp[max(max(a,b),c)] = rp[max(max(a,b),c)] = 1.0;
  for (int i=max(max(a,b),c)+1; i <= rupper; ++i) {
    rp[i] = brp[i] = brp[i-1] * 0.8;
  }
  for (int i=0; i < brlower; ++i) {
    brp[i] = rp[i] = 0.0;
  }

  int brhi = rupper+1, rhi = rupper+1;
  vector<F> bestr, bestbr;
  double bestrcost, bestbrcost;

  auto normalize = [&] () {
    double brptot = accumulate(brp.begin(),brp.end(),0.0);
    for (int i = 0; i < brp.size(); ++i) {
      brp[i] /= brptot;
    }
    double rptot = accumulate(rp.begin(),rp.end(),0.0);
    for (int i = 0; i < rp.size(); ++i) {
      rp[i] /= rptot;
    }
  };

  auto update = [&](int r, TRY_RES res, bool quick) {
    assert((a+b+c)*r == d.x.size());
    if (res == NEITHER) {
      for (int i=0; i <= r; ++i) {
        rp[i] *= quick ? 0.6 : 0.4;
        brp[i] *= quick ? 0.7 : 0.5;
      }
    } else if (res == BRLOWER) {
      for (int i=0; i <= r; ++i) {
        rp[i] *= quick ? 0.8 : 0.6;
      }
      for (int i=r+1; i <= rupper; ++i) {
        brp[i] = 0.0;
      }
      if (r < brhi) {
        brhi = r;
        bestbr = d.x;
        Problem::EvaluateOptions eopts;
        d.problem.GetResidualBlocks(&eopts.residual_blocks);
        d.problem.Evaluate(eopts,&bestbrcost,0,0,0);
      }
    } else if (res == RLOWER) {
      for (int i=r+1; i <= rupper; ++i) {
        rp[i] = 0.0;
        brp[i] = 0.0;
      }
      if (r < rhi) {
        rhi = r;
        bestr = d.x;
        Problem::EvaluateOptions eopts;
        d.problem.GetResidualBlocks(&eopts.residual_blocks);
        d.problem.Evaluate(eopts,&bestrcost,0,0,0);
      }
      if (r < brhi) {
        brhi = r;
        bestbr = d.x;
        bestbrcost = bestrcost;
      }
    }
    normalize();
  };

  auto nextbrcheck = [&]() {
    double p = brp[0];
    for (int r=1; r <= rupper; ++r) {
      double pcur = p + brp[r];
      if (pcur >= 0.5) {
        if (r == brhi) {
          return make_pair(r-1, p);
        } else {
          return make_pair(r, 1.0);
        }
      }
      p = pcur;
    }
    assert(false);
  };

  auto nextrcheck = [&]() {
    double thresh = accumulate(rp.begin(),rp.end(),0.0) / 2;
    double p = rp[0];
    for (int r=1; r <= rupper; ++r) {
      double pcur = p + rp[r];
      if (pcur >= 0.5) {
        if (r == rhi) {
          return make_pair(r-1, p);
        } else {
          return make_pair(r, 1.0);
        }
      }
      p = pcur;
    }
    assert(false);
  };

  if (find_brank) {
    auto [r,p] = nextbrcheck();
    while (p > 0.05) {
      printf("border rank try %d %g\n", r,p);
      copy(brp.begin(),brp.end(),ostream_iterator<double>(cout," ")); cout << endl;
      auto res = try1(d,r,false);
      update(r,res,false);
      tie(r,p) = nextbrcheck();
    }
    if (brhi <= rupper) {
      printf("border rank %d\n",brhi);
      if (restofile) {
        logsol(bestbr,brank_out);
      }
    }
  }

  if (find_rank) {
    auto [r,p] = nextrcheck();
    while (p > 1e-2) {
      printf("rank try %d %g\n", r,p);
      copy(rp.begin(),rp.end(),ostream_iterator<double>(cout," ")); cout << endl;
      auto res = try1(d,r,true);
      update(r,res,true);
      tie(r,p) = nextrcheck();
    }
    if (rhi <= rupper) {
      printf("rank %d\n",rhi);
      if (restofile) {
        logsol(bestr,rank_out);
      }
    }
  }

  return 0;
}
