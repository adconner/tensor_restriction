#include <random>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include <tuple>
#include "ceres/ceres.h"
#include "glog/logging.h"

#include "../opts.h"
#include "../util.h"
#include "../initial.h"
#include "../l2reg.h"
#include "../problem.h"
#include "../discrete.h"
#include "../cholmod.h"

#include "../restrict/als.h"
#include "../restrict/tensor.h"


using namespace ceres;
using namespace std;

int main(int argc, const char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);
  random_device rd;
  rng.seed(rd());
  /* rng.seed(8); */
  
  int a,b,c; cin >> TA >> TB >> TC;

  T.assign(TA,vector<vector<F> >(TB, vector<F>(TC)));
  for (int i=0; i<TA; ++i) {
    for (int j=0; j<TB; ++j) {
      for (int k=0; k<TC; ++k) {
        cin >> T[i][j][k];
      }
    }
  }
  
  assert(argc >= 2);
  int r = stoi(argv[1]);
  set_rank_r(r);
  set_params();

  Problem::Options popts;
  popts.enable_fast_removal = true;
  MyProblem p(popts,N);

  bool tight = false;
  vector<tuple<int,int,int> > eqs;
  for (int i=0; i<TA; ++i) {
    for (int j=0; j<TB; ++j) {
      for (int k=0; k<TC; ++k) {
        if (!SYM || (i <= j && j <= k)) {
          if (TIGHTA[i] + TIGHTB[j] + TIGHTC[k] <= 0) {
            eqs.push_back(make_tuple(i,j,k));
            AddToProblem(p.p,p.x.data(),(i*TB+j)*TC+k);
          } else {
            tight = true;
          }
        }
      }
    }
  }

  fill_initial(p,"");

  // pure als followed by trust region refine
  double costlast=1e7;
  int bad = 0;
  double sqalpha = 0.1;
  for (int it=0; it<2000; ++it) {
    if (SYM) {
      if (!tight) {
        als_sym(p.x.data(),sqalpha);
      } else {
        als_sym_some(p.x.data(),eqs,sqalpha);
      }
    } else {
      if (!tight) {
        als(p.x.data(),it%3,sqalpha);
      } else {
        als_some(p.x.data(),eqs,it%3,sqalpha);
      }
    }
    double cost; p.p.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0);
    double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
        {return max(a,std::abs(b));} ); 
    double l2 = sqrt(accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
        {return a+b*b;} )); 
    if ((costlast - cost) / costlast < 2e-4) {
      bad++;
    } else {
      bad=0;
    }
    printf("%4d %20.15g %20.15g %20.15g %10.5g %d %5.1e\n",it,2*cost,ma,l2,sqalpha,bad,
        (costlast-cost)/costlast );
    if (cost < 1e-27)
      break;
    if (bad >= 3) {
      if (sqalpha == 0.0 && bad >= 1000) {
        break;
      } else if (sqalpha > 0.0 && sqalpha < 1e-4) {
        sqalpha = 0.0;
        bad = 0;
      } else if (sqalpha > 0.0) {
        sqalpha *= 0.9;
        bad = 0;
      }
    }
    costlast = cost;
  }
  double cost; p.p.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0);
  printf("%20.15g\n",2*cost);
  logsol(p,"out_dense.txt");
  /* if (2*cost > 1.0) { */
  /*   return 1; */
  /* } */

  /* trust_region_f(p,[&](){ */
  /*     for (int it=0; it<3*4; ++it) { */
  /*       /1* als_some(p.x.data(),eqs,it%3,1e-3); *1/ */
  /*       als_sym(p.x.data()); */
  /*       /1* als_sym_some(p.x.data(),eqs); *1/ */
  /*       /1* als(p.x.data(),it%3); *1/ */
  /*     } */
  /*     },1e-7,300); */
  /* logsol(p,"out_dense.txt"); */

  /* // trust region refine/als with l2 regularization curretly trust region */
  /* double costlast=1e7; */
  /* int bad = 0; */
  /* double sqalpha = 0.1; */
  /* /1* for (int it=0; it<2000; ++it) { *1/ */
  /* int it=0; */ 
  /* trust_region_f(p,[&](){ */
  /*   for (int i=0; i<3*4; ++i) { */
  /*     als_some(p.x.data(),eqs,i%3); */
  /*   } */
  /*   double cost; p.p.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0); */
  /*   double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) */ 
  /*       {return max(a,std::abs(b));} ); */ 
  /*   double l2 = sqrt(accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) */ 
  /*       {return a+b*b;} )); */ 
  /*   if ((costlast - cost) / costlast < 1e-3) { */
  /*     bad++; */
  /*   } else { */
  /*     bad=0; */
  /*   } */
  /*   printf("%4d %20.15g %20.15g %20.15g %10.5g %d\n",it,2*cost,ma,l2,sqalpha,bad); */
  /*   if (bad >= 3) { */
  /*     if (sqalpha < 1e-4) { */
  /*       sqalpha = 0.0; */
  /*     } else { */
  /*       sqalpha *= 0.7; */
  /*     } */
  /*     bad = 0; */
  /*   } */
  /*   costlast = cost; */
  /*   it++; */
  /* },1e-7,1000); */


  /* /1* int r = 21, R = 23; *1/ */
  /* /1* for (int i=0; i<N/R; ++i) { *1/ */
  /* /1*   for (int j=r; j<R; ++j) { *1/ */
  /* /1*     int k=i*R+j; *1/ */
  /* /1*     for (int l=0; l<MULT; ++l) { *1/ */
  /* /1*       p.x[MULT*k+l] = 0.0; *1/ */
  /* /1*     } *1/ */
  /* /1*     set_value_constant(p,k); *1/ */
  /* /1*   } *1/ */
  /* /1* } *1/ */

  MyTerminationType term = NO_SOLUTION;

  /* for (int bi=0; bi < BLOCKS; ++bi) { */
  /*   for (int i=0; i< MULT*(BBOUND[i+1] - BBOUND[i]); ++i) { */
  /*     p.p.SetParameterLowerBound(p.x.data()+MULT*BBOUND[bi],i,-2.0); */
  /*     p.p.SetParameterUpperBound(p.x.data()+MULT*BBOUND[bi],i,2.0); */
  /*   } */
  /* } */
  /* term = l2_reg_search(p, 1e-3, 1e-5, true, 3000, 0.1); */
  /* term = l2_reg_search(p, 1e-3, 1e-3, true, 3000, 0.1); */

  term = l2_reg_search(p, 1e-3, 1e-3, false, 3000, 0.0);

  /* return 0; */
  /* term = l2_reg_search(p, 1e-2, 1e-5, true, 1000, 0.1); */
  /* term = l2_reg_search(p, 1e-2, 1e-5, false, 100, 0.0001); */
  /* term = l2_reg_search(p, 1e-2, 1e-16, true, 100); */
  /* term = SOLUTION; */

  /* int successes = 0; */
  /* greedy_discrete(p, successes, DA_ZERO, N); */
  /* greedy_discrete(p, successes, DA_E3, N); */
  /* Solver::Summary s; term = solve(p, s, 1e-13); */
  /* cout << s.FullReport() << endl; */
  /* logsol(p,"out.txt"); */
  /* return 0; */


  logsol(p,"out_dense.txt");
  /* if (term == SOLUTION || term == BORDER_LIKELY) { */
  /*   return 0; */
  /* } else { */
  /*   return 1; */
  /* } */

  /* if (term == SOLUTION) { */
  /*   logsol(p,"out_dense.txt"); */
  /*   double ma = minimize_max_abs(p, 1e-1); */
  /*   sparsify(p, 1.0, 1e-4); */
  /*   sparsify(p, 1.0, 1e-4); */
  /*   cout << "ma " << ma << endl; */
  /*   logsol(p,"out.txt"); */
  /*   return 0; */
  /* } */
/* return 1; */

  if (term == SOLUTION) {
    minimize_max_abs(p, 1e-1);
    /* sparsify(p, 1.0, 1e-4); */
    int successes = 0;
    /* greedy_discrete_careful(p, successes, DA_ZERO); */
    /* greedy_discrete_careful(p, successes, DA_E3); */
    greedy_discrete(p, successes, DA_ZERO, N);
    /* greedy_discrete(p, successes, DA_PM_ONE, N); */
    greedy_discrete(p, successes, DA_E3, N);
    /* greedy_discrete_pairs(p, N*100); */
    double ma = minimize_max_abs(p);
    cout << "ma " << ma << endl;
    logsol(p,"out.txt");
    return 0;
  }
  return 1;

  /* logsol(p,"out.txt"); */
  /* return 0; */
}
