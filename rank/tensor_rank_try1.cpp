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
  string outf = "out.txt";
  if (argc >= 3) {
    outf = argv[2];
  }
  
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

  // pure als
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

  MyTerminationType term = NO_SOLUTION;

  /* for (int bi=0; bi < BLOCKS; ++bi) { */
  /*   for (int i=0; i< MULT*(BBOUND[i+1] - BBOUND[i]); ++i) { */
  /*     p.p.SetParameterLowerBound(p.x.data()+MULT*BBOUND[bi],i,-2.0); */
  /*     p.p.SetParameterUpperBound(p.x.data()+MULT*BBOUND[bi],i,2.0); */
  /*   } */
  /* } */
  /* term = l2_reg_search(p, 1e-3, 1e-5, true, 3000, 0.1); */
  /* term = l2_reg_search(p, 1e-3, 1e-3, true, 3000, 0.1); */

  term = l2_reg_search(p, 5e-3, 5e-4, false, 10000, 0.0, false);
  
  switch (term) {
    case CONTINUE: cout << "CONTINUE" << endl; break;
    case CONTINUE_RESET: cout << "CONTINUE_RESET" << endl; break;
    case SOLUTION: cout << "SOLUTION" << endl; break;
    case BORDER: cout << "BORDER" << endl; break;
    case BORDER_OR_NO_SOLUTION: cout << "BORDER_OR_NO_SOLUTION" << endl; break;
    case NO_SOLUTION: cout << "NO_SOLUTION" << endl; break;
    case UNKNOWN: cout << "UNKNOWN" << endl; break;
  }

  logsol(p.x,outf);
  
  if (term == SOLUTION) {
    minimize_max_abs(p, 1e-2);
    int successes = 0;
    greedy_discrete(p, successes, DA_ZERO, N);
    minimize_max_abs(p, 1e-2);
#ifdef CX
    greedy_discrete(p, successes, DA_E3, N);
#else
    greedy_discrete(p, successes, DA_PM_ONE, N);
#endif
    double ma = minimize_max_abs(p);
    cout << "ma " << ma << endl;
    logsol(p.x,outf);
  }
  
  return 0;
}
