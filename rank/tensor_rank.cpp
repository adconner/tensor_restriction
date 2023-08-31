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

const double pthresh = 1e-1;

MyTerminationType try1(int r, vector<double> &x, bool stop_on_br = true) {
  set_rank_r(r);
  set_params();
  
  Problem::Options popts;
  /* popts.enable_fast_removal = true; */
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
  int fac = 0;
  for (int it=0; it<2000; ++it) {
    if (SYM) {
      if (!tight) {
        als_sym(p.x.data(),sqalpha);
      } else {
        als_sym_some(p.x.data(),eqs,sqalpha);
      }
    } else {
      if (!tight) {
        als(p.x.data(),fac,sqalpha);
      } else {
        als_some(p.x.data(),eqs,fac,sqalpha);
      }
      uniform_int_distribution<int> dist(0,1);
      fac = (fac + dist(rng)) % 3;
    }
    double cost; p.p.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0);
    if ((costlast - cost) / costlast < 5e-5) {
      bad++;
    } else {
      bad=0;
    }
    /* double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double TA, double TB) */ 
    /*     {return max(TA,std::abs(TB));} ); */ 
    /* double l2 = sqrt(accumulate(p.x.begin(),p.x.end(),0.0,[](double TA, double TB) */ 
    /*     {return TA+TB*TB;} )); */ 
    /* printf("%4d %20.15g %20.15g %20.15g %10.5g %d %5.1e\n",it,2*cost,ma,l2,sqalpha,bad, */
    /*     (costlast-cost)/costlast ); */
    if (sqalpha == 0.0 && cost > costlast) 
      break;
    if (bad >= 3 || cost > costlast) {
      if (sqalpha == 0.0 && bad >= 20) {
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

  MyTerminationType term = l2_reg_search(p, 2e-3, 1e-3, stop_on_br, 3000, 0.0);
  x.swap(p.x);
  return term;
}

int main(int argc, const char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::digits10);
  random_device rd;
  rng.seed(rd());
  /* rng.seed(8); */
  
  cin >> TA >> TB >> TC;
  T.assign(TA,vector<vector<F> >(TB, vector<F>(TC)));
  for (int i=0; i<TA; ++i) {
    for (int j=0; j<TB; ++j) {
      for (int k=0; k<TC; ++k) {
        cin >> T[i][j][k];
      }
    }
  }

  bool find_brank = true;
  string brank_out = "outbr.txt";

  bool find_rank = false;
  string rank_out = "outr.txt";
  
  if (argc >= 2) {
    brank_out = string(argv[1]);
    find_brank = brank_out != "no";
  }
  if (argc >= 3) {
    rank_out = string(argv[2]);
    find_rank = rank_out != "no";
  }

  int brlower = argc >= 4 ? stoi(argv[3]) : 0;
  int rupper = argc >= 5 ? stoi(argv[4]) : min(min(TA*TB,TA*TC), TB*TC);

  vector<double> brp(rupper+1);
  vector<double> rp(rupper+1);
  for (int i=0; i <= min(max(max(TA,TB),TC), rupper); ++i) {
    rp[i] = brp[i] = 1.0;
  }
  for (int i=max(max(TA,TB),TC)+1; i <= rupper; ++i) {
    rp[i] = brp[i] = brp[i-1] * 0.8;
  }
  for (int i=0; i < brlower; ++i) {
    brp[i] = rp[i] = 0.0;
  }

  int brhi = rupper, rhi = rupper;
  vector<double> x, bestr, bestbr;

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
  normalize();

  auto update = [&](int r, MyTerminationType res) {
    if (res == SOLUTION) {
      for (int i=r+1; i <= rupper; ++i) {
        rp[i] = 0.0;
        brp[i] = 0.0;
      }
      if (r < rhi) {
        rhi = r;
        bestr = x;
      }
      if (r < brhi) {
        brhi = r;
        bestbr = x;
      }
    } else if (res == BORDER) {
      double mult = 0.5;
      for (int i=r; i >= 0; --i) {
        rp[i] *= mult;
        mult *= 0.5;
      }
      for (int i=r+1; i <= rupper; ++i) {
        brp[i] = 0.0;
      }
      if (r < brhi) {
        brhi = r;
        bestbr = x;
      }
    } else if (res == BORDER_OR_NO_SOLUTION) {
      double mult = 0.5;
      for (int i=r; i >= 0; --i) {
        rp[i] *= mult;
        mult *= 0.5;
      }
      mult = 0.9;
      for (int i=r+1; i <= rupper; ++i) {
        brp[i] *= mult;
        mult *= 0.9;
      }
    } else {
      if (res != NO_SOLUTION) {
        printf("warn: treating unknown as no solution\n");
      }
      double mult = 0.5;
      for (int i=r; i >= 0; --i) {
        rp[i] *= 0.9*mult;
        brp[i] *= mult;
        mult *= 0.5;
      }
    }
    normalize();
  };

  auto nextbrcheck = [&]() {
    double p = brp[0];
    for (int r=1; r <= rupper; ++r) {
      p += brp[r];
      if (p >= 0.5) {
        if (r == brhi) {
          return make_pair(r-1, 1-brp[r]);
        } else {
          return make_pair(r, 1-brp[r]);
        }
      }
    }
    assert(false);
  };

  auto nextrcheck = [&]() {
    double p = rp[0];
    for (int r=1; r <= rupper; ++r) {
      p += rp[r];
      if (p >= 0.5) {
        if (r == rhi) {
          return make_pair(r-1, 1-rp[r]);
        } else {
          return make_pair(r, 1-rp[r]);
        }
      }
    }
    assert(false);
  };

  if (find_brank) {
    auto [r,p] = nextbrcheck();
    while (p > pthresh) {
      printf("border rank try %d %g\n", r,p);
      copy(brp.begin(),brp.end(),ostream_iterator<double>(cout," ")); cout << endl;
      auto res = try1(r,x,false);
      update(r,res);
      tie(r,p) = nextbrcheck();
    }
  }

  if (find_rank) {
    auto [r,p] = nextrcheck();
    while (p > pthresh) {
      printf("rank try %d %g\n", r,p);
      copy(rp.begin(),rp.end(),ostream_iterator<double>(cout," ")); cout << endl;
      auto res = try1(r,x);
      update(r,res);
      tie(r,p) = nextrcheck();
    }
  }
  
  if (find_brank && brhi <= rupper) {
    printf("border rank %d\n",brhi);
    logsol(bestbr,brank_out);
  }
  if (find_rank && rhi <= rupper) {
    printf("rank %d\n",rhi);
    logsol(bestr,rank_out);
  }

  return 0;
  
}
