#include <random>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cassert>
#include "initial.h"
#include "opts.h"

using namespace std;
using namespace ceres;
typedef vector<int> vi;
typedef vector<vi> vvi;

vvi ncombinations(const vi &mi, const vi &ma, const vi &costs, int cost) {
  vvi dp(ma.size()+1,vi(cost+1));
  dp[0][0] = 1;
  vi tmp(cost+1);
  cout << endl;
  for (int i=0; i<ma.size(); ++i) {
    for (int j=0; j<=cost; ++j) {
      tmp[j] = dp[i][j] + (j-costs[i] >= 0 ? tmp[j-costs[i]] : 0);
      dp[i+1][j] += j-mi[i]*costs[i] >= 0 ? tmp[j-mi[i]*costs[i]] : 0;
      dp[i+1][j] -= j-(ma[i]+1)*costs[i] >= 0 ? tmp[j-(ma[i]+1)*costs[i]] : 0;
    }
  }
  return dp;
}

vi combination(int ix, const vi &mi, const vi &ma, const vi &costs, int cost, const vvi &dp) {
  vi res(mi.size());
  int num = dp[ma.size()][cost];
  assert(0 <= ix && ix < num);
  int ccost = cost;
  for (int i=ma.size()-1; i >= 0; --i) {
    for (int j=mi[i]; j <= ma[i]; ++j) {
      assert(ccost - j*costs[i] >= 0);
      if (ix < dp[i][ccost - j*costs[i]]) {
        res[i] = j;
        ccost -= j * costs[i];
        break;
      }
      ix -= dp[i][ccost - j*costs[i]];
    }
  }
  assert(ccost == 0);
  return res;
}

void fill_initial(double *x, int argc, char **argv, Problem &problem) {
  random_device rd;
  mt19937 gen(rd());
  if (argc == 1) {
    normal_distribution<> dist(0,0.4);
    generate_n(x,MULT*N,[&] {return dist(gen);});
  } else {
    ifstream in(argv[1]);
    for (int i=0; i<MULT*N; ++i)
      in >> x[i];
  }
#ifdef ORBIT
  // omin omax ocost ovars oMAX TCOST
  vi comin(omin,omin+OS), comax(omax,omax+OS), cocost(ocost,ocost+OS), 
     coMAX(oMAX,oMAX+OS), covars(ovars,ovars+OS);
  vvi dp(ncombinations(comin,comax,cocost,TCOST));
  int ix = uniform_int_distribution<>(0,dp.back().back()-1)(gen);
  vi oix(combination(ix,comin,comax,cocost,TCOST,dp));
  if (verbose) {
    cout << "orbit structure " << ix << "/" << dp.back().back() << ", ";
    for (int i=0; i<oix.size(); ++i) cout << oix[i] << " "; cout << endl;
  }
  for (int i=0, curi=0; i < OS; curi += coMAX[i] * covars[i], ++i) {
    for (int j=oix[i] * covars[i]; j < coMAX[i] * covars[i]; j += covars[i]) {
      for (int k=0; k<covars[i]; ++k) {
        x[(curi+j+k)*MULT] = 0;
        if (MULT == 2) x[(curi+j+k)*MULT+1] = 0;
        problem.SetParameterBlockConstant(x+(curi+j+k)*MULT);
      }
    }
  }
#endif
}
