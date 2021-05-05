#include <algorithm>
#include <cstring>

#include <lapacke.h>
#include <cblas.h>

#include "tensor.h"

using namespace std;

void als(double *x, int group, double lambda = 0.0) {
  // let ta in {TA,TB,TC} correspond to group
  // and let tb tc be the other two (in cyclic order after ta)

  // we least squares solve AX = B, A is tc tb x sa, X is sa x ta , B is tc tb x ta
  // when lambda != 0, extend the rows of A and B by sa on the bottom
  // everything col major

  auto s = [=](int p, int a, int b, int c) {
    switch ((group+p) % 3) {
      case 0: return a;
      case 1: return b;
    }
    return c;
  };

  int base = s(2,TA,TB,TC)*s(1,TA,TB,TC);
  int m = base+(lambda?s(0,SA,SB,SC):0);
  int n = s(0,SA,SB,SC);
  int nrhs = s(0,TA,TB,TC);

  double B[max(max(TA*(TB*TC+SA),TB*(TA*TC+SB)),TC*(TA*TB+SC))];
  for (int i=0; i<TA; ++i) {
    for (int j=0; j<TB; ++j) {
      for (int k=0; k<TC; ++k) {
        B[s(0,i,j,k)*m + s(1,i,j,k)*s(2,TA,TB,TC) + s(2,i,j,k)] = T[i][j][k];
      }
    }
  }
  if (lambda) {
    for (int i=0; i<s(0,TA,TB,TC); ++i) {
      memset(B+i*m+base,0,s(0,SA,SB,SC)*sizeof(double));
    }
  }
  double A[max(max(SA*(TB*TC+SA),SB*(TA*TC+SB)),SC*(TA*TB+SC))];
  memset(A,0,m*n*sizeof(double));
  for (int t=0; t<SNZ; ++t) {
    for (int j=0; j<s(1,TA,TB,TC); ++j) {
      for (int k=0; k<s(2,TA,TB,TC); ++k) {
        double cur = 0;
        switch (group) {
          case 0: cur = x[TA*SA+j*SB+SJ[t]] * x[TA*SA+TB*SB+k*SC+SK[t]]; break;
          case 1: cur = x[TA*SA+TB*SB+j*SC+SK[t]] * x[k*SA+SI[t]]; break;
          case 2: cur = x[j*SA+SI[t]] * x[TA*SA+k*SB+SJ[t]]; break;
        }
        A[s(0,SI[t],SJ[t],SK[t])*m + j*s(2,TA,TB,TC) + k] += cur;
      }
    }
  }
  if (lambda) {
    for (int i=0; i<s(0,SA,SB,SC); ++i) {
      A[i*m+base+i] = lambda;
    }
  }
  LAPACKE_dgels(LAPACK_COL_MAJOR,'N',m,n,nrhs,A,m,B,m);
  LAPACKE_dlacpy(LAPACK_COL_MAJOR,'N',n,nrhs,B,m,x+s(0,0,TA*SA,TA*SA+TB*SB),s(0,SA,SB,SC));
}
