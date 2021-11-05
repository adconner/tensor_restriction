#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <cassert>

#include <lapacke.h>
#include <cblas.h>

#include "tensor.h"

#include "als.h"

using namespace std;

void als(double *x, int group, double lambda) {
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
        A[s(0,SI[t],SJ[t],SK[t])*m + j*s(2,TA,TB,TC) + k] += SV[t] * cur;
      }
    }
  }
  if (lambda) {
    for (int i=0; i<s(0,SA,SB,SC); ++i) {
      A[i*m+base+i] = lambda;
    }
  }

  int rank = 0;
  int jpvt[max(max(SA,SB),SC)];
  memset(jpvt,0,s(0,SA,SB,SC)*sizeof(int));
  LAPACKE_dgelsy(LAPACK_COL_MAJOR,m,n,nrhs,A,m,B,m,jpvt,1e-13,&rank);
  if (rank != s(0,SA,SB,SC))
    printf("als: rank deficient problem, m=%d, n=%d, rank=%d\n",m,n,rank);
  /* LAPACKE_dgels(LAPACK_COL_MAJOR,'N',m,n,nrhs,A,m,B,m); */

  LAPACKE_dlacpy(LAPACK_COL_MAJOR,'N',n,nrhs,B,m,x+s(0,0,TA*SA,TA*SA+TB*SB),s(0,SA,SB,SC));
}

void als_sym(double *x, double lambda, double step) {
  assert(TA == TB && TB == TC && SA == SB && SB == SC);
  double tripx[3*SA*TA];
  copy(x,x+SA*TA,tripx);
  copy(x,x+SA*TA,tripx+SA*TA);
  copy(x,x+SA*TA,tripx+2*SA*TA);
  als(tripx,0,lambda);
  for (int i=0; i<SA*TA; ++i) {
    x[i] = x[i] + step*(tripx[i] - x[i]);
  }
}

void als_some(double *x, const vector<tuple<int,int,int> > & eqs, int group, double lambda) {
  // computes the same as als, but allows some equations to be ignored (only
  // those in eqs are kept)
  auto s = [=](int p, int a, int b, int c, int sgn=1) {
    switch ((p+sgn*group+3) % 3) {
      case 0: return a;
      case 1: return b;
    }
    return c;
  };

  vector<vector<tuple<int,int> > > gs(s(0,TA,TB,TC));
  for (auto e : eqs) {
    auto [i, j, k] = e;
    gs[s(0,i,j,k)].push_back(make_tuple(s(1,i,j,k),s(2,i,j,k)));
  }
  map<vector<tuple<int,int> > , vector<int> > gsi;
  for (int i=0; i<gs.size(); ++i) {
    sort(gs[i].begin(),gs[i].end());
    gsi[gs[i]].push_back(i);
  }


  for (auto it = gsi.begin(); it != gsi.end(); ++it) {
    int base = it->first.size(); // s(2,TA,TB,TC)*s(1,TA,TB,TC);
    int m = base+(lambda?s(0,SA,SB,SC):0);
    int n = s(0,SA,SB,SC);
    int nrhs = it->second.size(); // s(0,TA,TB,TC);
    int ldb = max(m,n);

    double B[max(max(TA*(TB*TC+SA),TB*(TA*TC+SB)),TC*(TA*TB+SC))];

    for (int ii=0; ii < it->second.size(); ++ii) {
      for (int jk=0; jk < it->first.size(); ++jk) {
        int i = it->second[ii];
        int j,k; tie(j,k) = it->first[jk];
        B[ii*ldb + jk] = T[s(0,i,j,k,-1)][s(1,i,j,k,-1)][s(2,i,j,k,-1)];
      }
    }

    if (lambda) {
      for (int i=0; i<s(0,TA,TB,TC); ++i) {
        memset(B+i*ldb+base,0,s(0,SA,SB,SC)*sizeof(double));
      }
    }
    double A[max(max(SA*(TB*TC+SA),SB*(TA*TC+SB)),SC*(TA*TB+SC))];
    memset(A,0,m*n*sizeof(double));
    for (int t=0; t<SNZ; ++t) {
      for (int jk=0; jk < it->first.size(); ++jk) {
        int j,k; tie(j,k) = it->first[jk];
        double cur = 0;
        switch (group) {
          case 0: cur = x[TA*SA+j*SB+SJ[t]] * x[TA*SA+TB*SB+k*SC+SK[t]]; break;
          case 1: cur = x[TA*SA+TB*SB+j*SC+SK[t]] * x[k*SA+SI[t]]; break;
          case 2: cur = x[j*SA+SI[t]] * x[TA*SA+k*SB+SJ[t]]; break;
        }
        A[s(0,SI[t],SJ[t],SK[t])*m + jk] += SV[t] * cur;
      }
    }
    if (lambda) {
      for (int i=0; i<s(0,SA,SB,SC); ++i) {
        A[i*m+base+i] = lambda;
      }
    }

    int rank = 0;
    int jpvt[max(max(SA,SB),SC)];
    memset(jpvt,0,s(0,SA,SB,SC)*sizeof(int));
    LAPACKE_dgelsy(LAPACK_COL_MAJOR,m,n,nrhs,A,m,B,ldb,jpvt,1e-13,&rank);
    if (rank != min(m,n))
      printf("als: rank deficient problem, m=%d, n=%d, rank=%d\n",m,n,rank);
    /* LAPACKE_dgels(LAPACK_COL_MAJOR,'N',m,n,nrhs,A,m,B,m); */

    for (int j = 0; j<nrhs; ++j) {
      for (int i = 0; i<n; ++i) {
        x[s(0,0,TA*SA,TA*SA+TB*SB) +it->second[j]*n +i ] = B[j*ldb + i];
      }
    }
  }
}

void als_sym_some(double *x, const vector<tuple<int,int,int> > & eqs, double lambda, double step) {
  assert(TA == TB && TB == TC && SA == SB && SB == SC);
  double tripx[3*SA*TA];
  copy(x,x+SA*TA,tripx);
  copy(x,x+SA*TA,tripx+SA*TA);
  copy(x,x+SA*TA,tripx+2*SA*TA);
  als_some(tripx,eqs,0,lambda);
  for (int i=0; i<SA*TA; ++i) {
    x[i] = x[i] + step*(tripx[i] - x[i]);
  }
}
