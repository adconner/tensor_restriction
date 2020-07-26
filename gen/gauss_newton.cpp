#include <lapacke.h>
#include "gauss_newton.h"
/* #define SVD_GAUSS_NEWTON */

void gauss_newton(Problem &p, double *x, double xtol, int max_it, double rcond) {
  double cost; 
  vector<double> rs;
  CRSMatrix jac_sparse; 
  Problem::EvaluateOptions eopts;
  vector<int> xis;
  for (int i=0; i<N; ++i) {
    if (!p.IsParameterBlockConstant(x+MULT*i)) {
      eopts.parameter_blocks.push_back(x + MULT*i);
      xis.push_back(i);
    }
  }
  p.Evaluate(eopts,&cost,&rs,0,&jac_sparse);
  int m=jac_sparse.num_rows, n=jac_sparse.num_cols;
  vector<double> jac(m*n); // in fortran order
#ifdef SVD_GAUSS_NEWTON
  vector<double> s;
#else
  vector<int> jpvt;
#endif

  vector<double> sav(MULT*N);
  for (int it = 1; it <= max_it ; ++it) {
    jac.assign(jac.size(),0.0);
    for (int i=0; i < m; ++i) {
      for (int j=jac_sparse.rows[i]; j < jac_sparse.rows[i+1]; ++j) {
        jac[(jac_sparse.cols[j])*jac_sparse.num_rows+i] = jac_sparse.values[j];
      }
    }
    rs.resize(max(m,n));
    int rank;
#ifdef SVD_GAUSS_NEWTON
    s.assign(min(m,n),0);
    int info = LAPACKE_dgelsd(LAPACK_COL_MAJOR,m,n,1,
        jac.data(),m,rs.data(),rs.size(),
        s.data(), rcond, &rank);
#else
    jpvt.assign(n,0);
    int info = LAPACKE_dgelsy(LAPACK_COL_MAJOR,m,n,1,
        jac.data(),m,rs.data(),rs.size(),
        jpvt.data(), rcond, &rank);
#endif

    /* auto f = [&](double t) { */
    /*   copy(x,x+MULT*N,sav.begin()); */
    /*   for (int i=0; i<n; ++i) { */
    /*     x[xis[i/MULT]*MULT+i%MULT] -= t*rs[i]; */
    /*   } */
    /*   double cost; p.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0); */
    /*   copy(sav.begin(),sav.end(),x); */
    /*   return cost; */
    /* }; */

    /* double lo=0.0, hi=10.0; */
    /* while (hi-lo > 1e-10) { */
    /*   double lmid = (2*lo+hi) / 3, hmid = (lo+2*hi) / 3; */
    /*   if (f(lmid) < f(hmid)) { */
    /*     hi = hmid; */
    /*   } else { */
    /*     lo = lmid; */
    /*   } */
    /* } */
    /* double t0 = (lo + hi) / 2; */
    double t0 = 1.0;

    double dx = 0.0;
    for (int i=0; i<n; ++i) {
      x[xis[i/MULT]*MULT+i%MULT] -= t0*rs[i];
      dx += rs[i]*rs[i]*t0*t0;
    }
    double acost;
    p.Evaluate(eopts,&acost,&rs,0,&jac_sparse);

    if (verbose) {
      cout << it << " " << acost << " " << m << "x" << n << " jac, rank " << rank << " dx " << std::sqrt(dx) << " " << t0 << endl;
    }
    if (acost < 1e-26)
      break;
    if (dx < xtol)
      break;
    cost = acost;
  }

}
