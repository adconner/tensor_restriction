#include <ceres/ceres.h>
#include <functional>
#include <tuple>
#include <cholmod.h>

#include "problem.h"

using namespace ceres;
using namespace std;

// note that the lambda of this function is the l2 regularization on dx, ie
// should be viewed as 1/mu, where mu is the trust region radius. It is distinct
// from l2 regularization of the sequence of x values, which can be embedded in
// the equations of p
void cholmod(MyProblem &p, function<tuple<bool,bool,double>(double)> f) {
  double cost; vector<double> crs; CRSMatrix jac;
  Problem::EvaluateOptions eopts;
  eopts.parameter_blocks.clear();
  for (int i=0; i<BLOCKS; ++i)
    eopts.parameter_blocks.push_back(p.x.data()+MULT*BBOUND[i]);
  p.p.Evaluate(eopts,&cost,&crs,0,&jac);

  bool accept_step,cont; 
  double lambda[2] = {0.0,0.0};
  tie(accept_step,cont,lambda[0]) = f(cost);
  if (!cont) return;

  cholmod_common c;
  cholmod_start(&c);

  // rows,cols,nnz,sorted,packed,upper lower unstructered,entry type
  cholmod_sparse *jact = cholmod_allocate_sparse(jac.num_cols,jac.num_rows
      ,jac.values.size(),true,true,0,CHOLMOD_REAL,&c);
  copy(jac.rows.begin(),jac.rows.end(),(int*)jact->p);
  copy(jac.cols.begin(),jac.cols.end(),(int*)jact->i);
  copy(jac.values.begin(),jac.values.end(),(double*)jact->x);

  cholmod_factor *L = cholmod_analyze(jact,&c);

  cholmod_dense *rs = cholmod_allocate_dense(crs.size(),1,crs.size(),CHOLMOD_REAL,&c);
  copy(crs.begin(),crs.end(),(double*)rs->x);
  cholmod_dense *jact_rs = cholmod_zeros(jact->nrow,1,CHOLMOD_REAL,&c);

  cholmod_dense *dx = 0, *y = 0, *e = 0; // workspace for solve2, allocated by first call

  vector<double> oldx;
  vector<int> keep_jac_rows(jac.num_rows);
  for (int i=0; i<jac.num_rows; ++i) keep_jac_rows[i] = i;
  while (true) {
    double alpha[2] = {-1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cholmod_sdmult(jact,0,alpha,beta,rs,jact_rs,&c); 
    /* cholmod_factorize(jact,L,&c); */
    cholmod_factorize_p(jact,lambda,keep_jac_rows.data(),keep_jac_rows.size(),L,&c);
    cholmod_solve2(CHOLMOD_LDLt,L,jact_rs,0,&dx,0,&y,&e,&c);

    oldx = p.x;
    for (int i=0,li=0; i<p.x.size(); ++i) {
      if (p.variable_mask[i]) {
        p.x[i] += ((double*)dx->x)[li++];
      }
    }

    p.p.Evaluate(eopts,&cost,&crs,0,&jac);
    tie(accept_step,cont,lambda[0]) = f(cost);
    if (!cont) 
      break;
    if (!accept_step) {
      p.x = oldx;
      p.p.Evaluate(eopts,&cost,&crs,0,&jac);
    }

    copy(jac.rows.begin(),jac.rows.end(),(int*)jact->p);
    copy(jac.cols.begin(),jac.cols.end(),(int*)jact->i);
    copy(jac.values.begin(),jac.values.end(),(double*)jact->x);
  }

  cholmod_free_dense(&dx,&c);
  cholmod_free_dense(&y,&c);
  cholmod_free_dense(&e,&c);
  cholmod_free_dense(&rs,&c);
  cholmod_free_dense(&jact_rs,&c);
  cholmod_free_factor(&L,&c);
  cholmod_free_sparse(&jact,&c);
  cholmod_finish(&c);
}

void basic(MyProblem &p) {
  int it = 0;
  cholmod(p,[&](double cost) {
      double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
          {return max(a,std::abs(b));} ); 
      double l2 = sqrt(accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
            {return a+b*b;} )); 
      printf("%4d %20.15g %20.15g %20.15g\n",it,2*cost,ma,l2);
      return make_tuple(true,it++ < 10,100.0);
    });
}
