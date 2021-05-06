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
void cholmod(MyProblem &p, function<tuple<bool,bool,double>(double,double)> f) {
  double cost; vector<double> crs; CRSMatrix jac;
  Problem::EvaluateOptions eopts;
  eopts.parameter_blocks.clear();
  for (int i=0; i<BLOCKS; ++i)
    eopts.parameter_blocks.push_back(p.x.data()+MULT*BBOUND[i]);
  p.p.Evaluate(eopts,&cost,&crs,0,&jac);

  bool accept_step,cont; 
  double lambda[2] = {0.0,0.0};
  tie(accept_step,cont,lambda[0]) = f(cost,cost);
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
    cholmod_factorize_p(jact,lambda,keep_jac_rows.data(),keep_jac_rows.size(),L,&c);
    cholmod_solve2(CHOLMOD_A,L,jact_rs,0,&dx,0,&y,&e,&c);

    alpha[0] = 1.0; beta[0] = 1.0;
    cholmod_sdmult(jact,1,alpha,beta,dx,rs,&c);
    double model_cost = 0.5*accumulate((double*)rs->x,((double*)rs->x)+rs->nrow,0.0,
        [](double a, double b) {return a+b*b;});

    oldx = p.x;
    for (int i=0,li=0; i<p.x.size(); ++i) {
      if (p.variable_mask[i]) {
        p.x[i] += ((double*)dx->x)[li++];
      }
    }

    p.p.Evaluate(eopts,&cost,&crs,0,&jac);
    tie(accept_step,cont,lambda[0]) = f(cost,model_cost);
    if (!cont) 
      break;
    if (!accept_step) {
      // this method of evaluation is fast in the common case of steps being
      // accepted
      p.x = oldx;
      p.p.Evaluate(eopts,&cost,&crs,0,&jac);
    }

    copy(jac.rows.begin(),jac.rows.end(),(int*)jact->p);
    copy(jac.cols.begin(),jac.cols.end(),(int*)jact->i);
    copy(jac.values.begin(),jac.values.end(),(double*)jact->x);
    copy(crs.begin(),crs.end(),(double*)rs->x);
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

void levenberg_marquardt(MyProblem &p, function<bool(double,double,double)> f, 
    const double eps = 0.0, const double eta1=0.90, const double eta2=0.15, double mu=32.0) {
  double icost = -1.0;
  cholmod(p,[&](double cost, double model_cost) {
      if (icost == -1.0) { // first iteration
        icost = cost;
        return make_tuple(true,true,1/mu);
      }
      double rho = (icost - cost) / (icost - model_cost);
      if (!f(cost,icost,rho)) {
        return make_tuple(false,false,0.0);
      }
      /* return make_tuple(true,true,20.0); */
      bool accept_step = rho > eps;
      /* printf("lm icost=%g model_cost=%g cost=%g rho=%g mu=%g accepted=%d\n",icost,model_cost,cost,rho,mu,(int)accept_step); */
      if (0.99 < rho && rho < 1.01) {
        mu *= 8;
      } else if (rho > eta1) {
        mu *= 2;
      } else if (rho < eta2) {
        mu *= 0.5;
      }
      icost = cost;
      return make_tuple(accept_step,true,1/mu);
    });
}

void trust_region(MyProblem &p, double relftol = 1e-3, int maxit = 200) {
  int it = 1;
  levenberg_marquardt(p,[&](double cost, double icost, double rho) {
      double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
          {return max(a,std::abs(b));} ); 
      double l2 = sqrt(accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
            {return a+b*b;} )); 
      printf("%3d %15.10g %7.5g %7.5g %-+12.6g\n",it,2*cost,ma,l2,rho);
      it++;
      return !(it >= maxit || cost < 1e-26 
          || (cost < icost && (icost - cost) / cost < relftol));
    });
}
