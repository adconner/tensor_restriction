#include <ceres/ceres.h>
#include <functional>
#include <tuple>
#include <cholmod.h>

#include "problem.h"

using namespace ceres;
using namespace std;

cholmod_sparse *jacobian(MyProblem &p, Problem::EvaluateOptions eopts, cholmod_common *c);

// note that the lambda of this function is the l2 regularization on dx, ie
// should be viewed as 1/mu, where mu is the trust region radius. It is distinct
// from l2 regularization of the sequence of x values, which can be embedded in
// the equations of p
void cholmod(MyProblem &p, function<tuple<bool,bool,double>(double,double,double)> f) {
  double icost; vector<double> crs; CRSMatrix jac;
  Problem::EvaluateOptions eopts;
  eopts.parameter_blocks.clear();
  for (int i=0; i<BLOCKS; ++i)
    eopts.parameter_blocks.push_back(p.x.data()+MULT*BBOUND[i]);
  p.p.Evaluate(eopts,&icost,&crs,0,&jac);

  bool accept_step,cont; 
  double lambda[2] = {0.0,0.0};
  tie(accept_step,cont,lambda[0]) = f(-1.0,icost,-1.0);
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
  copy(crs.begin(),crs.end(),(double*)rs->x); // not actually used
  cholmod_dense *jact_rs = cholmod_zeros(jact->nrow,1,CHOLMOD_REAL,&c);

  cholmod_dense *dx = 0, *y = 0, *e = 0; // workspace for solve2, allocated by first call
  vector<double> oldx;

  vector<int> keep_jac_rows(jact->ncol);
  for (int i=0; i<jact->ncol; ++i) keep_jac_rows[i] = i;

  while (true) {
    p.p.Evaluate(eopts,&icost,&crs,0,&jac);
    copy(crs.begin(),crs.end(),(double*)rs->x);
    copy(jac.rows.begin(),jac.rows.end(),(int*)jact->p);
    copy(jac.cols.begin(),jac.cols.end(),(int*)jact->i);
    copy(jac.values.begin(),jac.values.end(),(double*)jact->x);

    double alpha[2] = {-1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cholmod_sdmult(jact,0,alpha,beta,rs,jact_rs,&c); 
    cholmod_factorize_p(jact,lambda,keep_jac_rows.data(),keep_jac_rows.size(),L,&c);
    cholmod_solve2(CHOLMOD_A,L,jact_rs,0,&dx,0,&y,&e,&c);

    alpha[0] = 1.0; beta[0] = 1.0;
    cholmod_sdmult(jact,1,alpha,beta,dx,rs,&c);
    double model_cost_change = icost - 0.5*accumulate((double*)rs->x,
        ((double*)rs->x)+rs->nrow,0.0, [](double a, double b) {return a+b*b;});

    oldx = p.x;
    for (int i=0,li=0; i<p.x.size(); ++i) {
      if (p.variable_mask[i]) {
        p.x[i] += ((double*)dx->x)[li++];
      }
    }

    double cost; p.p.Evaluate(eopts,&cost,0,0,0);
    tie(accept_step,cont,lambda[0]) = f(icost,cost,model_cost_change);
    if (!accept_step) {
      p.x = oldx;
    if (!cont) 
      break;
    }

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

void levenberg_marquardt(MyProblem &p, function<bool(double,double,double,bool)> f, 
    const double eps = 0.0, const double eta1=0.90, const double eta2=0.05, double mu=10000.0) {
  cholmod(p,[&](double icost, double cost, double model_cost_change) {
      if (icost == -1.0) { // first iteration
        return make_tuple(true,true,1/mu);
      }
      double rho = (icost - cost) / model_cost_change;
      bool accept_step = rho > eps;
      if (!f(accept_step?cost:icost,rho,mu,accept_step)) {
        return make_tuple(accept_step,false,0.0);
      }
      /* printf("lm icost=%g model_cost_change=%g cost=%g rho=%g mu=%g accepted=%d\n",icost,model_cost_change,cost,rho,mu,(int)accept_step); */
      if (0.99 < rho && rho < 1.01) {
        mu *= 8;
      } else if (rho > eta1) {
        mu *= 3;
      } else if (rho < eta2) {
        mu *= 0.5;
      }
      return make_tuple(accept_step,true,1/mu);
    });
}

void trust_region_f(MyProblem &p, function<void()> f, double relftol = 1e-3, int maxit = 200) {
  int it = 1;
  double icost = 1e10;
  levenberg_marquardt(p,[&](double cost, double rho, double mu, bool accept_step) {
      if (accept_step) { // only do user changes to state (eg, Algorithm 2 in ceres docs) 
                         // if it will not be blown away
        f();
      }
      double fcost; p.p.Evaluate(Problem::EvaluateOptions(),&fcost,0,0,0);
      double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
          {return max(a,std::abs(b));} ); 
      double l2 = sqrt(accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
            {return a+b*b;} )); 
      printf("%3d %15.10g %7.5g %7.5g %-+12.6g %12.6g %9g\n",it,2*cost,ma,l2,rho,(cost-fcost)/(icost-fcost),mu);
      it++;
      bool cont = !(it >= maxit || cost < 1e-26 
          || (cost < icost && (icost - cost) / cost < relftol));
      icost = cost;
      return cont;
    });
}

void trust_region(MyProblem &p, double relftol = 1e-3, int maxit = 200) {
  trust_region_f(p,[](){},relftol,maxit);
}

cholmod_sparse *jacobian(MyProblem &p, Problem::EvaluateOptions eopts, cholmod_common *c) {
  if (eopts.parameter_blocks.empty())
    p.p.GetParameterBlocks(&eopts.parameter_blocks);
  if (eopts.residual_blocks.empty())
    p.p.GetResidualBlocks(&eopts.residual_blocks);

  int nnz = 0;
  for (auto rid : eopts.residual_blocks) {
    vector<double*> xs;
    p.p.GetParameterBlocksForResidualBlock(rid,&xs);
    for (auto x : xs) {
      if (find(eopts.parameter_blocks.begin(),eopts.parameter_blocks.end(),x) !=
          eopts.parameter_blocks.end()) {
        nnz += p.p.ParameterBlockTangentSize(x) *
          p.p.GetCostFunctionForResidualBlock(rid)->num_residuals();
      }
    }
  }
  cholmod_triplet *jact = cholmod_allocate_triplet(N,M,nnz,0,CHOLMOD_REAL,c);
  int ix=0;

  vector<int> js;
  int j = 0;
  for (auto x : eopts.parameter_blocks) {
    js.push_back(j);
    j += p.p.ParameterBlockTangentSize(x);
  }

  int i = 0, ii = 0;
  for (auto rid : eopts.residual_blocks) {
    int isize = p.p.GetCostFunctionForResidualBlock(rid)->num_residuals();
    vector<double*> xs;
    p.p.GetParameterBlocksForResidualBlock(rid,&xs);
    vector<double*> jacobians;
    for (auto x : xs) {
      auto it = find(eopts.parameter_blocks.begin(),eopts.parameter_blocks.end(),x);
      if (it != eopts.parameter_blocks.end()) {
        int j = js[it - eopts.parameter_blocks.begin()];
        int jsize = p.p.ParameterBlockTangentSize(x);

        jacobians.push_back( ((double*)jact->x) + ii );
        for (int li = 0; li < isize; ++li) {
          for (int lj = 0; lj < jsize; ++lj) {
            ((int*)jact->i)[ii] = j + lj;
            ((int*)jact->j)[ii] = i + li;
            ii++;
          }
        }

      }
    }
    p.p.EvaluateResidualBlock(rid,eopts.apply_loss_function,0,0,jacobians.data());
    i += isize;
  }
  jact->nnz = nnz;
  assert(ii == nnz);

  return cholmod_triplet_to_sparse(jact,nnz,c);
}

