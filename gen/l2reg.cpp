#include "l2reg.h"
#include <iterator>
#include <algorithm>
#include <functional>

struct L2Regularization : public CostFunction {
  L2Regularization(int block_size, double *sqalpha, double *b) : 
      block_size_(block_size), sqalpha_(sqalpha), b_(b) {
    *mutable_parameter_block_sizes() = {MULT*block_size};
    set_num_residuals(MULT*block_size);
  }
  bool Evaluate(const double* const* x,
      double* residuals,
      double** jacobians) const {
    assert (MULT == 1);
    for (int i=0; i<num_residuals(); ++i) {
      double tar = max(min(x[0][i],b_[i]),-b_[i]);
      residuals[i] = sqalpha_[i]*(x[0][i]-tar);
    }
    if (jacobians && jacobians[0]) {
      fill(jacobians[0],jacobians[0]+num_residuals()*num_residuals(),0.0);
      for (int i=0; i<num_residuals(); ++i) {
        if (abs(x[0][i]) > b_[i]) {
          jacobians[0][i*num_residuals()+i] = sqalpha_[i];
        }
      }
    }
    return true;
  }
  int block_size_;
  double *sqalpha_;
  double *b_;
};

typedef function<MyTerminationType(const IterationSummary&)> upf;

class L2RegCallback : public IterationCallback {
  public:
    L2RegCallback(upf f) : f_(f) {}
    CallbackReturnType operator()(const IterationSummary& summary) {
      termination_ = f_(summary);
      return termination_ == CONTINUE ? SOLVER_CONTINUE : SOLVER_TERMINATE_SUCCESSFULLY;
    }

    upf f_;
    MyTerminationType termination_;
};

MyTerminationType l2_reg(Problem &problem, double *x, const Solver::Options &opts, double *sqalpha, double *b, upf f) {
  vector<ResidualBlockId> rids;
  for (int i=0; i<BLOCKS; ++i) {
    rids.push_back(problem.AddResidualBlock(
      new L2Regularization(BBOUND[i+1]-BBOUND[i],sqalpha+BBOUND[i],b+BBOUND[i])
          ,NULL,x+MULT*BBOUND[i]));
  }

  unique_ptr<L2RegCallback> callback(new L2RegCallback(f));
  Solver::Options options(opts);
  options.update_state_every_iteration = true;
  options.callbacks.push_back(callback.get());

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  for (auto rid : rids) {
    problem.RemoveResidualBlock(rid);
  }
  return callback->termination_;
}

MyTerminationType l2_reg_search(Problem &problem, double *x, double target_relative_decrease, double relftol) {
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = 1000;

  double sqalpha_mult = 0.5;
  deque<bool> recent_drop;

  double ma_last = 1.0;
  int consecutive_border_evidence = 0;

  vector<double> sqalpha(N,0.1), b(N,0.0);
  return l2_reg(problem,x,options,sqalpha.data(),b.data(),
        [&] (const IterationSummary &s) {
      double relative_decrease = s.cost_change / s.cost;
      double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) 
          {return max(a,std::abs(b));} ); 

      if (verbose) {
        printf("%3d %20.15g %20.15f %10.5g %-+12.6g %11.6g %20.15g %d\n",
            s.iteration,s.cost,ma,s.step_norm,relative_decrease,sqalpha[0],s.relative_decrease,consecutive_border_evidence);
      }

      if (s.cost < solved_fine) {
        return SOLUTION;
      }
      if (sqalpha[0] == 0.0 && relative_decrease > 0 && relative_decrease < relftol) {
        return NO_SOLUTION;
      }
      if (s.cost < 0.2) { // consider border solution
        // border rank
        // f low and df/f also low
        // ma consistently increasing
        if (ma > ma_last && ma > 2 && relative_decrease < 0.1) {
          consecutive_border_evidence += 1;
        } else {
          consecutive_border_evidence = 0;
        }
        if (consecutive_border_evidence >= 6) {
          return BORDER_LIKELY;
        }
      }
      if (relative_decrease > 0 && relative_decrease < target_relative_decrease) { // drop sqalpha
        if (count(recent_drop.begin(),recent_drop.end(),true)) {
          sqalpha_mult *= sqalpha_mult;
        }
        sqalpha[0] *= sqalpha_mult;
        if (sqalpha[0] < 1e-3) {
        /* if (sqalpha[0] < 1e-6 || s.step_norm < 5e-2) { */
          sqalpha[0] = 0.0;
        }
        recent_drop.push_back(true);
        fill(sqalpha.begin()+1,sqalpha.begin()+N,sqalpha[0]);
      } else {
        recent_drop.push_back(false);
      }
      if (recent_drop.size() > 3) {
        recent_drop.pop_front();
      }

      ma_last = ma;
      return CONTINUE;
      });
}

double minimize_max_abs(Problem &problem, double *x, double eps, double step_mult, double relftol) {
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;

  double lo = 0.0;
  double hi = accumulate(x,x+MULT*N,0.0,[](double a, double b) 
      {return max(a,std::abs(b));} ); 
  double cur = hi*step_mult;

  vector<double> sqalpha(N,1.0), b(N,1.0);
  for (int i=0; i<N; ++i) {
    if ((MULT==1?abs(x[i]):abs(cx(x[2*i],x[2*i+1]))) < sqrt(solved_fine)) {
      b[i] = 0.0;
    }
  }
  while (hi-lo > eps) {
    cout << lo << endl << hi << endl;
    vector<double> sav(x,x+N*MULT);
    for (int i=0; i<b.size(); ++i)
      if (b[i] != 0.0) 
        b[i] = cur;
    MyTerminationType termination = l2_reg(problem,x,options,
        sqalpha.data(),b.data(), [&] (const IterationSummary &s) {
      double relative_decrease = s.cost_change / s.cost;
      if (s.cost < solved_fine) { // sol
        return SOLUTION;
      } else if (relative_decrease > 0 && relative_decrease < relftol) { // no sol
        return NO_SOLUTION;
      }
      return CONTINUE;
    });
    if (termination == SOLUTION) {
      hi = cur;
      if (lo == 0.0) {
        cur = hi * step_mult;
      } else {
        cur = (lo+hi) / 2;
      }
    } else {
      lo = cur;
      cur = (lo+hi) / 2;
      copy(sav.begin(),sav.end(),x);
    }
  }

  return hi;
}


void sparsify(Problem &problem, double *x, double B, double relftol) {
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = 10000;

  int lo=0, hi=N;
  int tries = 10;

  vector<double> sqalpha(N), b(N);
  while (lo < hi) {
    int cur = (lo+hi-1) / 2 + 1;
  /* for (int cur = N; cur >= 1; --cur) { */
    /* cout << lo << " " << hi << endl; */
    vector<double> sav(x,x+N*MULT);
    MyTerminationType termination;
    for (int tr = 1; tr <= tries; ++tr) {
      // TODO do while smallest n are changing

      vector<pair<double,int> > xabs(N);
      for (int i=0; i<N; ++i) {
        xabs[i] = make_pair(MULT==1?abs(x[i]):abs(cx(x[2*i],x[2*i+1])),i);
      }
      assert(cur >= 1);
      nth_element(xabs.begin(),xabs.begin()+cur-1,xabs.end());
      fill(sqalpha.begin(),sqalpha.end(),1.0);
      fill(b.begin(),b.end(),B);
      for (int i=0; i < cur; ++i) {
        b[xabs[i].second] = 0.0;
        sqalpha[xabs[i].second] = 1e-2;
      }

      termination = l2_reg(problem,x,options,sqalpha.data(),b.data(),
          [&] (const IterationSummary &s) {
        double relative_decrease = s.cost_change / s.cost;

        if (verbose) {
          double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) 
              {return max(a,std::abs(b));} ); 
          printf("%3d %3d %3d %2d %3d %20.15g %20.15f %10.5g %-+12.6g %20.15g\n",
              lo,cur,hi,tr,s.iteration,s.cost,ma,s.step_norm,relative_decrease,s.relative_decrease);
        }

        if (s.cost < solved_fine) { // sol
          return SOLUTION;
        } else if (relative_decrease > 0 && relative_decrease < relftol) { // no sol
          return NO_SOLUTION;
        }
          return CONTINUE;
      });
      if (termination == SOLUTION) break;
    }
    if (termination == SOLUTION) {
      lo = cur;
      /* minimize_max_abs(problem, x, 1e-1, 0.8, relftol); */
    } else {
      hi = cur-1;
      copy(sav.begin(),sav.end(),x);
    }
  }
  /* minimize_max_abs(problem, x, 1e-14, 0.8, relftol); */
}


void l2_reg_refine(Problem &problem, double *x, double target_relative_decrease, double relftol) {
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = 1000;

  double sqalpha_mult = 0.5;
  deque<bool> recent_drop;

  vector<double> sav(x,x+MULT*N);
  vector<double> sqalpha(N,0.1), b(N,0.0);
  for (int i=0; i<N; ++i) {
    if ((MULT==1?abs(x[i]):abs(cx(x[2*i],x[2*i+1]))) < sqrt(solved_fine)) {
      sqalpha[i] = 1.0;
    }
  }
  double sqalphaval = 0.1;
  MyTerminationType termination = l2_reg(problem,x,options,sqalpha.data(),b.data(),
        [&] (const IterationSummary &s) {
      double relative_decrease = s.cost_change / s.cost;
      double ma = accumulate(x,x+MULT*N,0.0,[](double a, double b) 
          {return max(a,std::abs(b));} ); 

      if (verbose) {
        printf("%3d %20.15g %20.15f %10.5g %-+12.6g %11.6g %20.15g\n",
            s.iteration,s.cost,ma,s.step_norm,relative_decrease,sqalphaval,s.relative_decrease);
      }

      if (s.cost < solved_fine) {
        return SOLUTION;
      }
      if (sqalphaval == 0.0 && relative_decrease > 0 && relative_decrease < relftol) {
        return NO_SOLUTION;
      }
      if (relative_decrease > 0 && relative_decrease < target_relative_decrease) { // drop sqalpha
        if (count(recent_drop.begin(),recent_drop.end(),true)) {
          sqalpha_mult *= sqalpha_mult;
        }
        sqalphaval *= sqalpha_mult;
        if (sqalphaval < 1e-3) {
          sqalphaval = 0.0;
        }
        recent_drop.push_back(true);
        for (int i=0; i<N; ++i) {
          if (sqalpha[i] != 1.0) {
            sqalpha[i] = sqalphaval;
          }
        }
      } else {
        recent_drop.push_back(false);
      }
      if (recent_drop.size() > 3) {
        recent_drop.pop_front();
      }

      return CONTINUE;
      });
  if (termination == NO_SOLUTION) {
    copy(sav.begin(),sav.end(),x);
    cout << "l2_reg_refine fail" << endl;
  }
}


/* struct PiecewiseLinear : public CostFunction { */
/*   PiecewiseLinear(int block_size, double eps, double x1, double x2, double x3) : */
/*       eps_(eps), x1_(x1), x2_(x2), x3_(x3) { */
/*     *mutable_parameter_block_sizes() = {block_size}; */
/*     set_num_residuals(block_size); */
/*   } */
/*   bool Evaluate(const double* const* x, */
/*       double* residuals, */
/*       double** jacobians) const { */
/*     /1* for (int i=0; i<num_residuals(); ++i) { *1/ */
/*     /1*   residuals[i] = (x[0][i]-1)*x[0][i]*(x[0][i]+1); *1/ */
/*     /1* } *1/ */
/*     /1* if (jacobians && jacobians[0]) { *1/ */
/*     /1*   fill(jacobians[0],jacobians[0]+num_residuals()*num_residuals(),0.0); *1/ */
/*     /1*   for (int i=0; i<num_residuals(); ++i) { *1/ */
/*     /1*     jacobians[0][i*num_residuals()+i] = (x[0][i]-1)*x[0][i] *1/ */ 
/*     /1*       + (x[0][i]-1)*(x[0][i]+1) + x[0][i]*(x[0][i]+1); *1/ */
/*     /1*   } *1/ */
/*     /1* } *1/ */
/*     for (int i=0; i<num_residuals(); ++i) { */
/*       if (x[0][i] < -x3_) { */
/*         residuals[i] = -(x[0][i] + x3_); */
/*       } else if (x[0][i] < -x2_) { */
/*         residuals[i] = 0.0; */
/*       } else if (x[0][i] < -(x1_+x2_)/2) { */
/*         residuals[i] = -(x[0][i] + x2_); */
/*       } else if (x[0][i] < -x1_) { */
/*         residuals[i] = x[0][i] + x1_; */
/*       } else if (x[0][i] < x1_) { */
/*         residuals[i] = 0.0; */
/*       } else if (x[0][i] < (x1_+x2_)/2) { */
/*         residuals[i] = x[0][i] - x1_; */
/*       } else if (x[0][i] < x2_) { */
/*         residuals[i] = -(x[0][i] - x2_); */
/*       } else if (x[0][i] < x3_) { */
/*         residuals[i] = 0.0; */
/*       } else { */
/*         residuals[i] = -(x[0][i] - x3_); */
/*       } */
/*     } */
/*     if (jacobians && jacobians[0]) { */
/*       fill(jacobians[0],jacobians[0]+num_residuals()*num_residuals(),0.0); */
/*       for (int i=0; i<num_residuals(); ++i) { */
/*         if (x[0][i] < -x3_) { */
/*           jacobians[0][i*num_residuals()+i] = -1.0; */
/*         } else if (x[0][i] < -x2_) { */
/*           jacobians[0][i*num_residuals()+i] = 0.0; */
/*         } else if (x[0][i] < -(x1_+x2_)/2) { */
/*           jacobians[0][i*num_residuals()+i] = -1.0; */
/*         } else if (x[0][i] < -x1_) { */
/*           jacobians[0][i*num_residuals()+i] = 1.0; */
/*         } else if (x[0][i] < x1_) { */
/*           jacobians[0][i*num_residuals()+i] = 0.0; */
/*         } else if (x[0][i] < (x1_+x2_)/2) { */
/*           jacobians[0][i*num_residuals()+i] = 1.0; */
/*         } else if (x[0][i] < x2_) { */
/*           jacobians[0][i*num_residuals()+i] = -1.0; */
/*         } else if (x[0][i] < x3_) { */
/*           jacobians[0][i*num_residuals()+i] = 0.0; */
/*         } else { */
/*           jacobians[0][i*num_residuals()+i] = -1.0; */
/*         } */
/*       } */
/*     } */
/*     return true; */
/*   } */
/*   double eps_,x1_,x2_,x3_; */
/* }; */

/* typedef function<tuple<double,double,double,double,bool>(const IterationSummary&)> upf2; */

/* class PiecewiseCallback : public IterationCallback { */
/*   public: */
/*     PiecewiseCallback(upf2 f, const vector<PiecewiseLinear*> &terms) : */
/*       f_(f), terms_(terms) {} */
/*     CallbackReturnType operator()(const IterationSummary& summary) { */
/*       double eps,x1,x2,x3; */ 
/*       bool cont; */
/*       tie(eps,x1,x2,x3,cont) = f_(summary); */

/*       for (auto t : terms_) { */
/*         t->eps_ = eps; */
/*         t->x1_ = x1; */
/*         t->x2_ = x2; */
/*         t->x3_ = x3; */
/*       } */
/*       return cont ? SOLVER_CONTINUE : SOLVER_TERMINATE_SUCCESSFULLY; */
/*     } */

/*     upf2 f_; */
/*     vector<PiecewiseLinear*> terms_; */
/* }; */

/* void piecewise_solve(Problem &problem, const Solver::Options &opts, double init_eps, */ 
/*     double init_x1, double init_x2, double init_x3, upf2 f) { */
/*   vector<double *> blocks; */
/*   problem.GetParameterBlocks(&blocks); */
/*   vector<PiecewiseLinear *> terms; */
/*   vector<ResidualBlockId> rids; */
/*   for (double *b : blocks) { */
/*     terms.push_back(new PiecewiseLinear(problem.ParameterBlockSize(b), */
/*           init_eps,init_x1,init_x2,init_x3)); */
/*     rids.push_back(problem.AddResidualBlock(terms.back(),NULL,b)); */
/*   } */

/*   unique_ptr<PiecewiseCallback> callback(new PiecewiseCallback(f,terms)); */
/*   Solver::Options options(opts); */
/*   options.update_state_every_iteration = true; */
/*   options.callbacks.push_back(callback.get()); */

/*   Solver::Summary summary; */
/*   Solve(options, &problem, &summary); */

/*   for (auto rid : rids) { */
/*     problem.RemoveResidualBlock(rid); */
/*   } */
/* } */

/* double remove_small_values(Problem &problem, double *x, double step_mult, double relftol) { */
/*   Solver::Options options; */
/*   solver_opts(options); */
/*   options.function_tolerance = 1e-30; */
/*   options.parameter_tolerance = 1e-30; */
/*   options.gradient_tolerance = 1e-30; */
/*   options.max_num_iterations = 10000; */
/*   options.minimizer_progress_to_stdout = true; */

/*   double nztop = accumulate(x,x+MULT*N,0.0,[](double a, double b) */ 
/*       {return max(a,std::abs(b));} ); */ 

/*   double lo = 0.0, hi=nztop; // nzlow bounds */
/*   double cur = hi - (hi-lo) * step_mult; */
/*   MyTerminationType termination; */
/*   double eps = 1e-4; */

/*   while (hi-lo > 1e-10) { */
/*     cout << lo << " " << hi << endl; */
/*     vector<double> sav(x,x+N*MULT); */
/*     termination = NO_SOLUTION; */
/*     piecewise_solve(problem,options,eps,(nztop-cur)/2-cur,(nztop+cur)/2, */
/*         nztop,[&] (const IterationSummary &s) { */
/*       double relative_decrease = s.cost_change / s.cost; */
/*       if (s.cost < solved_fine) { // sol */
/*         termination = SOLUTION; */
/*         return make_tuple(eps,(nztop-cur)/2,(nztop+cur)/2,nztop,false); */
/*       } else if (relative_decrease > 0 && relative_decrease < relftol) { // no sol */
/*         termination = NO_SOLUTION; */
/*         return make_tuple(eps,(nztop-cur)/2,(nztop+cur)/2,nztop,false); */
/*       } */
/*       return make_tuple(eps,(nztop-cur)/2,(nztop+cur)/2,nztop,true); */
/*     }); */
/*     if (termination == SOLUTION) { */
/*       lo = cur; */
/*       if (hi == nztop) { */
/*         cur = hi - (hi-lo) * step_mult; */
/*       } else { */
/*         cur = (lo+hi) / 2; */
/*       } */
/*     } else { */
/*       hi = cur; */
/*       cur = (lo+hi) / 2; */
/*       copy(sav.begin(),sav.end(),x); */
/*     } */
/*   } */

/*   return (lo+hi)/2; */
/* } */


