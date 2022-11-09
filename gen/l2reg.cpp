#include "l2reg.h"
#include <iterator>
#include <algorithm>
#include <functional>

typedef function<MyTerminationType(const IterationSummary&)> upf;

class FunctorCallback : public IterationCallback {
  public:
    FunctorCallback(upf f) : f_(f) {}
    CallbackReturnType operator()(const IterationSummary& summary) {
      termination_ = f_(summary);
      return termination_ == CONTINUE ? SOLVER_CONTINUE : SOLVER_TERMINATE_SUCCESSFULLY;
    }

    upf f_;
    MyTerminationType termination_;
};

MyTerminationType solve(MyProblem &p, Solver::Summary &summary, 
    double relftol, int max_num_iterations) {
  if (!count(p.variable_mask.begin(),p.variable_mask.end(),true)) {
    return UNKNOWN;
  }
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = max_num_iterations;

  unique_ptr<FunctorCallback> callback(new FunctorCallback([&](const IterationSummary &s){
      /* if (verbose) { */
      /*   double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) */ 
      /*       {return max(a,std::abs(b));} ); */ 
      /*   printf("%3d %20.15g %20.15f %10.5g %10.5g\n", s.iteration,2*s.cost,ma,s.relative_decrease,s.trust_region_radius); */
      /* } */

      double relative_decrease = s.cost_change / s.cost;
      if (s.cost < solved_fine) {
        return SOLUTION;
      }
      if (relative_decrease > 0 && relative_decrease < relftol) {
        return s.cost < 1e-2 ? BORDER_LIKELY : NO_SOLUTION;
      }
      return CONTINUE;
    }));
  options.update_state_every_iteration = true;
  options.callbacks.push_back(callback.get());

  Solve(options, &p.p, &summary);

  return callback->termination_ == CONTINUE ? UNKNOWN : callback->termination_;
}

struct L2Regularization : public CostFunction {
  L2Regularization(int block_size, double *sqalpha, double *b) : 
      block_size_(block_size), sqalpha_(sqalpha), b_(b) {
    *mutable_parameter_block_sizes() = {MULT*block_size};
    set_num_residuals(MULT*block_size);
  }
  bool Evaluate(const double* const* x,
      double* residuals,
      double** jacobians) const {
    if (jacobians && jacobians[0]) {
      fill(jacobians[0],jacobians[0]+num_residuals()*num_residuals(),0.0);
    }
    for (int i=0; i<block_size_; ++i) {
      if (MULT == 1) {
        double tar = max(min(x[0][i],b_[i]),-b_[i]);
        residuals[i] = sqalpha_[i]*(x[0][i]-tar);
        if (jacobians && jacobians[0]) {
          jacobians[0][i*num_residuals()+i] = abs(x[0][i]) <= b_[i] ? 0 : sqalpha_[i];
        }
      } else { // MULT == 2
        cx xc(x[0][2*i],x[0][2*i+1]);
        if (abs(xc) <= b_[i]) {
          residuals[i*2] = residuals[i*2+1] = 0;
          if (jacobians && jacobians[0]) {
            jacobians[0][2*i*num_residuals()+2*i] = 0.0;
            jacobians[0][(2*i+1)*num_residuals()+2*i+1] = 0.0;
          }
        } else {
          cx tar = xc / abs(xc);
          cx r = sqalpha_[i]*(xc-tar);
          residuals[i*2] = real(r);
          residuals[i*2+1] = imag(r);
          if (jacobians && jacobians[0]) {
            jacobians[0][2*i*num_residuals()+2*i] = sqalpha_[i];
            jacobians[0][(2*i+1)*num_residuals()+2*i+1] = sqalpha_[i];
          }
        }
      }
    }
    return true;
  }
  int block_size_;
  double *sqalpha_;
  double *b_;
};

MyTerminationType l2_reg(MyProblem &p, const Solver::Options &opts, double *sqalpha, double *b, upf f) {
  if (!count(p.variable_mask.begin(),p.variable_mask.end(),true)) {
    return UNKNOWN;
  }
  vector<ResidualBlockId> rids;
  for (int i=0; i<BLOCKS; ++i) {
    rids.push_back(p.p.AddResidualBlock(
      new L2Regularization(BBOUND[i+1]-BBOUND[i],sqalpha+BBOUND[i],b+BBOUND[i])
          ,NULL,p.x.data()+MULT*BBOUND[i]));
  }

  unique_ptr<FunctorCallback> callback(new FunctorCallback(f));
  Solver::Options options(opts);
  options.update_state_every_iteration = true;
  options.callbacks.push_back(callback.get());

  do {
    Solver::Summary summary;
    Solve(options, &p.p, &summary);
  } while (callback->termination_ == CONTINUE_RESET);

  for (auto rid : rids) {
    p.p.RemoveResidualBlock(rid);
  }
  return callback->termination_ == CONTINUE ? UNKNOWN : callback->termination_;
}

MyTerminationType l2_reg_search(MyProblem &p, double target_relative_decrease, 
    double relftol, bool stop_on_br, int max_num_iterations, double sqinit) {
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = max_num_iterations;

  double sqalpha_mult = 0.5;
  deque<bool> recent_drop;

  double ma_last = 1.0;
  int consecutive_border_evidence = 0;

  vector<double> sqalpha(N,sqinit), b(N,0.0);
  return l2_reg(p,options,sqalpha.data(),b.data(),
        [&] (const IterationSummary &s) {
      double relative_decrease = s.cost_change / s.cost;
      double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
          {return max(a,std::abs(b));} ); 

      if (verbose) {
        printf("%3d %20.15g %20.15f %10.5g %-+12.6g %11.6g %20.15g %d\n",
            s.iteration,2*s.cost,ma,s.step_norm,relative_decrease,sqalpha[0],s.relative_decrease,consecutive_border_evidence);
      }

      if (s.cost < solved_fine) {
        return SOLUTION;
      }
      if (sqalpha[0] == 0.0 && relative_decrease > 0 && relative_decrease < relftol) {
        return s.cost < 2e-3 ? BORDER_LIKELY : NO_SOLUTION;
      }
      if (stop_on_br && s.cost < 0.2) { // consider border solution
        // border rank
        // f low and df/f also low
        // ma consistently increasing
        if (ma > ma_last && ma > 2 && relative_decrease < 0.1) {
          consecutive_border_evidence += 1;
        } else {
          consecutive_border_evidence = 0;
        }
        if (ma > 10 || consecutive_border_evidence >= 6) {
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

double minimize_max_abs(MyProblem &p, double eps, double step_mult, double relftol) {
  Solver::Options options;
  solver_opts(options);
  options.function_tolerance = 1e-30;
  options.parameter_tolerance = 1e-30;
  options.gradient_tolerance = 1e-30;
  options.max_num_iterations = 10000;
  /* options.minimizer_progress_to_stdout = true; */

  double lo = 0.0;
  double hi = max_abs(p);
  double cur = hi*step_mult;

  vector<double> sqalpha(N,1.0), b(N,1.0);
  while (hi-lo > eps) {
    /* cout << lo << " " << hi << endl; */
    vector<double> sav(p.x.begin(),p.x.end());
    for (int i=0; i<b.size(); ++i)
      b[i] = cur;
    MyTerminationType termination = l2_reg(p,options,
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
      copy(sav.begin(),sav.end(),p.x.begin());
    }
  }

  return hi;
}


void sparsify(MyProblem &p, double B, double relftol) {
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
    vector<double> sav(p.x.begin(),p.x.end());
    MyTerminationType termination;
    for (int tr = 1; tr <= tries; ++tr) {
      // TODO do while smallest n are changing

      vector<pair<double,int> > xabs(N);
      for (int i=0; i<N; ++i) {
        xabs[i] = make_pair(MULT==1?abs(p.x[i]):abs(cx(p.x[2*i],p.x[2*i+1])),i);
      }
      assert(cur >= 1);
      nth_element(xabs.begin(),xabs.begin()+cur-1,xabs.end());
      fill(sqalpha.begin(),sqalpha.end(),1.0);
      fill(b.begin(),b.end(),B);
      for (int i=0; i < cur; ++i) {
        b[xabs[i].second] = 0.0;
        sqalpha[xabs[i].second] = 1e-2;
      }

      termination = l2_reg(p,options,sqalpha.data(),b.data(),
          [&] (const IterationSummary &s) {
        double relative_decrease = s.cost_change / s.cost;

        if (verbose) {
          double ma = accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) 
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
      /* minimize_max_abs(p, x, 1e-1, 0.8, relftol); */
    } else {
      hi = cur-1;
      copy(sav.begin(),sav.end(),p.x.begin());
    }
  }
  /* minimize_max_abs(p, x, 1e-14, 0.8, relftol); */
}

