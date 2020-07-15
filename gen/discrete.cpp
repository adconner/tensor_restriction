#include "discrete.h"

void l2_reg_discrete(Problem &p, double *x, const Solver::Options & opts, const Problem::EvaluateOptions &eopts) {
  if (l2_reg_steps_discrete == 0) return;
  Solver::Options myopts(opts);
  myopts.max_num_iterations = iterations_rough;
  myopts.function_tolerance = ftol_rough;
  /* print_lines=true; */

  double start_alpha = alphastart_discrete;
  while (true) {
    vector<double> sav(x,x+N*MULT);
    sqalpha = std::sqrt(start_alpha); 
    for (int i=l2_reg_steps_discrete; i>0; --i, sqalpha *= std::sqrt(l2_reg_decay_discrete)) {
      if (verbose) {
        cout << "l2 regularization coefficient " << (sqalpha * sqalpha) << ".. "; cout.flush();
      }
      Solver::Summary summary;
      Solve(myopts, &p, &summary);
      if (verbose) {
        cout << summary.initial_cost << " -> " << summary.final_cost << endl; cout.flush();
      }
    }

    sqalpha = 0.0; 
    if (verbose) {
      cout << "getting back to solution.. "; cout.flush();
    }
    myopts.function_tolerance = ftol_rough*0.1;
    Solver::Summary summary;
    Solve(myopts, &p, &summary);
    if (verbose) {
      cout << summary.initial_cost << " -> " << summary.final_cost << endl; cout.flush();
    }

    print_lines = false;
    if (summary.final_cost > solved_fine) {
      copy(sav.begin(),sav.end(),x);
      if (verbose) cout << "l2 reg failed, retrying" << endl;
    } else{
      break;
    }
    start_alpha *= l2_reg_decay_discrete;
  }
}

void greedy_discrete(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    int &successes, DiscreteAttempt da, int faillimit) {
  vector<int> fails(N);
  if (faillimit == -1) faillimit = N - successes;
  while (true) {
    vector<tuple<double,cx,int> > vals(N);
    for (int i=0; i<N; ++i) {
      cx target;
      switch (da) {
        case DA_ZERO: target = 0.0; break;
        case DA_PM_ONE: target = x[i*MULT] >= 0 ? 1.0 : -1.0; break;
        case DA_PM_ONE_ZERO: target = std::abs(x[i*MULT]) < 1e-10 ? 0.0 : (x[i*MULT] >= 0 ? 1.0 : -1.0); break;
        case DA_INTEGER: target = std::round(x[i*MULT]); break;
      }
      cx cur = MULT == 1 ? cx(x[i]) : cx(x[i*MULT],x[i*MULT+1]);
      get<0>(vals[i]) = std::abs(cur - target);
      get<1>(vals[i]) = target;
      get<2>(vals[i]) = i;
    }
    sort(vals.begin(),vals.end(),[&](const auto &a,const auto &b) {
        auto key = [&](const auto &a) {
          double cost; cx target; int i; tie(cost,target,i) = a;
          return make_tuple(!(cost < 1e-13), fails[i], cost + 1.0 * fails[i]);
          /* return make_tuple(!(cost < 1e-13), fails[i], */ 
          /*     ! (i % 25 == 0), */
          /*     ! ((i % 5 == 0) || ((i / 5) % 5 == 0)), */
          /*     i, */ 
          /*     cost ); */
        };
        return key(a) < key(b);
    });
    vector<double> sav(x,x+N*MULT);
    for (int i=0; i<N; ++i) {
      double cost; cx target; int xi; tie(cost,target,xi) = vals[i];
      if (!p.IsParameterBlockConstant(x+MULT*xi)) {
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          /* cout << icost << " "; */
          cout << "successes " << successes
            << " fails " << std::accumulate(fails.begin(),fails.end(),0)
            << " lfails " << fails[xi]
            << " setting " << "x[" << MULT*xi << "] = "
            << target.real();
          if (MULT == 2) cout << ", x[" << MULT*xi + 1 << "] = "
            << target.imag();
          cout << "...";
          cout.flush();
        }
        x[xi*MULT] = target.real();
        if (MULT == 2) x[xi*MULT + 1] = target.imag();
        p.SetParameterBlockConstant(x+MULT*xi);
        double mcost; p.Evaluate(eopts,&mcost,0,0,0);
        if (mcost < std::max(better_frac*icost,solved_fine)) {
          if (verbose) cout << " success free " << icost << endl;
          successes++;
          goto found;
        } else {
          Solver::Summary summary;
          Solve(opts,&p,&summary);
          if (summary.final_cost <= std::max(better_frac*icost,solved_fine) 
              && *max_element(x,x+N*MULT) < max_elem) { // improved or good enough
            if (verbose) cout << " success " << summary.iterations.size() - 1
                << " iterations " << summary.final_cost << endl;
            logsol(x,"out_partial_sparse.txt");
            successes++;
            l2_reg_discrete(p,x,opts,eopts);
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
              << summary.final_cost << endl;
          p.SetParameterBlockVariable(x+MULT*xi);
          fails[xi]++;
          copy(sav.begin(),sav.end(),x);
          if (std::accumulate(fails.begin(),fails.end(),0) == faillimit) break;
        }
      }
    }
    break;
    found:;
  }
}

void greedy_discrete_pairs(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    const int faillimit) {
  set<pair<int,int> > fixed;
  while (true) {
    vector<pair<double,pair<int,int> > > vals(N * (N-1) / 2);
    int ix = 0;
    for (int i=0; i<N; ++i) {
      for (int j=i+1; j<N; ++j) {
        vals[ix].second.first = i*MULT;
        vals[ix].second.second = j*MULT;
        for (int k=0; k<MULT; ++k) {
          double dx = x[i*MULT + k] - x[j*MULT + k];
          vals[ix].first += dx*dx;
        }
        ix++;
      }
    }
    sort(vals.begin(),vals.end());
    int fails = faillimit;
    vector<double> sav(x,x+N*MULT);
    for (int i=0; i<vals.size(); ++i) {
      if (!p.IsParameterBlockConstant(x+vals[i].second.first) &&
          !p.IsParameterBlockConstant(x+vals[i].second.second) && 
          !fixed.count(vals[i].second)) {
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          cout << "cost " << icost << " attempting to set "; 
          cout << "x[" << vals[i].second.first << "] = x[" 
            << vals[i].second.second << "]...";
          cout.flush();
        }
        auto rid = p.AddResidualBlock(new LinearCombination(1.0,-1.0), NULL, 
            {x+vals[i].second.first, x+vals[i].second.second});
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        if (summary.final_cost <= std::max(icost,solved_fine)) {
          fixed.insert(vals[i].second);
          if (verbose) cout << " success " << summary.iterations.size() - 1
              << " iterations" << endl;
          logsol(x,"out_partial_sparse.txt");
          goto found;
        }
        if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
          << summary.final_cost << endl;
        p.RemoveResidualBlock(rid);
        copy(sav.begin(),sav.end(),x);
        if (faillimit > 0 && fails-- == 0) break;
      }
    }
    break;
    found:;
  }
}
