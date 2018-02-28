#include "discrete.h"

void greedy_discrete(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    int &successes, DiscreteAttempt da, const int faillimit) {
  vector<int> counts(N);
  /* const double fail_penalty = 0.05; */
  const double fail_penalty = 0.2;
  while (true) {
    vector<tuple<double,cx,int> > vals(N);
    for (int i=0; i<N; ++i) {
      cx target;
      switch (da) {
        case DA_ZERO: target = 0.0; break;
        case DA_PM_ONE: target = x[i*MULT] >= 0 ? 1.0 : -1.0; break;
        case DA_PM_ONE_ZERO: target = std::abs(x[i*MULT]) < 1e-2 ? 0.0 : (x[i*MULT] >= 0 ? 1.0 : -1.0); break;
        case DA_INTEGER: target = std::round(x[i*MULT]); break;
      }
      cx cur = MULT == 1 ? cx(x[i]) : cx(x[i*MULT],x[i*MULT+1]);
      get<0>(vals[i]) = std::abs(cur - target) + fail_penalty * counts[i];
      get<1>(vals[i]) = target;
      get<2>(vals[i]) = i;
    }
    sort(vals.begin(),vals.end(),[](const auto &a,const auto &b) {
        return get<0>(a) < get<0>(b);
    });
    int fails = faillimit;
    vector<double> sav(x,x+N*MULT);
    for (int i=0; i<N; ++i) {
      if (!p.IsParameterBlockConstant(x+MULT*get<2>(vals[i]))) {
        double icost; p.Evaluate(eopts,&icost,0,0,0);
        if (verbose) {
          cout << "successes " << successes
            << " rem " << (faillimit == -1 ? N-i : fails)
            << " lfails " << counts[get<2>(vals[i])]
            << " setting " << "x[" << get<2>(vals[i]) << "] = "
            << get<1>(vals[i]).real();
          if (MULT == 2) cout << ", x[" << get<2>(vals[i]) + 1 << "] = "
            << get<1>(vals[i]).imag();
          cout << "...";
          cout.flush();
        }
        x[get<2>(vals[i])*MULT] = get<1>(vals[i]).real();
        if (MULT == 2) x[get<2>(vals[i])*MULT + 1] = get<1>(vals[i]).imag();
        p.SetParameterBlockConstant(x+MULT*get<2>(vals[i]));
        if (icost < solved_fine) {
          if (verbose) cout << " success free " << endl;
          successes++;
          goto found;
        } else {
          Solver::Summary summary;
          Solve(opts,&p,&summary);
          if (summary.final_cost <= std::max(icost,solved_fine)) { // improved or good enough
            if (verbose) cout << " success " << summary.iterations.size() - 1
                << " iterations" << endl;
            logsol(x,"out_partial_sparse.txt");
            successes++;
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
              << summary.final_cost << endl;
          counts[get<2>(vals[i])]++;
          p.SetParameterBlockVariable(x+MULT*get<2>(vals[i]));
          copy(sav.begin(),sav.end(),x);
          if (faillimit > 0 && fails-- == 0) break;
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
