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
    int &successes, DiscreteAttempt da, int trylimit) {
  vector<int> fails(N);
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
          return make_tuple(!(cost < 1e-13), fails[i], 
              /* (i % 25 == 0), */
              /* ((i % 5 == 0) || ((i / 5) % 5 == 0)), */
              /* ! (i % 25 == 0), */
              /* ! ((i % 5 == 0) || ((i / 5) % 5 == 0)), */
              /* i, */ 
              cost );
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
            successes++;
            l2_reg_discrete(p,x,opts,eopts);
            logsol(x,"out_partial_sparse.txt");
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
              << summary.final_cost << endl;
          p.SetParameterBlockVariable(x+MULT*xi);
          fails[xi]++;
          copy(sav.begin(),sav.end(),x);
          if (std::accumulate(fails.begin(),fails.end(),0)+successes >= trylimit) break;
        }
      }
    }
    break;
    found:;
  }
}

void greedy_discrete_pairs(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    const int trylimit) {
  int successes = 0; // not from outside because these dont stack with outside
  map<int,int> vix;
  vector<int> vs;
  for (int i=0; i<N; ++i) {
    if (!p.IsParameterBlockConstant(x+MULT*i)) {
      vix[i] = vs.size();
      vs.push_back(i);
    }
  }
  for (auto it = vix.begin(); it != vix.end(); ++it) {
    vs[it->second] = it->first;
  }

  vector<int> par(vs.size()); for (int i=0; i<vs.size(); ++i) par[i] = i;
  function<int(int)> find = [&](int i) {
    if (par[i] == i) return i;
    int j = find(par[i]);
    par[i] = j;
    return j;
  };
  auto unio = [&](int i, int j) {
    i = find(i); j = find(j);
    if ((i+j) % 2)
      par[i] = j;
    else
      par[j] = i;
  };

  int tries = 0;
  while (true) {
    map<int, cx> vals;
    for (int i=0; i<vs.size(); ++i) {
      int j = find(i);
      vals[j] = MULT == 1 ? cx(x[i]) : cx(x[j*MULT],x[j*MULT+1]);
    }

    vector<tuple<double,int,int,double,double> > diffs;
    vector<pair<double, double> > alphabetas { {1.0, -1.0}, {1.0, 1.0} };
    for (auto it=vals.begin(); it != vals.end(); ++it) {
      auto jt = it; ++jt;
      for (auto ab : alphabetas) {
        for (; jt != vals.end(); ++jt) {
          diffs.push_back(make_tuple(std::abs(ab.first*it->second + ab.second*jt->second),
                std::min(it->first,jt->first),std::max(it->first,jt->first), 
                ab.first, ab.second));
        }
      }
    }
    int skip = 1;
    int mid = min(skip,(int)diffs.size());
    partial_sort(diffs.begin(), diffs.begin() + mid, diffs.end());

    vector<double> sav(x,x+N*MULT);
    for (int ii=0; ii<diffs.size(); ++ii) {
      if (ii == mid) {
        mid += skip; mid = min(mid,(int)diffs.size());
        skip *= 2;
        partial_sort(diffs.begin()+ii, diffs.begin() + mid, diffs.end());
      }

      const auto &curdiff = diffs[ii];
      double diff, alpha, beta; int i,j; tie(diff,i,j,alpha,beta) = curdiff;
      double icost; p.Evaluate(Problem::EvaluateOptions(),&icost,0,0,0);
      if (verbose) {
        cout << "successes " << successes 
          << " rem " << (trylimit - tries)
          /* << " unfixed " << vals.size() */ 
          << " setting " << alpha << "*x[" << i << "] + " 
          << beta << "*x[" << j << "] = 0...";
        cout.flush();
      }
      auto rid = p.AddResidualBlock(new LinearCombination(alpha,beta), NULL, {x+MULT*i, x+MULT*j});
      // evaluation seems not to work before resolving, so use diff directly to detect free relations
      if (diff < 1e-13) {
        if (verbose) cout << " success free " << diff << endl;
        unio(i,j);
        successes++;
        goto found;
      } else {
        Solver::Summary summary;
        Solve(opts,&p,&summary);
        tries++; 
        if (summary.final_cost <= std::max(better_frac*icost,solved_fine)) {
          if (verbose) cout << " success " << summary.iterations.size() - 1
              << " iterations " << summary.final_cost << endl;
          unio(i,j);
          successes++;
          l2_reg_discrete(p,x,opts,eopts);
          logsol(x,"out_partial_sparse.txt");
          goto found;
        }
        if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
          << summary.final_cost << endl;
        p.RemoveResidualBlock(rid);
        copy(sav.begin(),sav.end(),x);
        if (tries >= trylimit) break;
      }
    }
    break;
    found:;
    if (tries >= trylimit) break;
  }
}
