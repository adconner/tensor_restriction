#include "discrete.h"
#include "l2reg.h"

void greedy_discrete(Problem &p, double *x, 
    const Solver::Options & opts, const Problem::EvaluateOptions &eopts,
    int &successes, DiscreteAttempt da, int trylimit) {
  auto get_target = [&](cx cur) {
    // if a search over only reals, the real part of the target is used
    if (da == DA_ZERO || abs(cur) < 1e-10) return cx(0.0);
    vector<cx> targets;
    if (da == DA_PM_ONE) {
      targets = { {1.0, 0.0}, {-1.0, 0.0} };
    } else if (da == DA_E3) {
      targets = { {1.0,0.0}, {-0.5, 0.866025403784439}, {-0.5, -0.866025403784439} };
      for (int i=0; i<3; ++i) {
        targets.push_back(-targets[i]);
      }
      /* targets.push_back(cx(0.0)); */
    }
    double smallest = 1e7;
    cx target = 0.0;
    for (cx tar : targets) {
      double dist = abs(cur - tar);
      if (dist < smallest) {
        smallest = dist;
        target = tar;
      }
    }
    return target;
  };
  int tries = 0;
  vector<int> fails(N);
  while (true) {
    vector<tuple<double,cx,int> > vals(N);
    for (int i=0; i<N; ++i) {
      cx cur = MULT == 1 ? cx(x[i]) : cx(x[i*MULT],x[i*MULT+1]);
      cx target = get_target(cur);
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
          tries++;
          if (summary.final_cost <= std::max(better_frac*icost,solved_fine) 
              && *max_element(x,x+N*MULT) < max_elem) { // improved or good enough
            if (verbose) cout << " success " << summary.iterations.size() - 1
                << " iterations " << summary.final_cost << endl;
            successes++;
            l2_reg_refine(p,x,opts);
            logsol(x,"out_partial_sparse.txt");
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
              << summary.final_cost << endl;
          p.SetParameterBlockVariable(x+MULT*xi);
          fails[xi]++;
          copy(sav.begin(),sav.end(),x);
          if (tries >= trylimit) break;
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
          l2_reg_refine(p,x,opts);
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

class ContainedOnLine : public SizedCostFunction<1,MULT> {
  public:
    ContainedOnLine(cx a_) : a(a_) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      assert(MULT == 2);
      residuals[0] = x[0][0]*a.imag() - x[0][1]*a.real();
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = a.imag();
          jacobians[0][1] = -a.real();
        }
      }
      return true;
    }
  private:
    const cx a;
};

void greedy_discrete_lines(Problem &p, double *x, 
    const Solver::Options & opts, int ei, int trylimit) {
  assert(MULT == 2);
  auto get_target = [&](cx cur) {
    vector<cx> targets; //{ {1.0,0.0}, {-0.5, 0.866025403784439}, {-0.5, -0.866025403784439} };
    double pi = std::atan(1)*4;
    for (int i=0; i < ei%2 ? ei : ei/2 ; ++i) {
      targets.push_back(cx(std::cos(2*pi/ei),std::sin(2*pi/ei)));
    }
    double dist = 1e15;
    cx target = cx(0);
    for (cx a: targets) {
      double curdist = std::abs(a.imag()*cur.real()-a.real()*cur.imag());
      if (curdist < dist) {
        dist = curdist;
        target = a;
      }
    }
    return target;
  };
  vector<int> fails(N);
  int tries = 0;
  set<int> successes;
  while (true) {
    vector<tuple<double,cx,int> > vals(N);
    for (int i=0; i<N; ++i) {
      cx cur = cx(x[i*MULT],x[i*MULT+1]);
      cx target = get_target(cur);
      get<0>(vals[i]) = std::abs(target.imag()*cur.real()-target.real()*cur.imag());
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
      if (!p.IsParameterBlockConstant(x+MULT*xi) && !successes.count(xi)) {
        double icost; p.Evaluate(Problem::EvaluateOptions(),&icost,0,0,0);
        if (verbose) {
          /* cout << icost << " "; */
          cout << "successes " << successes.size()
            << " fails " << std::accumulate(fails.begin(),fails.end(),0)
            << " lfails " << fails[xi]
            << " setting " << target.imag() << "*x[" << MULT*xi << "] + "
            << -target.real() << "*x[" << MULT*xi + 1 << "] = 0...";
          cout.flush();
        }
        x[xi*MULT] = target.real();

        auto rid = p.AddResidualBlock(new ContainedOnLine(target), NULL, {x+MULT*i});
        p.Evaluate(Problem::EvaluateOptions(),&cost,0,0,0);
        if (cost < 1e-20) {
          if (verbose) cout << " success free " << cost << endl;
          successes.insert(xi);
          goto found;
        } else {
          Solver::Summary summary;
          Solve(opts,&p,&summary);
          tries++; 
          if (summary.final_cost <= std::max(better_frac*icost,solved_fine)) {
            if (verbose) cout << " success " << summary.iterations.size() - 1
                << " iterations " << summary.final_cost << endl;
            successes.insert(xi);
            l2_reg_refine(p,x,opts);
            logsol(x,"out_partial_sparse.txt");
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
            << summary.final_cost << endl;
          fails[xi]++;
          p.RemoveResidualBlock(rid);
          copy(sav.begin(),sav.end(),x);
          if (tries >= trylimit) break;
        }
      }
    }
    break;
    found:;
    if (tries >= trylimit) break;
  }
}
