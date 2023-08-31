#include "discrete.h"
#include "l2reg.h"

/* #define EQ_DISCRETE */
const double discrete_sqalpha = 1e4;

class Equal : public SizedCostFunction<MULT,MULT> {
  public:
    Equal(cx _x0, double _sqalpha) : x0(_x0), sqalpha(_sqalpha) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = sqalpha*(x[0][0]-x0.real());
      if (MULT == 2) residuals[1] = sqalpha*(x[0][1]-x0.imag());
      if (jacobians && jacobians[0]) {
        jacobians[0][0] = sqalpha;
        if (MULT == 2) {
          jacobians[0][1] = 0;
          jacobians[0][2] = 0;
          jacobians[0][3] = sqalpha;
        }
      }
      return true;
    }
    cx x0;
    double sqalpha;
};

void greedy_discrete(MyProblem &p, int &successes, DiscreteAttempt da, int trylimit) {
  auto get_target = [&](cx cur) {
    // if a search over only reals, the real part of the target is used
    if (da == DA_ZERO || abs(cur) < 1e-10) return cx(0.0);
    vector<cx> targets;
    if (da == DA_PM_ONE) {
      targets = { {1.0, 0.0}, {-1.0, 0.0} };
    } else if (da == DA_E3) {
      int k = 6;
      double pi = 4*atan(1.0);
      for (int i=0; i<k; ++i) {
        targets.push_back({cos(2*pi/k), sin(2*pi/k)});
        /* targets.push_back({cos(2*pi/k)/2, sin(2*pi/k)/2}); */
        /* targets.push_back({cos(2*pi/k)*2, sin(2*pi/k)*2}); */
        /* targets.push_back({cos(2*pi/k)/4, sin(2*pi/k)/4}); */
        /* targets.push_back({cos(2*pi/k)*4, sin(2*pi/k)*4}); */
        /* targets.push_back({cos(2*pi/k)/3, sin(2*pi/k)/3}); */
        /* targets.push_back({cos(2*pi/k)*6, sin(2*pi/k)*6}); */
        /* targets.push_back({cos(2*pi/k)/sqrt(2), sin(2*pi/k)/sqrt(2)}); */
        /* targets.push_back({cos(2*pi/k)*sqrt(2), sin(2*pi/k)*sqrt(2)}); */
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
      cx cur = MULT == 1 ? cx(p.x[i]) : cx(p.x[i*MULT],p.x[i*MULT+1]);
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
    vector<double> sav(p.x.begin(),p.x.end());
    for (int i=0; i<N; ++i) {
      double cost; cx target; int xi; tie(cost,target,xi) = vals[i];
      if (p.variable_mask[xi]) {
        double icost; p.p.Evaluate(Problem::EvaluateOptions(),&icost,0,0,0);
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
#ifdef EQ_DISCRETE
        auto rid = p.AddResidualBlock(new Equal(target,1.0), NULL, {x+MULT*xi});
#else
        p.x[xi*MULT] = target.real();
        if (MULT == 2) p.x[xi*MULT + 1] = target.imag();
        set_value_constant(p,xi);
#endif
        double mcost; p.p.Evaluate(Problem::EvaluateOptions(),&mcost,0,0,0);
        if (mcost < std::max(better_frac*icost,solved_fine)) {
          if (verbose) cout << " success free " << icost << endl;
#ifdef EQ_DISCRETE
          x[xi*MULT] = target.real();
          if (MULT == 2) x[xi*MULT + 1] = target.imag();
          p.SetParameterBlockConstant(x+MULT*xi);
          p.RemoveResidualBlock(rid);
#endif
          successes++;
          goto found;
        } else {
#ifdef EQ_DISCRETE
          ((Equal *)p.GetCostFunctionForResidualBlock(rid))->sqalpha = discrete_sqalpha;
#endif
          Solver::Summary summary;
          solve(p,summary,5e-4,200);
          /* solve(p,summary,1e-6,4000); */
          tries++;
          // If we have set constant the last variable the solver will not be
          // called and will terminate with FAILURE and summary will be
          // otherwise uninitialized. Don't count this as a success here if this
          // happens (if it is good assignment it will usually be a success free)
          if (summary.termination_type != FAILURE 
              && summary.final_cost <= std::max(better_frac*icost,solved_fine) 
              && max_abs(p) < max_elem) { // improved or good enough
            if (verbose) cout << " success " << summary.iterations.size() - 1
                << " iterations " << summary.final_cost << " ma " 
                << max_abs(p) << endl;
#ifdef EQ_DISCRETE
            x[xi*MULT] = target.real();
            if (MULT == 2) x[xi*MULT + 1] = target.imag();
            p.SetParameterBlockConstant(x+MULT*xi);
            p.RemoveResidualBlock(rid);
#endif
            successes++;
            minimize_max_abs(p,1e-2);
            /* l2_reg_search(...) */
            logsol(p.x,"out_partial_sparse.txt");
            goto found;
          }
          if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
              << summary.final_cost << " ma " 
              << max_abs(p) << endl;
#ifdef EQ_DISCRETE
          p.RemoveResidualBlock(rid);
#else
          set_value_variable(p,xi);
#endif
          fails[xi]++;
          copy(sav.begin(),sav.end(),p.x.begin());
          if (tries >= trylimit) break;
        }
      }
    }
    break;
    found:;
  }
}

void greedy_discrete_careful(MyProblem &p, int &successes, DiscreteAttempt da) {
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

  double last_cost = max_abs(p);
  vector<int> fails(N);
  while (true) {
    vector<tuple<double,cx,int> > vals(N);
    for (int i=0; i<N; ++i) {
      cx cur = MULT == 1 ? cx(p.x[i]) : cx(p.x[i*MULT],p.x[i*MULT+1]);
      cx target = get_target(cur);
      get<0>(vals[i]) = std::abs(cur - target);
      get<1>(vals[i]) = target;
      get<2>(vals[i]) = i;
    }
    sort(vals.begin(),vals.end(),[&](const auto &a,const auto &b) {
        auto key = [&](const auto &a) {
          double cost; cx target; int i; tie(cost,target,i) = a;
          return make_tuple(!(cost < 1e-13), fails[i], 
              cost );
        };
        return key(a) < key(b);
    });

    vector<tuple<double,int,vector<double> > > ok;
    vector<double> sav(p.x.begin(),p.x.end());
    for (int vi=0; vi<N; ++vi) {
      double cost; cx target; int xi; tie(cost,target,xi) = vals[vi];
    /* for (int xi=0; xi<N; ++xi) { */
      if (!p.variable_mask[xi]) continue;
      cout << xi;
      /* cx cur = MULT == 1 ? cx(p.x[xi]) : cx(p.x[xi*MULT],p.x[xi*MULT+1]); */
      /* cx target = get_target(cur); */

      double icost; p.p.Evaluate(Problem::EvaluateOptions(),&icost,0,0,0);

      p.x[xi*MULT] = target.real();
      if (MULT == 2) p.x[xi*MULT + 1] = target.imag();
      set_value_constant(p,xi);

      Solver::Summary summary;
      solve(p,summary,1e-3,100);

      if (summary.termination_type != FAILURE 
          && summary.final_cost <= std::max(better_frac*icost,solved_fine) 
          && max_abs(p) < max_elem) { // improved or good enough
        minimize_max_abs(p,1e-1);
        double cost = max_abs(p);
        ok.push_back(make_tuple(cost,xi,p.x));
        cout << "s ";
        // can optionally break if a sufficiently good guy is seen to continue
        // immediately
        cout << cost << " ";
        if (cost <= last_cost*1.001) break;
      } else {
        fails[xi]++;
        cout << "f ";
      }
      cout.flush();

      set_value_variable(p,xi);
      copy(sav.begin(),sav.end(),p.x.begin());
    }
    if (ok.empty()) {
      return;
    }
    double cost; int xi; vector<double> x;
    tie(cost,xi,x) = *min_element(ok.begin(),ok.end());
    /* // this is situational */
    /* if (cost > last_cost*1.001) { */
    /*   return; */
    /* } */
    p.x = x;
    set_value_constant(p,xi);

    int free = 0, successes = 0;
    for (int i=0; i<N; ++i) {

      if (!p.variable_mask[i]) {
        successes++;
        continue;
      }
      cx cur = MULT == 1 ? cx(p.x[i]) : cx(p.x[i*MULT],p.x[i*MULT+1]);
      cx target = get_target(cur);
      if (abs(cur - target) < solved_fine) {
        p.x[i*MULT] = target.real();
        if (MULT == 2) p.x[i*MULT + 1] = target.imag();
        set_value_constant(p,i);
        free++;
        successes++;
      }
    }

    logsol(p.x,"out_partial_sparse.txt");
    cout << "\nsuccesses " << successes << " new max " << cost << " free equations " << free << endl;
    last_cost = cost;

  }
}




class LinearCombination : public SizedCostFunction<MULT,MULT,MULT> {
  public:
    LinearCombination(double _a, double _b, double _sqalpha = 1.0)
      : a(_a), b(_b), sqalpha(_sqalpha) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = sqalpha*(a*x[0][0] + b*x[1][0]);
      if (MULT == 2) residuals[1] = sqalpha*(a*x[0][1] + b*x[1][1]);
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = sqalpha*a;
          if (MULT == 2) {
            jacobians[0][1] = 0;
            jacobians[0][2] = 0;
            jacobians[0][3] = sqalpha*a;
          }
        }
        if (jacobians[1]) {
          jacobians[1][0] = sqalpha*b;
          if (MULT == 2) {
            jacobians[1][1] = 0;
            jacobians[1][2] = 0;
            jacobians[1][3] = sqalpha*b;
          }
        }
      }
      return true;
    }
    double a,b;
    double sqalpha;
};

void greedy_discrete_pairs(MyProblem &p, const int trylimit) {
  int successes = 0; // not from outside because these dont stack with outside
  map<int,int> vix;
  vector<int> vs;
  for (int i=0; i<N; ++i) {
    if (p.variable_mask[i]) {
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
      vals[j] = MULT == 1 ? cx(p.x[i]) : cx(p.x[j*MULT],p.x[j*MULT+1]);
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

    vector<double> sav(p.x.begin(),p.x.end());
    for (int ii=0; ii<diffs.size(); ++ii) {
      if (ii == mid) {
        mid += skip; mid = min(mid,(int)diffs.size());
        skip *= 2;
        partial_sort(diffs.begin()+ii, diffs.begin() + mid, diffs.end());
      }

      const auto &curdiff = diffs[ii];
      double diff, alpha, beta; int i,j; tie(diff,i,j,alpha,beta) = curdiff;
      double icost; p.p.Evaluate(Problem::EvaluateOptions(),&icost,0,0,0);
      if (verbose) {
        cout << "successes " << successes 
          << " rem " << (trylimit - tries)
          /* << " unfixed " << vals.size() */ 
          << " setting " << alpha << "*x[" << i << "] + " 
          << beta << "*x[" << j << "] = 0...";
        cout.flush();
      }
      auto rid = p.p.AddResidualBlock(new LinearCombination(alpha,beta), NULL, 
          {p.x.data()+MULT*i, p.x.data()+MULT*j});
      double mcost; p.p.Evaluate(Problem::EvaluateOptions(),&mcost,0,0,0);
      if (mcost < std::max(better_frac*icost,solved_fine)) {
        if (verbose) cout << " success free " << diff << endl;
        unio(i,j);
        successes++;
        goto found;
      } else {
        Solver::Summary summary;
        ((LinearCombination*)p.p.GetCostFunctionForResidualBlock(rid))->sqalpha = discrete_sqalpha;
        solve(p,summary,1e-3,100);
        ((LinearCombination*)p.p.GetCostFunctionForResidualBlock(rid))->sqalpha = 1.0;
        tries++; 
        if (summary.final_cost <= std::max(better_frac*icost,solved_fine)) {
          if (verbose) cout << " success " << summary.iterations.size() - 1
            << " iterations " << summary.final_cost << endl;
          unio(i,j);
          successes++;
          minimize_max_abs(p,1e-1);
          logsol(p.x,"out_partial_sparse.txt");
          goto found;
        }
        if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations "
          << summary.final_cost << endl;
        p.p.RemoveResidualBlock(rid);
        copy(sav.begin(),sav.end(),p.x.begin());
        if (tries >= trylimit) break;
      }
    }
    break;
found:;
      if (tries >= trylimit) break;
  }
}


// havnt ported below to new rewrite

/* class ContainedOnLine : public SizedCostFunction<1,MULT> { */
/*   public: */
/*     ContainedOnLine(cx _a, double _sqalpha=1.0) : a(_a), sqalpha(_sqalpha) {} */
/*     bool Evaluate(const double* const* x, */
/*         double* residuals, */
/*         double** jacobians) const { */
/*       assert(MULT == 2); */
/*       residuals[0] = sqalpha*(x[0][0]*a.imag() - x[0][1]*a.real()); */
/*       if (jacobians) { */
/*         if (jacobians[0]) { */
/*           jacobians[0][0] = sqalpha*a.imag(); */
/*           jacobians[0][1] = -sqalpha*a.real(); */
/*         } */
/*       } */
/*       return true; */
/*     } */
/*     cx a; */
/*     double sqalpha; */
/* }; */

/* void greedy_discrete_lines(Problem &p, double *x, */ 
/*     const Solver::Options & opts, int ei, int trylimit) { */
/*   assert(MULT == 2); */
/*   auto get_target = [&](cx cur) { */
/*     vector<cx> targets(ei%2 ? ei : ei/2); */
/*     double pi = std::atan(1)*4; */
/*     for (int i=0; i < targets.size() ; ++i) { */
/*       targets[i] = cx(std::cos(2*pi*i/ei),std::sin(2*pi*i/ei)); */
/*     } */
/*     double dist = 1e15; */
/*     cx target = cx(0); */
/*     for (cx a: targets) { */
/*       double curdist = std::abs(a.imag()*cur.real()-a.real()*cur.imag()); */
/*       if (curdist < dist) { */
/*         dist = curdist; */
/*         target = a; */
/*       } */
/*     } */
/*     return target; */
/*   }; */
/*   vector<int> fails(N); */
/*   int tries = 0; */
/*   set<int> successes; */
/*   while (true) { */
/*     vector<tuple<double,cx,int> > vals(N); */
/*     for (int i=0; i<N; ++i) { */
/*       cx cur = cx(x[i*MULT],x[i*MULT+1]); */
/*       cx target = get_target(cur); */
/*       get<0>(vals[i]) = std::abs(target.imag()*cur.real()-target.real()*cur.imag()); */
/*       get<1>(vals[i]) = target; */
/*       get<2>(vals[i]) = i; */
/*     } */
/*     sort(vals.begin(),vals.end(),[&](const auto &a,const auto &b) { */
/*         auto key = [&](const auto &a) { */
/*           double cost; cx target; int i; tie(cost,target,i) = a; */
/*           return make_tuple(!(cost < 1e-13), fails[i], */ 
/*               /1* (i % 25 == 0), *1/ */
/*               /1* ((i % 5 == 0) || ((i / 5) % 5 == 0)), *1/ */
/*               /1* ! (i % 25 == 0), *1/ */
/*               /1* ! ((i % 5 == 0) || ((i / 5) % 5 == 0)), *1/ */
/*               /1* i, *1/ */ 
/*               cost ); */
/*         }; */
/*         return key(a) < key(b); */
/*     }); */
/*     vector<double> sav(x,x+N*MULT); */
/*     for (int i=0; i<N; ++i) { */
/*       double cost; cx target; int xi; tie(cost,target,xi) = vals[i]; */
/*       if (!p.IsParameterBlockConstant(x+MULT*xi) && !successes.count(xi)) { */
/*         double icost; p.Evaluate(Problem::EvaluateOptions(),&icost,0,0,0); */
/*         if (verbose) { */
/*           /1* cout << icost << " "; *1/ */
/*           cout << "successes " << successes.size() */
/*             << " fails " << std::accumulate(fails.begin(),fails.end(),0) */
/*             << " lfails " << fails[xi] */
/*             << " setting " << target.imag() << "*x[" << MULT*xi << "] + " */
/*             << -target.real() << "*x[" << MULT*xi + 1 << "] = 0..."; */
/*           cout.flush(); */
/*         } */
/*         x[xi*MULT] = target.real(); */

/*         auto rid = p.AddResidualBlock(new ContainedOnLine(target), NULL, {x+MULT*xi}); */
/*         double mcost; p.Evaluate(Problem::EvaluateOptions(),&mcost,0,0,0); */
/*         if (mcost < std::max(better_frac*icost,solved_fine)) { */
/*           if (verbose) cout << " success free " << mcost << endl; */
/*           successes.insert(xi); */
/*           goto found; */
/*         } else { */
/*           Solver::Summary summary; */
/*           ((ContainedOnLine*)p.GetCostFunctionForResidualBlock(rid))->sqalpha = discrete_sqalpha; */
/*           Solve(opts,&p,&summary); */
/*           ((ContainedOnLine*)p.GetCostFunctionForResidualBlock(rid))->sqalpha = 1.0; */
/*           tries++; */ 
/*           if (summary.final_cost <= std::max(better_frac*icost,solved_fine)) { */
/*             if (verbose) cout << " success " << summary.iterations.size() - 1 */
/*               << " iterations " << summary.final_cost << endl; */
/*             successes.insert(xi); */
/*             l2_reg_refine(p,x,opts); */
/*             logsol(x,"out_partial_sparse.txt"); */
/*             goto found; */
/*           } */
/*           if (verbose) cout << " fail " << summary.iterations.size() - 1 << " iterations " */
/*             << summary.final_cost << endl; */
/*           fails[xi]++; */
/*           p.RemoveResidualBlock(rid); */
/*           copy(sav.begin(),sav.end(),x); */
/*           if (tries >= trylimit) break; */
/*         } */
/*       } */
/*     } */
/*     break; */
/* found:; */
/*       if (tries >= trylimit) break; */
/*   } */
/* } */
