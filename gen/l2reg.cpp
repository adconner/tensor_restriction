#include "l2reg.h"

class L2Regularization : public SizedCostFunction<MULT,MULT> {
  public:
    L2Regularization(double &sqalpha_) : sqalpha(sqalpha_) {}
    bool Evaluate(const double* const* x,
                        double* residuals,
                        double** jacobians) const {
      residuals[0] = sqalpha * x[0][0];
      if (MULT == 2) residuals[1] = sqalpha * x[0][1];
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = sqalpha;
          if (MULT == 2) {
            jacobians[0][1] = 0;
            jacobians[0][2] = 0;
            jacobians[0][3] = sqalpha;
          }
        }
      }
      return true;
    }
  private:
    const double &sqalpha;
};

void l2_reg_search(Problem &problem, double *x, const Solver::Options &opts) {
  Solver::Options options(opts);
  options.max_num_iterations = iterations_rough;
  options.function_tolerance = ftol_rough;

  double sqalpha = std::sqrt(alphastart);
  vector<ResidualBlockId> rids;
  for (int i=0; i<N; ++i) {
    rids.push_back(problem.AddResidualBlock(new L2Regularization(sqalpha), NULL, &x[MULT*i]));
  }

  for (int i=l2_reg_steps; i>0; --i, sqalpha *= std::sqrt(l2_reg_decay)) {
    Solver::Summary summary;
    if (verbose) {
      cout << "l2 regularization coefficient " << (sqalpha * sqalpha) << endl;
      cout.flush();
    }
    Solve(options, &problem, &summary);
  }

  for (auto rid : rids) {
    problem.RemoveResidualBlock(rid);
  }

  if (verbose) cout << "rough solving..." << endl;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
}

void l2_reg_refine(Problem &problem, double *x, const Solver::Options & opts) {
  if (l2_reg_steps_refine == 0) return;
  Solver::Options options(opts);
  options.max_num_iterations = iterations_rough;
  options.function_tolerance = ftol_rough;

  double sqalpha = std::sqrt(alphastart);
  vector<ResidualBlockId> rids;
  for (int i=0; i<N; ++i) {
    rids.push_back(problem.AddResidualBlock(new L2Regularization(sqalpha), NULL, &x[MULT*i]));
  }

  double start_alpha = alphastart_refine;
  while (true) {
    vector<double> sav(x,x+N*MULT);
    sqalpha = std::sqrt(start_alpha); 
    for (int i=l2_reg_steps_refine; i>0; --i, sqalpha *= std::sqrt(l2_reg_decay_refine)) {
      if (verbose >= 2) {
        cout << "l2 regularization coefficient " << (sqalpha * sqalpha) << ".. "; cout.flush();
      }
      Solver::Summary summary;
      Solve(options, &problem, &summary);
      if (verbose >= 2) {
        cout << summary.initial_cost << " -> " << summary.final_cost << endl; cout.flush();
      }
    }

    sqalpha = 0.0; 
    if (verbose >= 2) {
      cout << "getting back to solution.. "; cout.flush();
    }
    options.function_tolerance = ftol_rough*0.1;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    if (verbose >= 2) {
      cout << summary.initial_cost << " -> " << summary.final_cost << endl; cout.flush();
    }

    print_lines = false;
    if (summary.final_cost > solved_fine) {
      copy(sav.begin(),sav.end(),x);
      if (verbose >= 2) cout << "l2 reg failed, retrying" << endl;
    } else{
      break;
    }
    start_alpha *= l2_reg_decay_refine;
  }

  for (auto rid : rids) {
    problem.RemoveResidualBlock(rid);
  }
}

