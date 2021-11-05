#include "prob.h"
#include "tensor.h"

#include <complex>
typedef std::complex<double> cx;

using namespace std;

struct Equation : public Eq {
  Equation(int _i, int _j, int _k) : i(_i), j(_j), k(_k) {
#ifdef SYM
    if (i == j && j == k) {
      *mutable_parameter_block_sizes() = {MULT*SA};
    } else if (i == j || j == k || i == k) {
      *mutable_parameter_block_sizes() = {MULT*SA,MULT*SA};
    } else {
      *mutable_parameter_block_sizes() = {MULT*SA,MULT*SA,MULT*SA};
    }
#else
    *mutable_parameter_block_sizes() = {MULT*SA,MULT*SB,MULT*SC};
#endif
    set_num_residuals(MULT);
  }
  bool Evaluate(double const* const* x, double* residuals, double** jacobians) const {
#ifdef SYM
    double const* _x[3];
    double * _jacobians[3];
    if (i == j && j == k) {
      _x[0] = x[0]; _x[1] = x[0]; _x[2] = x[0];
      x = _x;
      if (jacobians) {
        _jacobians[0] = jacobians[0];
        _jacobians[1] = jacobians[0];
        _jacobians[2] = jacobians[0];
        jacobians = _jacobians;
      }
    } else if (i == j) {
      _x[0] = x[0]; _x[1] = x[0]; _x[2] = x[1];
      x = _x;
      if (jacobians) {
        _jacobians[0] = jacobians[0];
        _jacobians[1] = jacobians[0];
        _jacobians[2] = jacobians[1];
        jacobians = _jacobians;
      }
    } else if (i == k) {
      _x[0] = x[0]; _x[1] = x[1]; _x[2] = x[0];
      x = _x;
      if (jacobians) {
        _jacobians[0] = jacobians[0];
        _jacobians[1] = jacobians[1];
        _jacobians[2] = jacobians[0];
        jacobians = _jacobians;
      }
    } else if (j == k) {
      _x[0] = x[0]; _x[1] = x[1]; _x[2] = x[1];
      x = _x;
      if (jacobians) {
        _jacobians[0] = jacobians[0];
        _jacobians[1] = jacobians[1];
        _jacobians[2] = jacobians[1];
        jacobians = _jacobians;
      }
    } 
#endif

    if (jacobians) {
      if (jacobians[0])
        fill(jacobians[0],jacobians[0]+MULT*SA,0.0);
      if (jacobians[1])
        fill(jacobians[1],jacobians[1]+MULT*SB,0.0);
      if (jacobians[2])
        fill(jacobians[2],jacobians[2]+MULT*SC,0.0);
    }
#ifndef CX
    residuals[0] = -T[i][j][k];
    for (int t=0; t<SNZ; ++t) {
      residuals[0] += alpha*x[0][SI[t]]*x[1][SJ[t]]*x[2][SK[t]]*SV[t];
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][SI[t]] += alpha*x[1][SJ[t]]*x[2][SK[t]]*SV[t];
        }
        if (jacobians[1]) {
          jacobians[1][SJ[t]] += alpha*x[0][SI[t]]*x[2][SK[t]]*SV[t];
        }
        if (jacobians[2]) {
          jacobians[2][SK[t]] += alpha*x[0][SI[t]]*x[1][SJ[t]]*SV[t];
        }
      }
    }
    return true;
#else
    cx(residuals[0],residuals[1]) = -T[i][j][k];
    for (int t=0; t<SNZ; ++t) {
      cx a = cx(x[0][2*SI[t]],x[0][2*SI[t]+1]);
      cx b = cx(x[1][2*SJ[t]],x[1][2*SJ[t]+1]);
      cx c = cx(x[2][2*SK[t]],x[2][2*SK[t]+1]);
      cx(residuals[0],residuals[1]) += alpha*a*b*c*SV[t];
      if (jacobians) {
        if (jacobians[0]) {
          cx e = alpha*b*c*SV[t];
          cx(jacobians[0][2*SI[t]],jacobians[0][2*SA+2*t]) += e;
          cx(jacobians[0][2*SI[t]+1],jacobians[0][2*SA+2*t+1]) += cx(0,1)*e;
        }
        if (jacobians[1]) {
          cx e = alpha*a*c*SV[t];
          cx(jacobians[1][2*SJ[t]],jacobians[1][2*SB+2*t]) += e;
          cx(jacobians[1][2*SJ[t]+1],jacobians[1][2*SB+2*t+1]) += cx(0,1)*e;
        }
        if (jacobians[2]) {
          cx e = alpha*a*b*SV[t];
          cx(jacobians[2][2*SK[t]],jacobians[2][2*SC+2*t]) += e;
          cx(jacobians[2][2*SK[t]+1],jacobians[2][2*SC+2*t+1]) += cx(0,1)*e;
        }
      }
    }
    return true;
#endif
  }
  int i,j,k;
};

#ifndef SYM
ceres::ResidualBlockId AddToProblem(ceres::Problem &p, double *x, int eqi) {
  int k = eqi % TC; eqi /= TC;
  int j = eqi % TB; eqi /= TB;
  int i = eqi;
  return p.AddResidualBlock(new Equation(i,j,k),0,
      {x+MULT*BBOUND[i],x+MULT*BBOUND[TA+j],x+MULT*BBOUND[TA+TB+k]});
}
#else
ceres::ResidualBlockId AddToProblem(ceres::Problem &p, double *x, int eqi) {
  assert(TA == TB && TB == TC);
  assert(SA == SB && SB == SC);
  int k = eqi % TC; eqi /= TC;
  int j = eqi % TB; eqi /= TB;
  int i = eqi;
  if (i == j && j == k) {
    return p.AddResidualBlock(new Equation(i,j,k),0,
        x+MULT*BBOUND[i]);
  } else if (i == j) {
    return p.AddResidualBlock(new Equation(i,j,k),0,
        {x+MULT*BBOUND[i],x+MULT*BBOUND[k]});
  } else if (i == k) {
    return p.AddResidualBlock(new Equation(i,j,k),0,
        {x+MULT*BBOUND[i],x+MULT*BBOUND[j]});
  } else if (j == k) {
    return p.AddResidualBlock(new Equation(i,j,k),0,
        {x+MULT*BBOUND[i],x+MULT*BBOUND[j]});
  } else {
    return p.AddResidualBlock(new Equation(i,j,k),0,
        {x+MULT*BBOUND[i],x+MULT*BBOUND[j],x+MULT*BBOUND[k]});
  }
}
#endif

void SetParameterBlockOrdering(ceres::ParameterBlockOrdering &pbo, double *x) {
}
