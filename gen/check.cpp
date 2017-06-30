#include <random>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <sstream>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "prob.h"

using namespace ceres;
using namespace std;
double cost_thresh = 1e-23;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  cout.precision(numeric_limits<double>::max_digits10);

  double x[N];
  Problem problem;
  ceres::ParameterBlockOrdering pbo;
  AddToProblem(problem,pbo,x);
  Problem::EvaluateOptions eopts;
  problem.GetResidualBlocks(&eopts.residual_blocks);

  string line;
  while (getline(cin,line)) {
    stringstream in(line);
    for (int i=0; i<N; ++i)
      in >> x[i];
    double cost; problem.Evaluate(eopts,&cost,0,0,0);
    if (cost < cost_thresh)
      cout << line << endl;
  }

  return 0;
}
