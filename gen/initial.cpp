#include <random>
#include <algorithm>
#include <fstream>
#include "initial.h"
#include "opts.h"

using namespace std;
using namespace ceres;

void fill_initial(double *x, int argc, char **argv, Problem &problem) {
  if (argc == 1) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0,0.4);
    generate_n(x,MULT*N,[&] {return dist(gen);});
  } else {
    ifstream in(argv[1]);
    for (int i=0; i<MULT*N; ++i)
      in >> x[i];
  }
}
