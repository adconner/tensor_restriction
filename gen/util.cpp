#include <algorithm>
#include <iterator>
#include <fstream>
#include <memory>
#include <limits>
#include <tuple>
#include "gen/prob.h"
#include "util.h"

// control variables
mt19937 rng;

void logsol(const MyProblem &p, string fname) {
  if (tostdout) {
    cout.precision(numeric_limits<double>::max_digits10);
    cout << "OUTPUT " << fname << " ";
    copy(p.x.begin(),p.x.end(),ostream_iterator<double>(cout," "));
    cout << endl;
  } else {
    ofstream out(fname);
    out.precision(numeric_limits<double>::max_digits10);
    copy(p.x.begin(),p.x.end(),ostream_iterator<double>(out,"\n"));
  }
}

pair<int,int> get_coordinates(int i) {
  assert(i < N);
  int k = *(find_if(BBOUND+1,BBOUND+BLOCKS+1,[&](int j){return j > i;})-1);
  return make_pair(k,i-k);
}

void set_value_constant_or_variable(MyProblem &p, int i, bool variable) {
  if (p.variable_mask[i] == variable) 
    return;
  p.variable_mask[i] = variable;
  int b,k; tie(b,k) = get_coordinates(i);
  int bsize = BBOUND[b+1] - BBOUND[b];
  if (bsize == 1) {
    assert(k==0);
    if (variable) {
      p.p.SetParameterBlockVariable(p.x.data()+b*MULT);
    } else {
      p.p.SetParameterBlockConstant(p.x.data()+b*MULT);
    }
  } else {
    vector<int> v;
    for (int j=0; j<bsize; ++j) {
      if (!p.variable_mask[BBOUND[b]+j]) {
        for (int s=0; s<MULT; ++s) {
          v.push_back(j*MULT + s);
        }
      }
    }
    if (v.empty()) {
      p.p.SetParameterization(p.x.data()+b*MULT,0);
    } else {
      p.p.SetParameterization(p.x.data()+b*MULT,
          new SubsetParameterization(bsize*MULT,v));
    }
  }
}

void set_value_variable(MyProblem &p, int i) {
  set_value_constant_or_variable(p,i,true);
}

void set_value_constant(MyProblem &p, int i) {
  set_value_constant_or_variable(p,i,false);
}
