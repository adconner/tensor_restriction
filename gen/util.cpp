#include <algorithm>
#include <iterator>
#include <fstream>
#include <memory>
#include <limits>
#include <tuple>
#include "prob.h"
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
  int bi = find_if(BBOUND+1,BBOUND+BLOCKS+1,[&](int j){return j > i;})-BBOUND-1;
  return make_pair(bi,i-BBOUND[bi]);
}

void set_value_constant_or_variable(MyProblem &p, int i, bool variable) {
  if (p.variable_mask[i] == variable) 
    return;
  p.variable_mask[i] = variable;
  int bi,k; tie(bi,k) = get_coordinates(i);
  int bsize = BBOUND[bi+1] - BBOUND[bi];
  if (bsize == 1) {
    assert(k==0);
    if (variable) {
      p.p.SetParameterBlockVariable(p.x.data()+BBOUND[bi]*MULT);
    } else {
      p.p.SetParameterBlockConstant(p.x.data()+BBOUND[bi]*MULT);
    }
  } else {
    vector<int> v;
    for (int j=0; j<bsize; ++j) {
      if (!p.variable_mask[BBOUND[bi]+j]) {
        for (int s=0; s<MULT; ++s) {
          v.push_back(j*MULT + s);
        }
      }
    }
    /* cout << endl<< (variable ? "adding back " : "deleting ") << bi << " " << k << " "; */
    /* copy(v.begin(),v.end(),ostream_iterator<int>(cout," ")); cout << endl; */
    if (v.empty()) {
      p.p.SetParameterization(p.x.data()+BBOUND[bi]*MULT,0);
    } else {
      p.p.SetParameterization(p.x.data()+BBOUND[bi]*MULT,
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

double max_abs(const MyProblem &p) {
  if (MULT == 1) {
    return accumulate(p.x.begin(),p.x.end(),0.0,[](double a, double b) {return max(a,std::abs(b));} );
  } else {
    double ma = 0.0;
    for (int i=0; i<N; ++i) {
      ma = max(ma,abs(cx(p.x[2*i],p.x[2*i+1])));
    }
    return ma;
  }
}
