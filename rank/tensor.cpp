#include <vector>
#include "tensor.h"
#include "prob.h"

using namespace std;

bool SYM;

int TA, TB, TC;
int SA, SB, SC;
int SNZ;

vector<vector<vector<F> > > T;

vector<int> SI, SJ, SK;
vector<F> SV;

vector<int> TIGHTA, TIGHTB, TIGHTC;

void set_rank_r(int r) {
  TA = T.size();
  TB = T[0].size();
  TC = T[0][0].size();
  SA = SB = SC = SNZ = r;
  SI.resize(r); for (int i=0; i<r; ++i) {SI[i] = i;}
  SJ.resize(r); for (int i=0; i<r; ++i) {SJ[i] = i;}
  SK.resize(r); for (int i=0; i<r; ++i) {SK[i] = i;}
  SV.resize(r); for (int i=0; i<r; ++i) {SV[i] = 1.0;}
  TIGHTA.assign(TA,0);
  TIGHTB.assign(TB,0);
  TIGHTC.assign(TC,0);
}

void set_params() {
  static vector<int> bbound;
  assert(T.size() == TA);
  assert(T[0].size() == TB);
  assert(T[0][0].size() == TC);
  if (SYM) {
    assert(TA == TB && TB == TC && SA == SB && SB == SC);
    N = TA*SA;
    BLOCKS = TA;
      
    bbound.assign(BLOCKS+1,0);
    for (int i=0; i<=BLOCKS; ++i) {
      bbound[i] = i*SA;
    }
    BBOUND = bbound.data();
    
  } else {
    N = TA*SA + TB*SB + TC*SC;
    BLOCKS = TA+TB+TC;
    
    bbound.assign(BLOCKS+1,0);
    for (int i=0; i < TA; ++i) {
      bbound[i] = i*SA;
    }
    for (int i=0; i < TB; ++i) {
      bbound[TA+i] = TA*SA + i*SB;
    }
    for (int i=0; i <= TC; ++i) {
      bbound[TA+TB+i] = TA*SA + TB*SB + i*SC;
    }
    BBOUND = bbound.data();
  }
  M = TA*TB*TC;
}
  
