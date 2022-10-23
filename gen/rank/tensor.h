#ifndef _RANK_TENSOR_H_
#define _RANK_TENSOR_H_

#include <vector>
using namespace std;

extern bool SYM;

#ifdef CX
#include <complex>
using F = std::complex<double>;
#else
using F = double;
#endif

extern int TA, TB, TC;
extern int SA, SB, SC;
extern int SNZ;

extern vector<vector<vector<F> > > T;

extern vector<int> SI, SJ, SK;
extern vector<F> SV;

extern vector<int> TIGHTA, TIGHTB, TIGHTC;

void set_rank_r(int r);
void set_params();
#endif
