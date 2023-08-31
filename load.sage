from itertools import chain

def code_export_restrict(T,S,sym=False,cx=False,tight=None,path='.'):
    if tight is None:
        tight = [0]*len(T) if sym else [[0]*len(T),[0]*T[0].nrows(),[0]*T[0].ncols()]
    def estr(e):
        if cx:
            e = CDF(e)
            if e.imag() == 0:
                return '%.15g'%e.real()
            else:
                return '{%.15g,%.15g}'%(e.real(),e.imag())
        else:
            e = RDF(e)
            return '%.15g' % e
    tdim = (len(T),) + T[0].dimensions()
    ta,tb,tc = tdim
    sdim = (len(S),) + S[0].dimensions()
    sa,sb,sc = sdim
    assert not sym or (ta == tb and tb == tc and sa == sb and sb == sc)
    Se = [(i,j,k,e) for i,m in enumerate(S) for (j,k),e in sorted(m.dict().items())]
    with open("%s/prob.h" % path,"w") as prob:
        prob.write('#include "restrict/prob.h"\n')
    with open("%s/restrict/prob.h" % path,"w") as prob:
        prob.write('''#ifndef _RESTRICT_PROB_H_
#define _RESTRICT_PROB_H_
#include "ceres/ceres.h"
const int N = %d;
const int M = %d;
const int MULT = %d;
const int BLOCKS = %d;
const int BBOUND[] = {%s};
ceres::ResidualBlockId AddToProblem(ceres::Problem &p, double *x, int eqi);
void SetParameterBlockOrdering(ceres::ParameterBlockOrdering &pbo, double *x);
class Eq : public ceres::CostFunction {
    public: double alpha = 1.0;
};
%s
#endif''' % (ta*sa if sym else sum(a*b for a,b in zip(tdim,sdim)),prod(tdim),2 if cx else
        1,ta if sym else sum(tdim), ','.join(map(str,range(0,ta*sa+sa,sa))) if
        sym else ','.join(map(str,chain(range(0,ta*sa,sa),range(ta*sa,ta*sa+tb*sb,sb),\
                range(ta*sa+tb*sb,ta*sa+tb*sb+tc*sc +sc,sc)))),
        '#define SYM' if sym else ''))
    with open("%s/restrict/tensor.h" % path,"w") as prob:
        prob.write('''#ifndef _RESTRICT_TENSOR_H_
#define _RESTRICT_TENSOR_H_
%s
using F = %s;

const int TA = %d, TB = %d, TC = %d;
const int SA = %d, SB = %d, SC = %d;
const int SNZ = %d;

const F T[][TB][TC] = {%s};
const int SI[] = {%s};
const int SJ[] = {%s};
const int SK[] = {%s};
const F SV[] = {%s};

const int TIGHTA[] = {%s};
const int TIGHTB[] = {%s};
const int TIGHTC[] = {%s};
#endif
''' % ('#define CX\n#include <complex>\n' if cx else '',
    'std::complex<double>' if cx else 'double',
    ta,tb,tc,sa,sb,sc,len(Se),
    ','.join(['{'+','.join(['{'+','.join([estr(e) for e in r])+'}' 
        for r in m])+'}' for m in T]),
    ','.join([str(i) for i,j,k,e in Se]),
    ','.join([str(j) for i,j,k,e in Se]),
    ','.join([str(k) for i,j,k,e in Se]),
    ','.join([estr(e) for i,j,k,e in Se]),
    ','.join(map(str,tight if sym else tight[0])),
    ','.join(map(str,tight if sym else tight[1])),
    ','.join(map(str,tight if sym else tight[2])),
    ))
