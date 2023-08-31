#ifndef _CHOLMOD_H_
#define _CHOLMOD_H_

void trust_region_f(MyProblem &p, function<void()> f, double relftol = 1e-3, int maxit = 200);
void trust_region(MyProblem &p, double relftol = 1e-3, int maxit = 200);

#endif
