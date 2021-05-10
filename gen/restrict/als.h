#ifndef _ALS_H_
#define _ALS_H_

#include <vector>
#include <tuple>

void als(double *x, int group, double lambda = 0.0);
void als_some(double *x, const std::vector<std::tuple<int,int,int> > &eqs, int group, double lambda = 0.0);

#endif
