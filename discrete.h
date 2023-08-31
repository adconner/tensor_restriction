#ifndef _DISCRETE_H_
#define _DISCRETE_H_
#include "util.h"
#include "problem.h"

enum DiscreteAttempt {
  DA_ZERO,
  DA_PM_ONE,
  DA_E3
};
void greedy_discrete(MyProblem &p, int &successes, 
    DiscreteAttempt da = DA_ZERO, int trylimit = 10);
void greedy_discrete_careful(MyProblem &p, int &successes, DiscreteAttempt da);
void greedy_discrete_pairs(MyProblem &p, const int trylimit = 10);
void greedy_discrete_lines(MyProblem &p, int ei, int trylimit);

#endif
