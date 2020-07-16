#ifndef _OPTS_H_
#define _OPTS_H_

#include "prob.h"

const int verbose = 1;
const bool tostdout = false;
const bool attemptsparse = true; 
const bool l2_reg_always = false;
const bool l2_reg_random_start = true;
const bool record_iterations = false;
const bool log_rough = !tostdout;

const double ftol = 1e-13;
const double gtol = 1e-30;
const double ptol = 1e-30;

const int l2_reg_steps = 8;
const double l2_reg_decay = 0.316227766016838; // sqrt(0.1)
const double alphastart = 0.001;
const double ftol_rough = 1e-3;
const int iterations_rough = 500;
const double abort_worse = 1e-4;

const int iterations_fine = 1000;

const double solved_fine = 1e-25;
const double attempt_sparse_thresh = 1e-12;

// parameters for descretization
const int l2_reg_steps_discrete = 1;
const double l2_reg_decay_discrete = 0.316227766016838; // sqrt(0.1)
const double alphastart_discrete = 0.0005;

const int iterations_discrete = 60;
const double ftol_discrete = 0.1; // ~ solved_fine ^ (1/iterations_discrete)
const double max_elem = 1e8;
const double better_frac = 1.0; // will accept if still solved or no worse than this times previous

/*
 * some settings for sparsifying border rank decomps 
 *  const double solved_fine = 1e-5;
 *  const double attempt_sparse_thresh = 1e-5;
 *  const int iterations_discrete = 120;
 *  const double ftol_discrete = 0.0004; // ~ solved_fine ^ (1/iterations_discrete)
 */

#endif
