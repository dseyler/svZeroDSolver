// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "PiecewiseValve.h"

// Block.h only forward-declares Model, but prepare_step reads parameter
// values off it
#include "Model.h"

void PiecewiseValve::setup_dofs(DOFHandler& dofhandler) {
  // set_up_dofs args: dofhandler (passed in), num equations, list of internal
  // variable names (strings) 2 eqns, one for Pressure, one for Flow
  Block::setup_dofs_(dofhandler, 2, {});
}

// update_constant updates matrices E and F from E(y,t)*y_dot + F(y,t)*y +
// c(y,t) = 0 with terms that DO NOT DEPEND ON THE SOLUTION
void PiecewiseValve::update_constant(SparseSystem& system,
                                     std::vector<double>& parameters) {
  // Set element contributions
  // coeffRef args are the indices (i,j) of the matrix
  // global_eqn_ids: number of rows in the matrix, set in setup_dofs
  // global_var_ids: number of columns, organized as pressure and flow of all
  // inlets and then all outlets of the block
  double Rmin = parameters[global_param_ids[ParamId::RMIN]];
  double Rmax = parameters[global_param_ids[ParamId::RMAX]];

  system.F.coeffRef(global_eqn_ids[0], global_var_ids[0]) = 1.0;
  system.F.coeffRef(global_eqn_ids[0], global_var_ids[2]) = -1.0;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[1]) = 1.0;
  system.F.coeffRef(global_eqn_ids[1], global_var_ids[3]) = -1.0;
}

// update_solution updates matrices E and F from E(y,t)*y_dot + F(y,t)*y +
// c(y,t) = 0 with terms that DO DEPEND ON THE SOLUTION (will change with each
// time step)
void PiecewiseValve::update_solution(
    SparseSystem& system, std::vector<double>& parameters,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) {
  double resistance = 0;

  if (freeze_state) {
    // Valve state was decided once for this driver step in prepare_step
    resistance = R_cached;
  } else {
    // Get states
    double p_in = y[global_var_ids[0]];
    double p_out = y[global_var_ids[2]];

    // Get parameters
    double Rmin = parameters[global_param_ids[ParamId::RMIN]];
    double Rmax = parameters[global_param_ids[ParamId::RMAX]];

    if (p_out < p_in) {
      resistance = Rmin;
    } else {
      resistance = Rmax;
    }
  }

  system.F.coeffRef(global_eqn_ids[0], global_var_ids[1]) = -resistance;
}

// prepare_step decides the valve state once per driver step, from the state
// converged at the end of the previous one, so that update_solution can hold
// the resistance fixed across every Newton iteration of the step
void PiecewiseValve::prepare_step(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y_old,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy_old) {
  if (!freeze_state) {
    return;
  }

  // Get previous converged states
  double p_in_old = y_old[global_var_ids[0]];
  double p_out_old = y_old[global_var_ids[2]];

  // Get parameters. prepare_step is not handed the parameter vector, but
  // get_parameter_value reads the same values update_solution receives.
  double Rmin = model->get_parameter_value(global_param_ids[ParamId::RMIN]);
  double Rmax = model->get_parameter_value(global_param_ids[ParamId::RMAX]);

  if (p_out_old < p_in_old) {
    R_cached = Rmin;
  } else {
    R_cached = Rmax;
  }
}

void PiecewiseValve::set_freeze_state(bool freeze) { freeze_state = freeze; }
