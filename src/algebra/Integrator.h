// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file Integrator.h
 * @brief Integrator source file
 */
#ifndef SVZERODSOLVER_ALGEBRA_INTEGRATOR_HPP_
#define SVZERODSOLVER_ALGEBRA_INTEGRATOR_HPP_

#include <Eigen/Dense>
#include <vector>

#include "Model.h"
#include "State.h"

/**
 * @brief Generalized-alpha integrator
 *
 * This class handles the time integration scheme for solving 0D blood
 * flow system using the generalized-\f$\alpha\f$ method \cite JANSEN2000305.
 *
 * Mathematical details are available on the <a
 * href="https://simvascular.github.io/documentation/rom_simulation.html#0d-solver-theory">SimVascular
 * documentation</a>.
 */

class Integrator {
 private:
  double alpha_m{0.0};
  double alpha_f{0.0};
  double gamma{0.0};
  double time_step_size{0.0};
  double ydot_init_coeff{0.0};
  double y_coeff{0.0};
  double y_coeff_jacobian{0.0};
  double atol{0.0};
  int max_iter{0};
  int size{0};
  int n_iter{0};
  int n_nonlin_iter{0};
  Eigen::Matrix<double, Eigen::Dynamic, 1> y_af;
  Eigen::Matrix<double, Eigen::Dynamic, 1> ydot_am;
  SparseSystem system;
  Model* model{nullptr};

  // Forward-sensitivity tracking for analytical dP/dQ at coupling boundaries.
  // arm_sensitivity(q_dofs, external_step_size) populates sensitivity_q_dofs_
  // and zeros dy_dQ_/dydot_dQ_; each step() then propagates dy/dQ and dydot/dQ
  // for every tracked q_dof using one extra LU back-solve against the same
  // factorization the Newton step just used. After a multi-step
  // run_simulation, dy_dQ_(p_dof, k) holds the accumulated dy_N/dQ_k —
  // which is what get_dP_dQ returns. Empty sensitivity_q_dofs_ disables
  // tracking (legacy single-step fallback in get_dP_dQ).
  //
  // sensitivity_external_dt_ is needed because each internal step k of the
  // run_simulation evaluates ∂Q_input(t_mid)/∂Q_new = t_mid/external_dt for
  // a 2-point linear time interpolation Q_old → Q_new across the external
  // step. With N internal steps, this factor is (k + α_f) / N for step k —
  // NOT just α_f (which is the N=1 special case). Using α_f for N>1
  // under-counts by a factor of ~N_internal in the cumulative dP/dQ.
  std::vector<int> sensitivity_q_dofs_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dy_dQ_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dydot_dQ_;
  double sensitivity_external_dt_{0.0};
  double sensitivity_step_start_time_{0.0};

 public:
  /**
   * @brief Construct a new Integrator object
   *
   * @param model The model to simulate
   * @param time_step_size Time step size for generalized-alpha step
   * @param rho Spectral radius for generalized-alpha step
   * @param atol Absolut tolerance for non-linear iteration termination
   * @param max_iter Maximum number of non-linear iterations
   */
  Integrator(Model* model, double time_step_size, double rho, double atol,
             int max_iter);

  /**
   * @brief Construct a new Integrator object
   *
   */
  Integrator();

  /**
   * @brief Destroy the Integrator object
   *
   */
  ~Integrator();

  /**
   * @brief Delete dynamically allocated memory (in class member
   * SparseSystem<double> system).
   */
  void clean();

  /**
   * @brief Update integrator parameter and system matrices with model parameter
   * updates.
   *
   * @param time_step_size Time step size for 0D model
   */
  void update_params(double time_step_size);

  /**
   * @brief Perform a time step
   *
   * @param state Current state
   * @param time Current time
   * @return New state
   */
  State step(const State& state, double time);

  /**
   * @brief Get average number of nonlinear iterations in all step calls
   *
   * @return Average number of nonlinear iterations in all step calls
   *
   */
  double avg_nonlin_iter();

  /**
   * @brief Compute analytical dP/dQ at a coupled boundary.
   *
   * If sensitivity tracking has been armed for q_dof prior to the most
   * recent run_simulation (via arm_sensitivity), returns the accumulated
   * dy_N/dQ at p_dof — i.e., the full sensitivity through all internal
   * steps. This is the correct dP/dQ for any number_of_time_pts ≥ 2.
   *
   * If sensitivity tracking is NOT armed (sensitivity_q_dofs_ empty, or
   * q_dof not in the tracked list), falls back to the legacy single-step
   * formula: solve J·w = e_{q_dof} against the LU left by the LAST internal
   * step and return y_coeff_jacobian·w[p_dof]. This is exact only for
   * number_of_time_pts == 2 (one internal step per external step).
   *
   * @param p_dof Global index of the pressure DOF to read out
   * @param q_dof Global index of the flow DOF to perturb
   * @return dP/dQ at the converged new-step state
   */
  double get_dP_dQ(int p_dof, int q_dof);

  /**
   * @brief Arm forward-sensitivity tracking for one or more flow DOFs.
   *
   * Resets dy_dQ/dydot_dQ to zero (initial conditions ∂y_0/∂Q = 0,
   * ∂ẏ_0/∂Q = 0 at the start of an external step). Call BEFORE
   * run_simulation begins; each subsequent step() propagates the
   * sensitivities through one internal step using one extra LU back-solve
   * per tracked q_dof against the Newton solve's factorization.
   *
   * Auto-called from run_simulation() (interface.cpp) for every
   * FlowReferenceBC in the model, so external solvers don't need to call
   * it directly — they just call get_dP_dQ() after run_simulation
   * completes. Pass empty vector to disable tracking.
   *
   * @param q_dofs List of flow DOFs to track sensitivity for
   * @param external_dt The external step duration (sum of all internal
   *   time_step_size's). Needed for the per-internal-step ∂Q/∂Q_new
   *   factor t_mid/external_dt under 2-point linear time interpolation.
   * @param step_start_time The external solver's time at the start of
   *   the upcoming run_simulation call (i.e., the value passed as
   *   external_time). t_mid for internal step k is computed as
   *   (time_at_step_k + α_f·time_step_size) − step_start_time.
   */
  void arm_sensitivity(const std::vector<int>& q_dofs, double external_dt,
                       double step_start_time);
};

#endif  // SVZERODSOLVER_ALGEBRA_INTEGRATOR_HPP_
