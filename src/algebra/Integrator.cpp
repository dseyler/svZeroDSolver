// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "Integrator.h"

Integrator::Integrator(Model* model, double time_step_size, double rho,
                       double atol, int max_iter) {
  this->model = model;
  alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho);
  alpha_f = 1.0 / (1.0 + rho);
  gamma = 0.5 + alpha_m - alpha_f;
  ydot_init_coeff = 1.0 - 1.0 / gamma;

  y_coeff = gamma * time_step_size;
  y_coeff_jacobian = alpha_f * y_coeff;

  size = model->dofhandler.size();
  system = SparseSystem(size);
  this->time_step_size = time_step_size;
  this->atol = atol;
  this->max_iter = max_iter;

  y_af = Eigen::Matrix<double, Eigen::Dynamic, 1>(size);
  ydot_am = Eigen::Matrix<double, Eigen::Dynamic, 1>(size);

  // Make some memory reservations
  system.reserve(model);
}

// Must declare default constructord and dedtructor
// because of Eigen.
Integrator::Integrator() {}
Integrator::~Integrator() {}

void Integrator::clean() {
  // Cannot be in destructor because dynamically allocated pointers will be lost
  // when objects are assigned from temporary objects.
  system.clean();
}

void Integrator::update_params(double time_step_size) {
  this->time_step_size = time_step_size;
  y_coeff = gamma * time_step_size;
  y_coeff_jacobian = alpha_f * y_coeff;
  model->update_constant(system);
  model->update_time(system, 0.0);
}

// Forward-sensitivity propagation through one internal step. Called by
// step() AFTER Newton converges. For each tracked q_dof, advance
// (s_n, t_n) := (∂y_n/∂Q_new, ∂ẏ_n/∂Q_new) → (s_n+1, t_n+1) using the LU
// left in system.solver from the last Newton solve.
//
// Generalized-α relationships within one step (reading off step()):
//   ẏ_am = (1−α_m)·ẏ_n + α_m·ẏ_n+1
//   y_n+1 = y_n + γ·Δt·ẏ_n+1 + (1−γ)·Δt·ẏ_n          [the integrator update]
//   y_af  = (1−α_f)·y_n + α_f·y_n+1
//        = y_n + α_f·γ·Δt·ẏ_n+1 + α_f·(1−γ)·Δt·ẏ_n   [substituting y_n+1]
// Differentiating the converged residual
//   r = -C(t,Q) - (E+dC_dydot)·ẏ_am - (F+dC_dy)·y_af = 0
// w.r.t. Q (held fixed during the step) and grouping the unknown t_n+1:
//   J · t_n+1 = -∂C/∂Q
//              - (E+dC_dydot)·(1−α_m)·t_n
//              - (F+dC_dy)·[s_n + α_f·(1−γ)·Δt·t_n]
// where J = (E+dC_dydot)·α_m + (F+dC_dy)·α_f·γ·Δt  is the same Jacobian
// the Newton solve just factorized.
//
// Then accumulate the running sensitivities:
//   t_n+1 := solve(J, rhs)
//   s_n+1 := s_n + γ·Δt·t_n+1 + (1−γ)·Δt·t_n
//
// Key correction: the legacy single-step formula in get_dP_dQ assumed
// s_n = 0 and t_n = 0 (initial conditions for N=1). For N>1, the
// (1−α_m), α_f·(1−γ)·Δt and (1−γ)·Δt factors above carry the prior
// step's sensitivity through. Without them, the cumulative dP/dQ misses
// O(N) of the correct value (off by ~factor of N internal steps).
//
// The dQ_dQnew_factor = -∂C[q_dof]/∂Q_new = (t_mid - step_start)/external_dt
// captures FlowReferenceBC's linear time interpolation: at internal step
// k of N, this is (k + α_f)/N, NOT just α_f.
static void propagate_sensitivity(
    SparseSystem& system,
    const std::vector<int>& q_dofs,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& dy_dQ,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& dydot_dQ,
    double dQ_dQnew_factor, double alpha_f, double alpha_m, double gamma,
    double dt, int size) {
  if (q_dofs.empty()) return;

  Eigen::SparseMatrix<double> A_ydot = system.E + system.dC_dydot;
  Eigen::SparseMatrix<double> A_y    = system.F + system.dC_dy;

  const double one_minus_am = 1.0 - alpha_m;
  const double af_one_minus_g_dt = alpha_f * (1.0 - gamma) * dt;
  const double y_coeff_full = gamma * dt;
  const double dot_carry    = (1.0 - gamma) * dt;

  for (size_t k = 0; k < q_dofs.size(); ++k) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> t_old = dydot_dQ.col(k);
    Eigen::Matrix<double, Eigen::Dynamic, 1> s_old = dy_dQ.col(k);

    Eigen::Matrix<double, Eigen::Dynamic, 1> rhs =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(size);
    rhs[q_dofs[k]] = dQ_dQnew_factor;
    rhs.noalias() -= A_ydot * (one_minus_am * t_old);
    rhs.noalias() -= A_y    * (s_old + af_one_minus_g_dt * t_old);

    Eigen::Matrix<double, Eigen::Dynamic, 1> t_new = system.solver->solve(rhs);
    dydot_dQ.col(k) = t_new;
    dy_dQ.col(k)    = s_old + y_coeff_full * t_new + dot_carry * t_old;
  }
}

void Integrator::arm_sensitivity(const std::vector<int>& q_dofs,
                                 double external_dt,
                                 double step_start_time) {
  sensitivity_q_dofs_ = q_dofs;
  sensitivity_external_dt_ = external_dt;
  sensitivity_step_start_time_ = step_start_time;
  if (q_dofs.empty()) {
    dy_dQ_.resize(0, 0);
    dydot_dQ_.resize(0, 0);
    return;
  }
  dy_dQ_.setZero(size, static_cast<int>(q_dofs.size()));
  dydot_dQ_.setZero(size, static_cast<int>(q_dofs.size()));
}

State Integrator::step(const State& old_state, double time) {
  // Predictor: Constant y, consistent ydot
  State new_state = State::Zero(size);
  new_state.ydot += old_state.ydot * ydot_init_coeff;
  new_state.y += old_state.y;

  // Determine new time (evaluate terms at generalized mid-point)
  double new_time = time + alpha_f * time_step_size;

  // Evaluate time-dependent element contributions in system
  model->update_time(system, new_time);

  // NOTE: prepare_step is intentionally NOT called here.
  // The original implementation invoked it on every internal substep, which
  // re-evaluated PiecewiseValve resistances mid-external-step using
  // intra-step LPN state — making R depend on the trial Q supplied by the
  // 3D solver. That broke svMultiPhysics' Newton (line search saw a
  // non-smooth merit function across α whenever any internal substep crossed
  // a valve threshold). Now prepare_step is invoked once per EXTERNAL step
  // from interface.cpp (run_simulation and increment_time), using the
  // canonical state at the start of the external step. All internal
  // substeps within one external step now share the same R_cached, making
  // ‖R‖² smooth along the 3D Newton search direction.

  // Count total number of step calls
  n_iter++;

  // Non-linear Newton-Raphson iterations
  for (size_t i = 0; i < max_iter; i++) {
    // Initiator: Evaluate the iterates at the intermediate time levels
    ydot_am.setZero();
    y_af.setZero();
    ydot_am += old_state.ydot + (new_state.ydot - old_state.ydot) * alpha_m;
    y_af += old_state.y + (new_state.y - old_state.y) * alpha_f;

    // Update solution-dependent element contribitions
    model->update_solution(system, y_af, ydot_am);

    // Evaluate residual
    system.update_residual(y_af, ydot_am);

    // Check termination criterium
    if (system.residual.cwiseAbs().maxCoeff() < atol) {
      break;
    }

    // Abort if maximum number of non-linear iterations is reached
    else if (i == max_iter - 1) {
      throw std::runtime_error(
          "Maximum number of non-linear iterations reached at time " +
          std::to_string(time));
    }

    // Evaluate Jacobian
    system.update_jacobian(alpha_m, y_coeff_jacobian);

    // Solve system for increment in ydot
    system.solve();

    // Perform post-solve actions on blocks
    model->post_solve(new_state.y);

    // Update the solution
    new_state.ydot += system.dydot;
    new_state.y += system.dydot * y_coeff;

    // Count total number of nonlinear iterations
    n_nonlin_iter++;
  }

  // After Newton convergence: propagate forward sensitivities ∂y/∂Q,
  // ∂ẏ/∂Q for every armed q_dof through this step. No-op if not armed.
  // Uses the LU factorization left in system.solver from the last solve()
  // — one extra back-solve per tracked q_dof.
  //
  // Per-step factor ∂C[q_dof]/∂Q_new from the FlowReferenceBC's linear
  // time interpolation (Q_old at step_start_time → Q_new at
  // step_start_time + external_dt):
  //   t_mid = time + α_f · time_step_size
  //   factor = (t_mid - step_start_time) / external_dt
  // This is (k + α_f) / N for the k-th of N internal steps. The legacy
  // single-step formula assumed N=1 (factor = α_f); for N>1 that loses
  // a factor of ~N in the cumulative dP/dQ.
  if (!sensitivity_q_dofs_.empty()) {
    // Defensive: if Newton converged in 0 iters (initial residual already
    // below atol), system.solver was never factorize()'d during this step
    // and the LU is stale (or never built). Refresh it here against the
    // current y_af/ydot_am Jacobian so the back-solve in
    // propagate_sensitivity has a valid factorization. Mirrors the same
    // guard in get_dP_dQ.
    if (!system.has_factorized) {
      system.update_jacobian(alpha_m, y_coeff_jacobian);
      system.solver->factorize(system.jacobian);
      if (system.solver->info() != Eigen::Success) {
        throw std::runtime_error(
            "[Integrator::step] sensitivity factorize failed (singular?)");
      }
      system.has_factorized = true;
    }

    const double t_mid = time + alpha_f * time_step_size;
    const double factor = (sensitivity_external_dt_ > 0.0)
        ? (t_mid - sensitivity_step_start_time_) / sensitivity_external_dt_
        : alpha_f;  // fallback if external_dt unknown (N=1 special case)
    propagate_sensitivity(system, sensitivity_q_dofs_, dy_dQ_, dydot_dQ_,
                          factor, alpha_f, alpha_m, gamma, time_step_size,
                          size);
  }

  return new_state;
}

double Integrator::avg_nonlin_iter() {
  return (double)n_nonlin_iter / (double)n_iter;
}

double Integrator::get_dP_dQ(int p_dof, int q_dof) {
  // Preferred path: forward-sensitivity tracking. If arm_sensitivity()
  // was called (via interface->run_simulation()'s auto-arming) for this
  // q_dof prior to the most recent run_simulation, dy_dQ_ holds the
  // accumulated dy_N/dQ across ALL internal steps of that run. This is
  // exact for any number_of_time_pts ≥ 2 — including the pathological
  // multi-step regime where the legacy single-step formula below would
  // miss N-1 steps of sensitivity propagation and produce a wrong dP/dQ.
  for (size_t k = 0; k < sensitivity_q_dofs_.size(); ++k) {
    if (sensitivity_q_dofs_[k] == q_dof) {
      return dy_dQ_(p_dof, static_cast<int>(k));
    }
  }

  // Legacy fallback: single-step formula. Only correct when run_simulation
  // performs exactly ONE internal step (number_of_time_pts == 2). Used
  // when arm_sensitivity has not been called for this q_dof.
  //
  // The most recent step() left system.solver holding the LU
  // factorization of J = (E + dC_dydot)·α_m + (F + dC_dy)·α_f·γ·Δt.
  // Back-solve J·w = e_{q_dof}, then map dydot → dy_new via y_coeff.
  //
  // svMultiPhysics calls this from baf_ini before the first time step
  // has run. At that point the SparseLU has only had analyzePattern()
  // called (during SparseSystem::reserve), not factorize() — calling
  // solve() on it would segfault. Detect this via info() and refresh
  // the factorization from the current F/E (which the model populated
  // during reserve via update_constant + update_time at t=0). This
  // also makes get_dP_dQ defensive against any future call site that
  // queries before a step has run.
  if (!system.has_factorized) {
    system.update_jacobian(alpha_m, y_coeff_jacobian);
    system.solver->factorize(system.jacobian);
    if (system.solver->info() != Eigen::Success) {
      throw std::runtime_error(
          "get_dP_dQ: SparseLU factorize failed (singular Jacobian?)");
    }
    system.has_factorized = true;
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> rhs =
      Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(size);
  rhs[q_dof] = 1.0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> w = system.solver->solve(rhs);
  // Scale by y_coeff_jacobian = α_f·γ·Δt, NOT y_coeff = γ·Δt. Reason:
  // the FlowReferenceBC parameter Q is time-dependent (linear ramp from
  // Q_old at t_n to Q_new at t_n+1) and update_time evaluates it at the
  // generalized-α mid-step time t_n + α_f·Δt, giving
  //   C[r] = -((1-α_f)·Q_old + α_f·Q_new).
  // svMultiPhysics perturbs only Q_new (via perturb_flowrate, which
  // bumps Qn_), so ∂C/∂Q_new = -α_f·e_r and ∂R/∂Q_new = +α_f·e_r. The
  // extra α_f turns y_coeff into y_coeff_jacobian. Without this scaling
  // the analytical bc.r is exactly 1/α_f = 1.5× the FD value (verified
  // empirically on tests/cases/fluid/pipe_RCR_sv0D), causing a ~22 dB
  // Newton-overshoot at the second outer iter.
  return y_coeff_jacobian * w[p_dof];
}
