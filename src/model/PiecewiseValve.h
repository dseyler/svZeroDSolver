// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

/**
 * @file PiecewiseValve.h
 * @brief model::PiecewiseValve source file
 */
#ifndef SVZERODSOLVER_MODEL_PIECEWISE_VALVE_HPP_
#define SVZERODSOLVER_MODEL_PIECEWISE_VALVE_HPP_

#include <math.h>

#include "Block.h"
#include "SparseSystem.h"
#include "debug.h"

/**
 * @brief Valve (tanh) block.
 *
 * Models the pressure drop across a diode-like valve, which is implemented as a
 * non-linear piecewise resistor. See \cite Regazzoni2022
 * (equations 16 and 22).
 *
 * \f[
 * \begin{circuitikz} \draw
 * node[left] {$Q_{in}$} [-latex] (0,0) -- (0.8,0);
 * \draw (1,0) node[anchor=south]{$P_{in}$}
 * to [D, l=$R_v$, *-*] (3,0)
 * node[anchor=south]{$P_{out}$};
 * \end{circuitikz}
 * \f]
 *
 * ### Governing equations
 *
 * \f[
 * P_{in}-P_{out}-Q_{in}\left[R(P_{out},P_{in})\right]=0
 * \f]
 *
 * \f[
 * Q_{in}-Q_{out}=0
 * \f]
 *
 * ### Local contributions
 *
 * \f[
 * R_i(p_1, p_2) =
 * \begin{cases}
 * R_{\min}, & p_1 < p_2, \\[0.5em]
 * R_{\max}, & p_1 \ge p_2.
 * \end{cases}
 * \f]
 *
 * \f[
 * \mathbf{y}^{e}=\left[\begin{array}{llll}P_{in} & Q_{in} &
 * P_{out} & Q_{out}\end{array}\right]^{T} \f]
 *
 * \f[
 * \mathbf{E}^{e}=\left[\begin{array}{cccc}
 * 0 & 0 & 0 & 0 \\
 * 0 & 0 & 0 & 0
 * \end{array}\right]
 * \f]
 *
 * \f[
 * \mathbf{F}^{e}=\left[\begin{array}{cccc}
 * 1 & -(R(P_{in},P_{out})) & -1 & 0 \\
 * 0 &      1                 &  0 & -1
 * \end{array}\right]
 * \f]
 *
 * \f[
 * \mathbf{c}^{e}=\left[\begin{array}{c}
 * 0 \\
 * 0
 * \end{array}\right]
 * \f]
 *
 * ### Parameters
 *
 * Parameter sequence for constructing this block
 *
 * * `0` Rmax: Maximum (closed) valve resistance
 * * `1` Rmin: Minimum (open) valve resistance
 * * `2` upstream_block: Name of block connected upstream
 * * `3` downstream_block: Name of block connected downstream
 *
 * ### Freezing the valve state
 *
 * By default \f$R\f$ is re-evaluated from the current Newton iterate at every
 * nonlinear iteration. The valve state can therefore flip back and forth
 * within a single step, which converges poorly near a valve-switching event
 * and is especially problematic when this model is coupled to an external
 * (e.g. 3D) solver, where \f$R\f$ can end up chasing the trial solution the
 * external solver supplies partway through a step.
 *
 * Setting the optional boolean `freeze_state` to `true` decides the valve
 * state once per driver step instead, from the state converged at the end of
 * the previous driver step, and holds \f$R\f$ fixed for the whole step. A
 * driver step is one time step in a standalone simulation, or one external
 * step (spanning several internal time steps) when coupled to an external
 * solver. This introduces an O(\f$\Delta t\f$) splitting error in exchange
 * for a much better conditioned nonlinear solve. It defaults to `false`,
 * which reproduces the per-iteration behavior exactly.
 *
 * ### Usage in json configuration file
 *
 *     "valves": [
 *         {
 *             "type": "PiecewiseValve",
 *             "name": "valve",
 *             "params": {
 *                 "Rmax": 100000.0,
 *                 "Rmin": 100.0,
 *                 "freeze_state": false,
 *                 "upstream_block": "upstream_vessel",
 *                 "downstream_block": "downstream_vessel"
 *             }
 *         }
 *     ]
 *
 * ### Internal variables
 *
 * This block has no internal variables.
 *
 */
class PiecewiseValve : public Block {
 public:
  /**
   * @brief Local IDs of the parameters
   *
   */
  enum ParamId {
    RMAX = 0,
    RMIN = 1,
    STEEPNESS = 2,
  };

  /**
   * @brief Construct a new PiecewiseValve object
   *
   * @param id Global ID of the block
   * @param model The model to which the block belongs
   */
  PiecewiseValve(int id, Model* model)
      : Block(id, model, BlockType::piecewise_valve, BlockClass::valve,
              {{"Rmax", InputParameter()},
               {"Rmin", InputParameter()},
               // Optional and non-numeric, so generate_block accepts the key
               // but skips it: it adds no entry to global_param_ids and the
               // ParamId values above are unaffected. It is read directly
               // from the json in create_valves.
               {"freeze_state", InputParameter(true, false, false)},
               {"upstream_block", InputParameter(false, false, false)},
               {"downstream_block", InputParameter(false, false, false)}}) {}

  /**
   * @brief Set up the degrees of freedom (DOF) of the block
   *
   * Set global_var_ids and global_eqn_ids of the element based on the
   * number of equations and the number of internal variables of the
   * element.
   *
   * @param dofhandler Degree-of-freedom handler to register variables and
   * equations at
   */
  void setup_dofs(DOFHandler& dofhandler) override;

  /**
   * @brief Update the constant contributions of the element in a sparse
   system
   *
   * @param system System to update contributions at
   * @param parameters Parameters of the model
   */
  void update_constant(SparseSystem& system,
                       std::vector<double>& parameters) override;

  /**
   * @brief Update the solution-dependent contributions of the element in a
   * sparse system
   *
   * @param system System to update contributions at
   * @param parameters Parameters of the model
   * @param y Current solution
   * @param dy Current derivate of the solution
   */
  void update_solution(
      SparseSystem& system, std::vector<double>& parameters,
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) override;

  /**
   * @brief Decide the valve state for the upcoming driver step
   *
   * No-op unless freeze_state is set. Otherwise evaluates the valve state
   * against the previous converged pressures and caches the resulting
   * resistance, which update_solution then holds fixed for the whole step.
   *
   * @param y_old Converged solution at the end of the previous driver step
   * @param ydot_old Converged time-derivative at the end of the previous
   * driver step (unused)
   */
  void prepare_step(
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& y_old,
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& ydot_old) override;

  /**
   * @brief Hold the valve resistance constant over each driver step
   *
   * @param freeze Whether to freeze the valve state
   */
  void set_freeze_state(bool freeze) override;

  /**
   * @brief Number of triplets of element
   *
   * Number of triplets that the element contributes to the global system
   * (relevant for sparse memory reservation)
   */
  TripletsContributions num_triplets{5, 0, 3};

 private:
  /// Hold the valve resistance constant over each driver step
  bool freeze_state{false};

  /// Valve resistance decided in prepare_step, used when freeze_state is set
  double R_cached{0.0};
};

#endif  // SVZERODSOLVER_MODEL_PiecewiseValve_HPP_
