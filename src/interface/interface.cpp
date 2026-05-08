// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "interface.h"

#include <cmath>

#include "BlockType.h"
#include "SimulationParameters.h"

// Static member data.
int SolverInterface::problem_id_count_ = 0;
std::map<int, SolverInterface*> SolverInterface::interface_list_;

//-----------------
// SolverInterface
//-----------------
SolverInterface::SolverInterface(const std::string& input_file_name)
    : input_file_name_(input_file_name) {
  problem_id_ = problem_id_count_++;
  SolverInterface::interface_list_[problem_id_] = this;
}

SolverInterface::~SolverInterface() {}

//////////////////////////////////////////////////////////
//            Callable interface functions              //
//////////////////////////////////////////////////////////

extern "C" void initialize(std::string input_file, int& problem_id,
                           int& pts_per_cycle, int& num_cycles,
                           int& num_output_steps,
                           std::vector<std::string>& block_names,
                           std::vector<std::string>& variable_names);

extern "C" void set_external_step_size(int problem_id,
                                       double external_step_size);

extern "C" void increment_time(int problem_id, const double external_time,
                               std::vector<double>& solution);

extern "C" void run_simulation(int problem_id, const double external_time,
                               std::vector<double>& output_times,
                               std::vector<double>& output_solutions,
                               int& error_code);

extern "C" void update_block_params(int problem_id, std::string block_name,
                                    std::vector<double>& params);

extern "C" void read_block_params(int problem_id, std::string block_name,
                                  std::vector<double>& params);

extern "C" void get_block_node_IDs(int problem_id, std::string block_name,
                                   std::vector<int>& IDs);

extern "C" void update_state(int problem_id, std::vector<double> new_state_y,
                             std::vector<double> new_state_ydot);

extern "C" void return_y(int problem_id, std::vector<double>& ydot);

extern "C" void get_coupling_jacobian(int problem_id, std::string block_name,
                                      double& dP_dQ);

extern "C" void return_ydot(int problem_id, std::vector<double>& ydot);

/**
 * @brief Initialize the 0D solver interface.
 *
 * @param input_file_arg The name of the JSON 0D solver configuration file.
 * @param problem_id The returned ID used to identify the 0D problem.
 * @param pts_per_cycle Number of time steps per cycle in the 0D model.
 * @param num_cycles Number of cardiac cycles in the 0D model.
 * @param num_output_steps Number of steps at which outputs are recorded.
 * @param block_names Vector of all the 0D block names.
 * @param variable_names Vector of all the 0D variable names.
 */
void initialize(std::string input_file_arg, int& problem_id, int& pts_per_cycle,
                int& num_cycles, int& num_output_steps,
                std::vector<std::string>& block_names,
                std::vector<std::string>& variable_names) {
  DEBUG_MSG("========== svZeroD initialize ==========");
  std::string input_file(input_file_arg);
  DEBUG_MSG("[initialize] input_file: " << input_file);
  std::string output_file = "svzerod.csv";

  auto interface = new SolverInterface(input_file);
  problem_id = interface->problem_id_;
  DEBUG_MSG("[initialize] problem_id: " << problem_id);

  // Create configuration reader.
  std::ifstream ifs(input_file);
  const auto& config = nlohmann::json::parse(ifs);
  auto simparams = load_simulation_params(config);

  auto model = std::shared_ptr<Model>(new Model());

  load_simulation_model(config, *model.get());

  // Propagate the operator-split valve flag from sim params to model.
  model->freeze_piecewise_valve_state =
      simparams.sim_freeze_piecewise_valve_state;

  auto state = load_initial_condition(config, *model.get());

  // Check that steady initial is not set when ClosedLoopHeartAndPulmonary is
  // used
  if ((simparams.sim_steady_initial == true) && (model->has_block("CLH"))) {
    std::runtime_error(
        "ERROR: Steady initial condition is not compatible with "
        "ClosedLoopHeartAndPulmonary block.");
  }

  // Set default cardiac cycle period if not set by model
  if (model->cardiac_cycle_period < 0.0) {
    model->cardiac_cycle_period =
        1.0;  // If it has not been read from config or Parameter
              // yet, set as default value of 1.0
  }

  // Calculate time step size
  if (!simparams.sim_coupled) {
    simparams.sim_time_step_size = model->cardiac_cycle_period /
                                   (double(simparams.sim_pts_per_cycle) - 1.0);
  } else {
    simparams.sim_time_step_size = simparams.sim_external_step_size /
                                   (double(simparams.sim_num_time_steps) - 1.0);
  }

  // Create a model.
  interface->model_ = model;

  // Create a vector containing all block names
  for (size_t i = 0; i < model->get_num_blocks(); i++) {
    block_names.push_back(model->get_block(i)->get_name());
  }
  variable_names = model->dofhandler.variables;

  // Get simulation parameters
  interface->time_step_size_ = simparams.sim_time_step_size;
  interface->rho_infty_ = simparams.sim_rho_infty;
  interface->max_nliter_ = simparams.sim_nliter;
  interface->absolute_tolerance_ = simparams.sim_abs_tol;
  interface->time_step_ = 0;
  interface->system_size_ = model->dofhandler.size();
  interface->num_time_steps_ = simparams.sim_num_time_steps;
  interface->pts_per_cycle_ = simparams.sim_pts_per_cycle;
  pts_per_cycle = simparams.sim_pts_per_cycle;
  num_cycles = simparams.sim_num_cycles;
  interface->external_step_size_ = simparams.sim_external_step_size;

  // For how many time steps are outputs being returned?
  // NOTE: Only tested num_output_steps = interface->num_time_steps_
  if (simparams.output_mean_only) {
    num_output_steps = 1;
    throw std::runtime_error(
        "ERROR: Option output_mean_only has not been implemented when "
        "using the svZeroDSolver interface library.");
  } else if (!simparams.output_all_cycles) {
    num_output_steps = interface->pts_per_cycle_;
    throw std::runtime_error(
        "ERROR: Option output_last_cycle_only has been implemented but not "
        "tested when using the svZeroDSolver interface library. Please test "
        "this "
        "functionality before removing this message.");
  } else {
    num_output_steps = interface->num_time_steps_;
  }
  interface->num_output_steps_ = num_output_steps;
  DEBUG_MSG("[initialize] System size: " << interface->system_size_);

  // Create steady initial state.
  if (simparams.sim_steady_initial) {
    DEBUG_MSG("[initialize] ----- Calculating steady initial condition ----- ");
    double time_step_size_steady = model->cardiac_cycle_period / 10.0;
    DEBUG_MSG("[initialize] Create steady model ... ");

    auto model_steady = model;
    model_steady->to_steady();
    Integrator integrator_steady(
        model_steady.get(), time_step_size_steady, interface->rho_infty_,
        interface->absolute_tolerance_, interface->max_nliter_);

    for (size_t i = 0; i < 31; i++) {
      // Per-pseudo-step prepare (steady spinup): refresh per-block caches.
      // This preserves the legacy per-step semantics for the spinup loop;
      // the production-time semantics (one prepare per external timestep)
      // applies to run_simulation/increment_time only.
      model_steady->prepare_step(state.y, state.ydot);
      state = integrator_steady.step(state, time_step_size_steady * double(i));
    }
    model_steady->to_unsteady();
  }

  interface->state_ = state;

  // Initialize states and times vectors because size is now known
  interface->times_.resize(num_output_steps);
  interface->states_.resize(num_output_steps);

  // Initialize integrator
  interface->integrator_ =
      Integrator(model.get(), interface->time_step_size_, interface->rho_infty_,
                 interface->absolute_tolerance_, interface->max_nliter_);

  DEBUG_MSG("[initialize] Done");
}

/**
 * @brief Set the timestep of the external program. For cases when 0D time step
 * depends on external time step.
 *
 * @param problem_id The returned ID used to identify the 0D problem.
 * @param external_step_size The time step size of the external program.
 */
void set_external_step_size(int problem_id, double external_step_size) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;

  // Update external step size in model and interface
  interface->external_step_size_ = external_step_size;

  // Update time step size in interface
  double zerod_step_size =
      external_step_size / (double(interface->num_time_steps_) - 1.0);
  interface->time_step_size_ = zerod_step_size;
}

/**
 * @brief Update the parameters of a particular block.
 *
 * @param problem_id The returned ID used to identify the 0D problem.
 * @param block name The name of the block to update.
 * @param params New parameters for the block (structure depends on block type).
 */
void update_block_params(int problem_id, std::string block_name,
                         std::vector<double>& params) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;

  // Find the required block
  auto block = model->get_block(block_name);
  auto block_type = model->get_block_type(block_name);
  // Update is handled differently for blocks that have time-varying parameters
  // (PRESSUREBC and FLOWBC)
  // TODO: Does this need to be done for OPENLOOPCORONARYBC and RESISTANCEBC
  // too?
  if ((block_type == BlockType::pressure_bc) ||
      (block_type == BlockType::flow_bc)) {
    std::vector<double> times_new;
    std::vector<double> values_new;
    int num_time_pts = (int)params[0];
    for (int i = 0; i < num_time_pts; i++) {
      times_new.push_back(params[1 + i]);
      values_new.push_back(params[1 + num_time_pts + i]);
    }
    model->get_parameter(block->global_param_ids[0])
        ->update(times_new, values_new);
  } else {
    if (block->global_param_ids.size() != params.size()) {
      throw std::runtime_error(
          "New parameter vector (given size = " +
          std::to_string(params.size()) +
          ") does not match number of parameters of block " + block_name +
          " (required size = " +
          std::to_string(block->global_param_ids.size()) + ")");
    }
    for (size_t i = 0; i < params.size(); i++) {
      model->get_parameter(block->global_param_ids[i])->update(params[i]);
      // parameter_values vector needs to be seperately updated for constant
      // parameters. This does not need to be done for time-dependent parameters
      // because it is handled in Model::update_time
      model->update_parameter_value(block->global_param_ids[i], params[i]);
    }
  }
}

/**
 * @brief Read the parameters of a particular block.
 *
 * @param problem_id The returned ID used to identify the 0D problem.
 * @param block name The name of the block to read.
 * @param params Parameters of the block (structure depends on block type).
 */
void read_block_params(int problem_id, std::string block_name,
                       std::vector<double>& params) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;
  auto block = model->get_block(block_name);
  if (params.size() != block->global_param_ids.size()) {
    throw std::runtime_error(
        "Parameter vector (given size = " + std::to_string(params.size()) +
        ") does not match number of parameters of block " + block_name +
        " (required size = " + std::to_string(block->global_param_ids.size()) +
        ")");
  }
  for (size_t i = 0; i < params.size(); i++) {
    params[i] = model->get_parameter_value(block->global_param_ids[i]);
  }
}

/**
 * @brief Return the IDs of the input and output nodes in the solution vector
 * for a given block.
 *
 * @param problem_id The returned ID used to identify the 0D problem.
 * @param block name The name of the block whose node IDs are returned.
 * @param IDs Vector containing IDs of input and output nodes in the following
 * order: {num inlet nodes, inlet flow[0], inlet pressure[0],..., num outlet
 * nodes, outlet flow[0], outlet pressure[0],...}.
 */
void get_block_node_IDs(int problem_id, std::string block_name,
                        std::vector<int>& IDs) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;

  // Find the required block
  auto block = model->get_block(block_name);

  // IDs are stored in the following format
  // {num inlet nodes, inlet flow[0], inlet pressure[0],..., num outlet nodes,
  // outlet flow[0], outlet pressure[0],...}
  IDs.clear();
  IDs.push_back(block->inlet_nodes.size());
  for (int i = 0; i < block->inlet_nodes.size(); i++) {
    IDs.push_back(block->inlet_nodes[i]->flow_dof);
    IDs.push_back(block->inlet_nodes[i]->pres_dof);
  }
  IDs.push_back(block->outlet_nodes.size());
  for (int i = 0; i < block->outlet_nodes.size(); i++) {
    IDs.push_back(block->outlet_nodes[i]->flow_dof);
    IDs.push_back(block->outlet_nodes[i]->pres_dof);
  }
}

/**
 * @brief Return the y state vector.
 *
 * @param problem_id The ID used to identify the 0D problem.
 * @param y The state vector containing all state.y degrees-of-freedom.
 */
void return_y(int problem_id, std::vector<double>& y) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;
  auto system_size = interface->system_size_;
  if (y.size() != system_size) {
    throw std::runtime_error(
        "ERROR: State vector size is wrong in return_y().");
  }

  auto state = interface->state_;
  for (int i = 0; i < system_size; i++) {
    y[i] = state.y[i];
  }
}

/**
 * @brief Return the ydot state vector.
 *
 * @param problem_id The ID used to identify the 0D problem.
 * @param ydot The state vector containing all state.ydot degrees-of-freedom.
 */
void return_ydot(int problem_id, std::vector<double>& ydot) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;
  auto system_size = interface->system_size_;
  if (ydot.size() != system_size) {
    throw std::runtime_error(
        "ERROR: State vector size is wrong in return_ydot().");
  }

  auto state = interface->state_;
  for (int i = 0; i < system_size; i++) {
    ydot[i] = state.ydot[i];
  }
}

/**
 * @brief Update the state vector.
 *
 * @param problem_id The ID used to identify the 0D problem.
 * @param new_state_y The new state vector containing all state.y
 * degrees-of-freedom.
 * @param new_state_ydot The new state vector containing all state.ydot
 * degrees-of-freedom.
 */
void update_state(int problem_id, std::vector<double> new_state_y,
                  std::vector<double> new_state_ydot) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;
  auto system_size = interface->system_size_;
  if ((new_state_y.size() != system_size) ||
      (new_state_ydot.size() != system_size)) {
    throw std::runtime_error(
        "ERROR: State vector size is wrong in update_state().");
  }

  auto state = interface->state_;
  for (int i = 0; i < system_size; i++) {
    state.y[i] = new_state_y[i];
    state.ydot[i] = new_state_ydot[i];
  }
  interface->state_ = state;
}

/**
 * @brief Return the analytical coupling Jacobian dP/dQ for a coupled
 * external_solver_coupling_block (FlowReferenceBC). Replaces the FD
 * perturbation probe in svMultiPhysics' set_bc.cpp:calc_der_cpl_bc.
 * Must be called after a converged run_simulation/increment_time so the
 * SparseLU factorization left in SparseSystem is current.
 *
 * @param problem_id The ID used to identify the 0D problem.
 * @param block_name Name of the external_solver_coupling_block (e.g. "LV_3D").
 * @param dP_dQ OUT: scalar derivative of the coupled pressure DOF wrt
 * the externally-prescribed flow DOF.
 */
void get_coupling_jacobian(int problem_id, std::string block_name,
                           double& dP_dQ) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;
  auto block = model->get_block(block_name);
  // The external_solver_coupling_block is a FlowReferenceBC with exactly
  // one connected node total, on either the inlet side (location=outlet
  // in the JSON) or the outlet side (location=inlet). Pick whichever is
  // populated; throw if both are empty (misconfigured) or both populated
  // (unexpected topology).
  Node* node = nullptr;
  if (!block->inlet_nodes.empty() && block->outlet_nodes.empty()) {
    node = block->inlet_nodes[0];
  } else if (block->inlet_nodes.empty() && !block->outlet_nodes.empty()) {
    node = block->outlet_nodes[0];
  } else {
    throw std::runtime_error(
        "ERROR: coupling block '" + block_name +
        "' has unexpected node topology in get_coupling_jacobian() "
        "(expected exactly one connected node).");
  }
  // Pass the FlowReferenceBC's equation row index (where ∂C/∂Q_input has
  // its single nonzero entry, per FlowReferenceBC.cpp:16) as the "q_dof"
  // arg. This is the correct RHS row for the sensitivity solve
  // J·w = e_{eqn} — NOT the flow variable's column index. The two happen
  // to coincide in some simple LPNs (test_04) by DOFHandler registration
  // ordering, but in general they differ; using flow_dof gave wildly
  // wrong dP/dQ for the cardiac closed-loop topology (off by 13 orders
  // of magnitude vs FD probe).
  if (block->global_eqn_ids.empty()) {
    throw std::runtime_error(
        "ERROR: coupling block '" + block_name +
        "' has no equation rows in get_coupling_jacobian().");
  }
  const int eqn_row = block->global_eqn_ids[0];
  dP_dQ = interface->integrator_.get_dP_dQ(node->pres_dof, eqn_row);
}

/**
 * @brief Increment the 0D solution by one time step.
 *
 * @param problem_id The ID used to identify the 0D problem.
 * @param external_time The current time in the external program.
 * @param solution The solution vector containing all degrees-of-freedom.
 */
void increment_time(int problem_id, const double external_time,
                    std::vector<double>& solution) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;

  auto time_step_size = interface->time_step_size_;
  auto absolute_tolerance = interface->absolute_tolerance_;
  auto max_nliter = interface->max_nliter_;
  Integrator integrator(model.get(), time_step_size, interface->rho_infty_,
                        absolute_tolerance, max_nliter);
  auto state = interface->state_;
  // External-step boundary: refresh per-block per-step caches (e.g.
  // PiecewiseValve R_cached) from the canonical state at the start of this
  // external step. See note in Integrator::step.
  model->prepare_step(state.y, state.ydot);
  interface->state_ = integrator.step(state, external_time);
  interface->time_step_ += 1;

  for (int i = 0; i < state.y.size(); i++) {
    solution[i] = state.y[i];
  }
}

/**
 * @brief Run a full 0D simululation.
 *
 * @param problem_id The ID used to identify the 0D problem.
 * @param external_time The current time in the external program.
 * @param output_times Vector containing time-stamps for output_solution
 * vectors.
 * @param output_solutions The solution vector containing all degrees-of-freedom
 * stored sequentially (1D vector).
 * @param error_code This is 1 if a NaN is found in the solution vector, 0
 * otherwise.
 */
void run_simulation(int problem_id, const double external_time,
                    std::vector<double>& output_times,
                    std::vector<double>& output_solutions, int& error_code) {
  auto interface = SolverInterface::interface_list_[problem_id];
  auto model = interface->model_;

  auto time_step_size = interface->time_step_size_;
  auto absolute_tolerance = interface->absolute_tolerance_;
  auto max_nliter = interface->max_nliter_;
  auto num_time_steps = interface->num_time_steps_;
  auto system_size = interface->system_size_;
  auto num_output_steps = interface->num_output_steps_;

  // NOTE: must be a reference, not a copy, so that updates to integrator
  // state (y_coeff via update_params, the LU factorization in
  // system.solver, internal counters) are visible to subsequent interface
  // queries like get_coupling_jacobian. The previous `auto integrator =
  // ...` copy left the per-step factorization on a temporary that died at
  // function exit AND left interface->integrator_.y_coeff stale, which
  // silently broke any post-run access to integrator state.
  auto& integrator = interface->integrator_;
  integrator.update_params(time_step_size);

  // Auto-arm forward-sensitivity tracking for every FlowReferenceBC in the
  // model. For each tracked Q dof, the Integrator propagates dy/dQ through
  // all internal steps so get_coupling_jacobian/get_dP_dQ returns the
  // correct multi-step accumulated dP/dQ — not just the last internal
  // step's contribution. This makes the analytical-Jacobian coupling path
  // exact for any number_of_time_pts ≥ 2 (the legacy single-step formula
  // is only correct at N == 2). Cost per step: one extra LU back-solve
  // per FlowReferenceBC, against the same factorization the Newton solve
  // just used (Eigen::SparseLU::solve, ~µs for typical LPN sizes).
  // Track the EQUATION ROW of each FlowReferenceBC (where ∂C/∂Q_input has
  // its single nonzero, per FlowReferenceBC.cpp:16), not the flow variable
  // column index. These are independent IDs in the DOFHandler — the
  // legacy code conflated them, which works for trivial LPNs by
  // registration-order coincidence but produces wildly wrong dP/dQ for
  // closed-loop cardiac topologies.
  std::vector<int> coupled_eqn_rows;
  for (auto& blk_ptr : model->get_blocks()) {
    if (!blk_ptr) continue;
    if (blk_ptr->block_type != BlockType::flow_bc) continue;
    if (blk_ptr->global_eqn_ids.empty()) continue;
    coupled_eqn_rows.push_back(blk_ptr->global_eqn_ids[0]);
  }
  // External step duration: in the run_simulation loop below, time advances
  // by time_step_size per iteration, total (num_time_steps - 1) iterations.
  const double external_dt = (num_time_steps - 1) * time_step_size;
  integrator.arm_sensitivity(coupled_eqn_rows, external_dt, external_time);

  auto state = interface->state_;
  double time = external_time;

  interface->times_[0] = time;
  interface->states_[0] = state;

  // External-step boundary: refresh per-block per-step caches (e.g.
  // PiecewiseValve R_cached) from the canonical state at the start of this
  // external step. Note that run_simulation may be called multiple times per
  // external timestep by svMP (assembly + line-search trials) — each call
  // re-prepares from the SAME state.y (interface->state_ is updated only by
  // increment_time/update_state between external steps), so R_cached is
  // consistent across all calls within the external step. See note in
  // Integrator::step.
  model->prepare_step(state.y, state.ydot);

  // Run integrator
  interface->time_step_ = 0;
  error_code = 0;
  bool isNaN = false;
  for (int i = 1; i < num_time_steps; i++) {
    interface->time_step_ += 1;
    state = integrator.step(state, time);
    // Check for NaNs in the state vector
    if ((i % 100) == 0) {
      for (int j = 0; j < system_size; j++) {
        isNaN = (state.y[j] != state.y[j]);
        if (isNaN) {
          std::cout << "Found NaN in state vector at timestep " << i
                    << " and index " << j << std::endl;
          error_code = 1;
          return;
        }
      }
    }
    time += time_step_size;
    interface->times_[i] = time;
    interface->states_[i] = state;
  }
  interface->state_ = state;

  // Write states to solution output vector
  if (output_solutions.size() != num_output_steps * system_size) {
    throw std::runtime_error("Solution vector size is wrong.");
  }
  int output_idx = 0;
  int soln_idx = 0;
  int start_idx = 0;
  double start_time = 0.0;
  if (interface->output_last_cycle_only_) {  // NOT TESTED
    start_idx = interface->num_time_steps_ - interface->pts_per_cycle_;
    start_time = interface->times_[start_idx];
    throw std::runtime_error(
        "ERROR: Option output_last_cycle_only has been implemented but not "
        "tested when using the svZeroDSolver interface library. Please test "
        "this "
        "functionality before removing this message.");
  }
  for (int t = start_idx; t < num_output_steps; t++) {
    state = interface->states_[t];
    output_times[t] = interface->times_[t] - start_time;
    for (int i = 0; i < system_size; i++) {
      soln_idx = output_idx * system_size + i;
      output_solutions[soln_idx] = state.y[i];
    }
    output_idx++;
  }

  // Release dynamic memory
  // integrator.clean();
}
