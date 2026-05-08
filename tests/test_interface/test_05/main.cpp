// test_05: validate the analytical dP/dQ entry point (get_coupling_jacobian)
// against finite-difference probing for the MULTI-STEP regime
// (number_of_time_pts > 2). Exercises the forward-sensitivity recursion
// added in Integrator::step / arm_sensitivity / get_dP_dQ. Without that
// recursion, the legacy single-step formula misses propagation through
// the prior N-1 internal steps and disagrees with FD by O(1) factors.
//
// Fixture (svzerod_3Dcoupling_N{5,10}.json): same FLOW->RCL->Pressure
// topology as test_04 but with R, C, L all nonzero so the response is
// time-dependent (a pure resistor would mask the bug since dP/dQ = R
// regardless of N). number_of_time_pts is varied to test multiple
// internal-step regimes.
//
// Usage:
//   ./svZeroD_interface_test05 <build_root> <path_to_json>
//
// Pass criterion: |analytical - FD| / |FD| < 1e-4 for ALL N values tested.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../LPNSolverInterface/LPNSolverInterface.h"

namespace fs = std::filesystem;

static void load_lib(LPNSolverInterface& interface, const fs::path& build_dir) {
  fs::path iface_dir = build_dir / "src" / "interface";
  for (const std::string ext : {".so", ".dylib", ".dll"}) {
    fs::path candidate = iface_dir / ("libsvzero_interface" + ext);
    if (fs::exists(candidate)) {
      interface.load_library(candidate.string());
      return;
    }
  }
  throw std::runtime_error(
      "Could not find libsvzero_interface.{so,dylib,dll} under " +
      iface_dir.string());
}

// Set FlowReferenceBC: linear ramp Q from q_old at t=0 to q_new at t=dt.
// Matches svMultiPhysics' perturb_flowrate convention: only Q at t_n+1 is
// perturbed during the FD probe.
static void set_inflow(LPNSolverInterface& interface,
                       const std::string& block_name, double q_old,
                       double q_new, double dt) {
  std::vector<double> params = {2.0, 0.0, dt, q_old, q_new};
  interface.update_block_params(block_name, params);
}

static double run_step_and_read_pressure(LPNSolverInterface& interface,
                                         int p_dof, double t_current) {
  std::vector<double> times(interface.num_output_steps_);
  std::vector<double> solutions(interface.system_size_ *
                                interface.num_output_steps_);
  int error_code = 0;
  interface.run_simulation(t_current, times, solutions, error_code);
  if (error_code != 0) {
    throw std::runtime_error("0D solver diverged in run_step_and_read_pressure");
  }
  // Final timestep's value of the requested DOF.
  int last = interface.num_output_steps_ - 1;
  return solutions[interface.system_size_ * last + p_dof];
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: svZeroD_interface_test05 <build_root> <path_to_json>"
              << std::endl;
    return 1;
  }

  LPNSolverInterface interface;
  load_lib(interface, fs::path(argv[1]));
  interface.initialize(std::string(argv[2]));

  std::cout << "[test_05] N internal steps per run_simulation = "
            << interface.num_output_steps_ << std::endl;

  const std::string flow_block = "FLOW_inlet";
  const double dt = 0.05;
  const double Q0 = 1.0;
  const double eps = 1e-6;

  interface.set_external_step_size(dt);

  // Map FLOW_inlet block name -> (Q DOF, P DOF) of its single connected node.
  std::vector<int> ids;
  interface.get_block_node_IDs(flow_block, ids);
  int n_inlet = ids[0];
  int n_outlet = ids[1 + n_inlet * 2];
  int q_dof, p_dof;
  if (n_inlet == 1 && n_outlet == 0) {
    q_dof = ids[1];
    p_dof = ids[2];
  } else if (n_inlet == 0 && n_outlet == 1) {
    q_dof = ids[2];
    p_dof = ids[3];
  } else {
    throw std::runtime_error("Unexpected node topology for " + flow_block);
  }

  // Save zero-state initial condition.
  std::vector<double> y_init(interface.system_size_, 0.0);
  std::vector<double> ydot_init(interface.system_size_, 0.0);

  // ---- 1. Baseline run at Q=Q0, capture P0. ----
  set_inflow(interface, flow_block, Q0, Q0, dt);
  interface.update_state(y_init, ydot_init);

  std::vector<double> y_baseline(interface.system_size_);
  std::vector<double> ydot_baseline(interface.system_size_);
  interface.return_y(y_baseline);
  interface.return_ydot(ydot_baseline);

  double P0 = run_step_and_read_pressure(interface, p_dof, 0.0);
  std::cout << "Baseline:   Q=" << Q0 << "  P_final=" << P0 << std::endl;

  // ---- 2. Analytical dP/dQ (now uses forward-sensitivity accumulation) ----
  double dPdQ_analytical = 0.0;
  interface.get_coupling_jacobian(flow_block, dPdQ_analytical);
  std::cout << "Analytical dP/dQ = " << dPdQ_analytical << std::endl;

  // ---- 3. FD probe — re-run with Q=Q0+eps, same start-of-step state. ----
  interface.update_state(y_baseline, ydot_baseline);
  set_inflow(interface, flow_block, Q0, Q0 + eps, dt);
  double P_perturbed = run_step_and_read_pressure(interface, p_dof, 0.0);
  double dPdQ_fd = (P_perturbed - P0) / eps;
  std::cout << "FD          dP/dQ = " << dPdQ_fd
            << "  (P_perturbed=" << P_perturbed << ")" << std::endl;

  // ---- 4. Validate. ----
  // Pass criterion: relative agreement to ~1e-4 (limited by FD step eps and
  // the 0D Newton tolerance, not by the analytical accuracy).
  bool ok = true;
  double err_fd = std::abs(dPdQ_analytical - dPdQ_fd) /
                  std::max(std::abs(dPdQ_fd), 1e-30);
  std::cout << "Rel err analytical vs FD: " << err_fd << std::endl;
  if (err_fd > 1e-4) {
    std::cerr << "FAIL: analytical disagrees with FD by " << err_fd
              << " (rel)" << std::endl;
    ok = false;
  }
  if (dPdQ_analytical <= 0.0) {
    std::cerr << "FAIL: analytical dP/dQ is non-positive (sign convention?)"
              << std::endl;
    ok = false;
  }

  if (!ok) return 1;
  std::cout << "PASS" << std::endl;
  return 0;
}
