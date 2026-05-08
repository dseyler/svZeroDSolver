// test_04: validate the analytical dP/dQ entry point (get_coupling_jacobian)
// against finite-difference probing of the .so the same way svMultiPhysics
// does in set_bc.cpp:calc_der_cpl_bc. Catches sign-convention, scaling, and
// DOF-mapping errors.
//
// Fixture (svzerod_3Dcoupling.json): one FLOW external_solver_coupling_block
// (location=inlet, drives Q at the inlet of vessel0), one BloodVessel
// resistor R=100, one PressureReferenceBC at the outlet (P=0). Configured
// with number_of_time_pts=2 — a single internal 0D step per external 3D
// step — so the analytical local-step Jacobian equals the FD probe to
// machine precision. With multiple internal steps the analytical only
// captures the last-step Jacobian (no cumulative sensitivity propagation),
// see Integrator::get_dP_dQ() docstring.
//
// Usage:
//   ./svZeroD_interface_test04 <build_root> <path_to_json>

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
  throw std::runtime_error("Could not find libsvzero_interface.{so,dylib,dll} under " +
                           iface_dir.string());
}

// Drive the FLOW coupling block. Two time points: [N=2, t0, t1, Q0, Q1].
// To mirror svMultiPhysics' FD probe (which perturbs only Q at t_n+1 via
// CoupledBoundaryCondition::perturb_flowrate), set Q0 (Q at t_n, the
// 3D-step start) and Q1 (Q at t_n+1, the end) independently. The
// generalized-α update_time evaluates the parameter at t_n + α_f·Δt,
// giving an effective Q of (1-α_f)·Q0 + α_f·Q1 in the residual.
static void set_inflow(LPNSolverInterface& interface,
                       const std::string& block_name,
                       double q_old,
                       double q_new,
                       double dt) {
  std::vector<double> params = {2.0, 0.0, dt, q_old, q_new};
  interface.update_block_params(block_name, params);
}

static double run_step_and_read_pressure(LPNSolverInterface& interface,
                                         int p_dof,
                                         double t_current) {
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
    std::cerr << "Usage: svZeroD_interface_test04 <build_root> <path_to_json>"
              << std::endl;
    return 1;
  }

  LPNSolverInterface interface;
  load_lib(interface, fs::path(argv[1]));
  interface.initialize(std::string(argv[2]));

  const std::string flow_block = "FLOW_inlet";
  const double dt = 0.05;           // arbitrary 3D step size for this test
  const double Q0 = 1.0;
  const double eps = 1e-6;          // FD perturbation (relative to Q0)

  interface.set_external_step_size(dt);

  // Map FLOW_inlet block name -> (Q DOF, P DOF) of its single connected
  // node. For external_solver_coupling_block with location=inlet, the
  // node is on the outlet side; with location=outlet, on the inlet side.
  // ids layout: [n_inlet, q0, p0, ..., n_outlet, q0, p0, ...]
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
    throw std::runtime_error(
        "Unexpected node topology for coupling block " + flow_block);
  }
  std::cout << "FLOW_inlet inlet node DOFs: Q=" << q_dof << " P=" << p_dof
            << std::endl;

  // ---- 1. Run one converged step at Q=Q0 (constant), capture P0. ----
  set_inflow(interface, flow_block, Q0, Q0, dt);
  // Save the initial state so we can restore it for the FD perturbation.
  std::vector<double> y_init(interface.system_size_, 0.0);
  std::vector<double> ydot_init(interface.system_size_, 0.0);
  interface.update_state(y_init, ydot_init);
  // Capture baseline state at the START of the step (post-update_state).
  std::vector<double> y_baseline(interface.system_size_);
  std::vector<double> ydot_baseline(interface.system_size_);
  interface.return_y(y_baseline);
  interface.return_ydot(ydot_baseline);

  double P0 = run_step_and_read_pressure(interface, p_dof, 0.0);
  std::cout << "Baseline: Q=" << Q0 << "  P=" << P0 << std::endl;

  // ---- 2. Query analytical dP/dQ from the converged factorization. ----
  double dPdQ_analytical = 0.0;
  interface.get_coupling_jacobian(flow_block, dPdQ_analytical);
  std::cout << "Analytical dP/dQ = " << dPdQ_analytical << std::endl;

  // ---- 3. FD probe — mirror what set_bc.cpp:calc_der_cpl_bc does. ----
  // Restore the start-of-step state so the perturbed run sees the same
  // initial conditions, then re-run with Q=Q0+eps.
  // Match svMultiPhysics' perturb_flowrate(diff): bump only Q at t_n+1.
  // Q at t_n stays at Q0; the parameter ramps Q0 → Q0+eps over the step.
  interface.update_state(y_baseline, ydot_baseline);
  set_inflow(interface, flow_block, Q0, Q0 + eps, dt);
  double P_perturbed = run_step_and_read_pressure(interface, p_dof, 0.0);
  double dPdQ_fd = (P_perturbed - P0) / eps;
  std::cout << "FD       dP/dQ = " << dPdQ_fd
            << "  (P_perturbed=" << P_perturbed << ")" << std::endl;

  // ---- 4. Validate. ----
  // Primary correctness property: analytical must match FD to within O(eps)
  // noise. This is the quantity svMultiPhysics consumes via bc.r in
  // set_bc.cpp:calc_der_cpl_bc.
  bool ok = true;
  double err_fd = std::abs(dPdQ_analytical - dPdQ_fd) / std::abs(dPdQ_fd);
  std::cout << "Rel err analytical vs FD: " << err_fd << std::endl;
  if (err_fd > 1e-4) {
    std::cerr << "FAIL: analytical disagrees with FD by " << err_fd
              << " (rel)" << std::endl;
    ok = false;
  }
  // Sign sanity: for a positive-resistance topology dP/dQ must be > 0.
  // Catches a wrong-sign on the unit RHS in get_dP_dQ.
  if (dPdQ_analytical <= 0.0) {
    std::cerr << "FAIL: analytical dP/dQ is non-positive (sign convention?)"
              << std::endl;
    ok = false;
  }

  if (!ok) {
    return 1;
  }
  std::cout << "PASS" << std::endl;
  return 0;
}
