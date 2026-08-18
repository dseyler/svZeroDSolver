// Test the PiecewiseValve freeze_state option through the external-solver
// interface.
//
// freeze_state decides the valve resistance once per driver step. When
// svZeroDSolver is driven by an external solver, one driver step is one
// external step, which run_simulation advances with many internal time steps.
// The resistance must therefore stay constant across every internal step of an
// external step, so that it cannot follow the trial solution the external
// solver supplies partway through the step.
//
// The model is a flow-driven vessel, a PiecewiseValve, and a vessel ending in a
// resistance BC. The imposed flow reverses within a single external step, which
// reverses the pressure drop across the valve, so an unfrozen valve switches
// state partway through the step. The test runs the same model twice, once with
// freeze_state on and once off, and checks that:
//
//   * the frozen valve holds one resistance for the whole external step, and
//   * the unfrozen valve does not, i.e. the model really does switch here and
//     the test is not passing vacuously.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../LPNSolverInterface/LPNSolverInterface.h"

namespace fs = std::filesystem;

namespace {

const double EXTERNAL_STEP_SIZE = 0.01;
const double EXTERNAL_TIME = 0.0;

// Flow imposed by the "external solver" at the start and end of the external
// step. The sign change is what drives the valve to switch.
const double FLOW_START = 10.0;
const double FLOW_END = -10.0;

void load_interface(LPNSolverInterface& interface, const fs::path& build_dir) {
  fs::path iface_dir = build_dir / "src" / "interface";
  fs::path lib_so = iface_dir / "libsvzero_interface.so";
  fs::path lib_dylib = iface_dir / "libsvzero_interface.dylib";
  fs::path lib_dll = iface_dir / "libsvzero_interface.dll";
  if (fs::exists(lib_so)) {
    interface.load_library(lib_so.string());
  } else if (fs::exists(lib_dylib)) {
    interface.load_library(lib_dylib.string());
  } else if (fs::exists(lib_dll)) {
    interface.load_library(lib_dll.string());
  } else {
    throw std::runtime_error("Could not find shared libraries " +
                             lib_so.string() + " or " + lib_dylib.string() +
                             " or " + lib_dll.string() + " !");
  }
}

// Advance one external step and return the effective valve resistance
// (p_in - p_out) / q_in at each internal time step of that external step.
std::vector<double> valve_resistance_over_external_step(
    const fs::path& build_dir, const std::string& config_file) {
  LPNSolverInterface interface;
  load_interface(interface, build_dir);
  interface.initialize(config_file);
  interface.set_external_step_size(EXTERNAL_STEP_SIZE);

  // Degree-of-freedom IDs at the valve. Format of IDs is
  // [num_inlets, (flow, pressure) per inlet, num_outlets, (flow, pressure) per
  // outlet]
  std::vector<int> ids;
  interface.get_block_node_IDs("valve", ids);
  int num_inlets = ids[0];
  int num_outlets = ids[1 + num_inlets * 2];
  if ((num_inlets != 1) || (num_outlets != 1)) {
    throw std::runtime_error("Wrong number of inlets/outlets for valve");
  }
  int inlet_flow_id = ids[1];
  int inlet_pressure_id = ids[2];
  int outlet_pressure_id = ids[5];

  // Impose a flow that reverses over the external step. Format of new_params
  // for flow/pressure blocks:
  // [N, time_1, ..., time_N, value_1, ..., value_N]
  std::vector<double> new_params(5);
  new_params[0] = 2.0;
  new_params[1] = EXTERNAL_TIME;
  new_params[2] = EXTERNAL_TIME + EXTERNAL_STEP_SIZE;
  new_params[3] = FLOW_START;
  new_params[4] = FLOW_END;
  interface.update_block_params("inflow_coupling", new_params);

  std::vector<double> solutions(interface.system_size_ *
                                interface.num_output_steps_);
  std::vector<double> times(interface.num_output_steps_);
  int error_code = 0;

  interface.run_simulation(EXTERNAL_TIME, times, solutions, error_code);
  if (error_code != 0) {
    throw std::runtime_error("svZeroD simulation failed for " + config_file);
  }

  std::vector<double> resistances;
  for (int step = 0; step < interface.num_output_steps_; step++) {
    int offset = interface.system_size_ * step;
    double p_in = solutions[offset + inlet_pressure_id];
    double p_out = solutions[offset + outlet_pressure_id];
    double q_in = solutions[offset + inlet_flow_id];

    // Skip steps where the flow is too close to zero to infer a resistance
    if (std::abs(q_in) < 1.0e-8) {
      continue;
    }
    resistances.push_back((p_in - p_out) / q_in);
  }
  if (resistances.size() < 2) {
    throw std::runtime_error("Not enough usable time steps to infer resistance");
  }
  return resistances;
}

double relative_spread(const std::vector<double>& values) {
  double lo = values[0];
  double hi = values[0];
  for (double v : values) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  return std::abs(hi - lo) / std::max(std::abs(hi), 1.0e-30);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    throw std::runtime_error(
        "Usage: svZeroD_interface_test04 <path_to_svZeroDSolver_build_folder> "
        "<path_to_frozen_json> <path_to_unfrozen_json>");
  }
  fs::path build_dir = argv[1];

  std::vector<double> frozen =
      valve_resistance_over_external_step(build_dir, std::string(argv[2]));
  std::vector<double> unfrozen =
      valve_resistance_over_external_step(build_dir, std::string(argv[3]));

  double frozen_spread = relative_spread(frozen);
  double unfrozen_spread = relative_spread(unfrozen);

  std::cout << "Simulation output:" << std::endl;
  std::cout << "  frozen   valve resistance spread over external step = "
            << frozen_spread << std::endl;
  std::cout << "  unfrozen valve resistance spread over external step = "
            << unfrozen_spread << std::endl;

  // The frozen valve must hold a single resistance for the whole external step.
  const double frozen_tolerance = 1.0e-8;
  if (frozen_spread > frozen_tolerance) {
    throw std::runtime_error(
        "Frozen valve resistance changed during an external step.");
  }

  // The unfrozen valve must not, otherwise the model never switches here and
  // the check above would pass no matter what freeze_state did.
  const double switching_threshold = 1.0e-2;
  if (unfrozen_spread < switching_threshold) {
    throw std::runtime_error(
        "Unfrozen valve did not switch during the external step, so this test "
        "cannot distinguish a frozen valve from an unfrozen one.");
  }

  std::cout << "Test passed." << std::endl;
  return 0;
}
