"""Tests for the PiecewiseValve freeze_state option.

freeze_state decides the valve resistance once per driver step, from the state
converged at the end of the previous step, instead of re-evaluating it at every
Newton iteration. When the valve never changes state there is nothing to lag,
so a frozen run must reproduce the unfrozen run. That is the property checked
here, and it is what distinguishes a correct implementation from one that
caches the wrong value or refreshes the cache at the wrong point.

tests/cases/piecewise_Chamber_and_Valve_frozen.json covers the code path on a
full closed-loop heart model, but only against its own stored output. These
tests are the ones that pin the physics.
"""

import copy

import numpy as np
import pytest

import pysvzerod


def valve_case(inlet_pressure, freeze_state, num_time_pts=201):
    """Two vessels separated by a PiecewiseValve, with a constant inlet pressure.

    The inlet pressure is held constant so that the sign of the pressure drop
    across the valve never changes and the valve stays in one state for the
    whole simulation.

    Args:
        inlet_pressure: Constant inlet pressure. Well above the distal pressure
            holds the valve open, well below it holds the valve closed.
        freeze_state: Value of the valve's freeze_state option.
        num_time_pts: Number of time points per cardiac cycle.
    """
    return {
        "simulation_parameters": {
            "number_of_cardiac_cycles": 3,
            "number_of_time_pts_per_cardiac_cycle": num_time_pts,
            "output_all_cycles": False,
            "steady_initial": True,
            "output_variable_based": True,
            "absolute_tolerance": 1e-9,
        },
        "boundary_conditions": [
            {
                "bc_name": "INLET",
                "bc_type": "PRESSURE",
                "bc_values": {"P": [inlet_pressure, inlet_pressure], "t": [0.0, 1.0]},
            },
            {
                "bc_name": "OUTLET",
                "bc_type": "RESISTANCE",
                "bc_values": {"Pd": 1.0, "R": 1000.0},
            },
        ],
        "vessels": [
            {
                "boundary_conditions": {"inlet": "INLET"},
                "vessel_id": 0,
                "vessel_length": 10.0,
                "vessel_name": "upstream_vessel",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"C": 1.0e-4, "R_poiseuille": 100.0},
            },
            {
                "boundary_conditions": {"outlet": "OUTLET"},
                "vessel_id": 1,
                "vessel_length": 10.0,
                "vessel_name": "downstream_vessel",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {"C": 1.0e-4, "R_poiseuille": 100.0},
            },
        ],
        "valves": [
            {
                "type": "PiecewiseValve",
                "name": "valve",
                "params": {
                    "Rmax": 100000.0,
                    "Rmin": 100.0,
                    "freeze_state": freeze_state,
                    "upstream_block": "upstream_vessel",
                    "downstream_block": "downstream_vessel",
                },
            }
        ],
        "junctions": [],
    }


def simulate(config):
    return pysvzerod.simulate(config).select_dtypes("number").values


@pytest.mark.parametrize(
    "inlet_pressure,expected_state",
    [(50.0, "open"), (-50.0, "closed")],
)
def test_freeze_matches_unfrozen_when_valve_does_not_switch(
    inlet_pressure, expected_state
):
    """A frozen valve that never switches must match the unfrozen solution.

    Freezing lags the valve state by one driver step. With a constant pressure
    drop there is no state change to lag, so the frozen and unfrozen runs solve
    the same problem and must agree to solver tolerance.
    """
    unfrozen = simulate(valve_case(inlet_pressure, False))
    frozen = simulate(valve_case(inlet_pressure, True))

    assert frozen.shape == unfrozen.shape
    np.testing.assert_allclose(
        frozen,
        unfrozen,
        rtol=1e-7,
        atol=1e-6,
        err_msg=f"frozen and unfrozen differ for a permanently {expected_state} valve",
    )


def test_freeze_state_defaults_to_off():
    """Omitting freeze_state must be identical to setting it to false."""
    explicit = valve_case(50.0, False)

    default = copy.deepcopy(explicit)
    del default["valves"][0]["params"]["freeze_state"]

    np.testing.assert_array_equal(simulate(default), simulate(explicit))


def test_freeze_state_changes_result_when_valve_switches():
    """Guard against the option being silently inert.

    On a model whose valve does switch, freezing must actually change the
    solution. Without this, the equivalence tests above would still pass if
    freeze_state were never plumbed through to the block at all.
    """
    import json
    import os

    case = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "cases",
        "piecewise_Chamber_and_Valve.json",
    )
    with open(case) as f:
        base = json.load(f)
    base["simulation_parameters"]["number_of_cardiac_cycles"] = 1

    unfrozen = copy.deepcopy(base)
    frozen = copy.deepcopy(base)
    for valve in frozen["valves"]:
        valve["params"]["freeze_state"] = True

    assert not np.allclose(simulate(frozen), simulate(unfrozen))


def test_freeze_state_rejected_by_other_valve_types():
    """freeze_state is specific to PiecewiseValve and must not be accepted
    silently by valve types that ignore it."""
    config = valve_case(50.0, True)
    config["valves"][0]["type"] = "ValveTanh"
    config["valves"][0]["params"]["Steepness"] = 100.0

    with pytest.raises(RuntimeError, match="Unknown parameter freeze_state"):
        pysvzerod.simulate(config)
