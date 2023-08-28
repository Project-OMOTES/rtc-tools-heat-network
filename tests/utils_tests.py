from unittest import TestCase

import numpy as np


def demand_matching_test(solution, results):
    for d in solution.heat_network_components.get("demand", []):
        target = solution.get_timeseries(f"{d}.target_heat_demand").values
        np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])


def heat_to_discharge_test(solution, results):
    test = TestCase()
    tol = 1.0e-6
    for d in solution.heat_network_components.get("demand", []):
        cp = solution.parameters(0)[f"{d}.cp"]
        rho = solution.parameters(0)[f"{d}.rho"]
        dt = solution.parameters(0)[f"{d}.dT"]
        np.testing.assert_allclose(results[f"{d}.Heat_demand"], results[f"{d}.Q"] * rho * cp * dt)

    for d in solution.heat_network_components.get("buffer", []):
        cp = solution.parameters(0)[f"{d}.cp"]
        rho = solution.parameters(0)[f"{d}.rho"]
        dt = solution.parameters(0)[f"{d}.dT"]
        np.testing.assert_allclose(
            np.clip(results[f"{d}.Heat_buffer"], 0.0, np.inf),
            np.clip(results[f"{d}.HeatIn.Q"], 0.0, np.inf) * rho * cp * dt,
        )
        test.assertTrue(
            expr=all(
                np.clip(results[f"{d}.Heat_buffer"], -np.inf, 0.0)
                <= np.clip(results[f"{d}.HeatIn.Q"], -np.inf, 0.0) * rho * cp * dt
            )
        )

    for d in solution.heat_network_components.get("source", []):
        cp = solution.parameters(0)[f"{d}.cp"]
        rho = solution.parameters(0)[f"{d}.rho"]
        dt = solution.parameters(0)[f"{d}.dT"]
        test.assertTrue(expr=all(results[f"{d}.Heat_source"] >= results[f"{d}.Q"] * rho * cp * dt))

    for d in solution.heat_network_components.get("ates", []):
        cp = solution.parameters(0)[f"{d}.cp"]
        rho = solution.parameters(0)[f"{d}.rho"]
        dt = solution.parameters(0)[f"{d}.dT"]
        test.assertTrue(
            expr=all(
                np.clip(results[f"{d}.Heat_ates"], 0.0, np.inf)
                >= np.clip(results[f"{d}.Q"], 0.0, np.inf) * rho * cp * dt
            )
        )
        np.testing.assert_allclose(
            np.clip(results[f"{d}.Heat_ates"], -np.inf, 0.0),
            np.clip(results[f"{d}.Q"], -np.inf, 0.0) * rho * cp * dt,
        )

    for p in solution.hot_pipes:
        cp = solution.parameters(0)[f"{p}.cp"]
        rho = solution.parameters(0)[f"{p}.rho"]
        dt = solution.parameters(0)[f"{p}.dT"]
        test.assertTrue(
            expr=all(
                abs(results[f"{p}.HeatIn.Heat"]) + tol >= abs(results[f"{p}.Q"] * rho * cp * dt)
            )
        )
        test.assertTrue(
            expr=all(abs(results[f"{p}.HeatOut.Heat"]) + tol >= results[f"{p}.Q"] * rho * cp * dt)
        )


def energy_conservation_test(solution, results):
    energy_sum = np.zeros(len(solution.times()))

    for d in solution.heat_network_components.get("demand", []):
        energy_sum -= results[f"{d}.Heat_demand"]

    for d in solution.heat_network_components.get("buffer", []):
        energy_sum -= results[f"{d}.Heat_buffer"]

    for d in solution.heat_network_components.get("source", []):
        energy_sum += results[f"{d}.Heat_source"]

    for d in solution.heat_network_components.get("ates", []):
        energy_sum += results[f"{d}.Heat_ates"]

    for p in solution.heat_network_components.get("pipe", []):
        energy_sum -= np.ones(len(solution.times())) * results[f"{p}__hn_heat_loss"]

    np.testing.assert_allclose(energy_sum, 0.0, atol=1.0e-6)
