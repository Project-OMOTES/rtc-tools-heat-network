from unittest import TestCase

import numpy as np


def demand_matching_test(solution, results):
    len_times = 0.0
    for d in solution.heat_network_components.get("demand", []):
        if len(solution.times()) > 0:
            len_times = len(solution.times())
        else:
            len_times = len(solution.get_timeseries(f"{d}.target_heat_demand").values)
        target = solution.get_timeseries(f"{d}.target_heat_demand").values[0:len_times]
        np.testing.assert_allclose(target, results[f"{d}.Heat_demand"], atol=1.0e-3, rtol=1.0e-6)


def heat_to_discharge_test(solution, results):
    test = TestCase()
    tol = 1.0e-5
    for d in solution.heat_network_components.get("demand", []):
        cp = solution.parameters(0)[f"{d}.cp"]
        rho = solution.parameters(0)[f"{d}.rho"]
        return_T = solution.parameters(0)[f"{d}.T_return"]
        supply_T = solution.parameters(0)[f"{d}.T_supply"]
        dt = solution.parameters(0)[f"{d}.dT"]
        np.testing.assert_allclose(results[f"{d}.Heat_demand"], results[f"{d}.HeatIn.Heat"]-results[f"{d}.HeatOut.Heat"])
        np.testing.assert_allclose(results[f"{d}.HeatOut.Heat"],results[f"{d}.Q"] * rho * cp * return_T)
        test.assertTrue(expr=all(results[f"{d}.HeatIn.Heat"]<=results[f"{d}.Q"] * rho * cp * supply_T))
        test.assertTrue(expr=all(results[f"{d}.Heat_demand"] <= results[f"{d}.Q"] * rho * cp * dt))

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
        supply_T = solution.parameters(0)[f"{d}.T_supply"]
        test.assertTrue(expr=all(results[f"{d}.Heat_source"] >= results[f"{d}.Q"] * rho * cp * dt))
        np.testing.assert_allclose(results[f"{d}.HeatOut.Heat"],
                                   results[f"{d}.Q"] * rho * cp * supply_T)

    for d in solution.heat_network_components.get("ates", []):
        cp = solution.parameters(0)[f"{d}.cp"]
        rho = solution.parameters(0)[f"{d}.rho"]
        dt = solution.parameters(0)[f"{d}.dT"]
        supply_T = solution.parameters(0)[f"{d}.T_supply"]
        return_T = solution.parameters(0)[f"{d}.T_return"]
        test.assertTrue(expr=all(
            np.clip(results[f"{d}.HeatIn.Heat"], 0.0, np.inf) <=
            (np.clip(results[f"{d}.HeatIn.Q"], 0.0, np.inf) * rho * cp * supply_T+ tol)
        ))
        np.testing.assert_allclose(
            np.clip(results[f"{d}.HeatOut.Heat"], 0.0, np.inf),
            np.clip(results[f"{d}.HeatIn.Q"], 0.0, np.inf) * rho * cp * return_T,
        )
        np.testing.assert_allclose(
            np.clip(results[f"{d}.HeatIn.Heat"], -np.inf, 0.0),
            np.clip(results[f"{d}.HeatIn.Q"], -np.inf, 0.0) * rho * cp * supply_T,
        )

        test.assertTrue(expr=all(
            np.clip(results[f"{d}.HeatOut.Heat"], -np.inf, 0.0) +tol >=
            (np.clip(results[f"{d}.HeatIn.Q"], -np.inf, 0.0) * rho * cp * return_T)
        ))

    for p in solution.heat_network_components.get("pipe", []):
        cp = solution.parameters(0)[f"{p}.cp"]
        rho = solution.parameters(0)[f"{p}.rho"]
        dt = solution.parameters(0)[f"{p}.dT"]
        temperature = solution.parameters(0)[f"{p}.temperature"]
        test.assertTrue(
            expr=all(
                abs(results[f"{p}.HeatIn.Heat"]) <= abs(results[f"{p}.Q"] * rho * cp * temperature + tol)
            )
        )
        test.assertTrue(
            expr=all(abs(results[f"{p}.HeatOut.Heat"]) <= abs(results[f"{p}.Q"]) * rho * cp * temperature + tol)
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
        energy_sum -= results[f"{d}.Heat_ates"]

    for p in solution.heat_network_components.get("pipe", []):
        energy_sum -= np.ones(len(solution.times())) * results[f"{p}__hn_heat_loss"]

    np.testing.assert_allclose(energy_sum, 0.0, atol=1e-6)
