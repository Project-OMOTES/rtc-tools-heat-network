from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test

class TestAtesTemperature(TestCase):
    """
    Checks the constraints concerning the temperature changes in the ates as a result of heat loss and charging
    """

    def test_ates_temperature(self):
        import models.ates_temperature.src.run_ates_temperature as run_ates_temperature
        from models.ates_temperature.src.run_ates_temperature import HeatProblem

        basefolder = Path(run_ates_temperature.__file__).resolve().parent.parent

        solution = run_optimization_problem(HeatProblem, base_folder=basefolder)

        results = solution.extract_results()
        parameters = solution.parameters(0)
        bounds = solution.bounds()

        times = solution.times()
        # check if
        # - discrete temperature ates is equal to temperature of pipe (carrier) at inport during discharging
        # - discrete temperature ates is equal or lower then temperature ates
        # - discrete tempearture ates is equal to the set booleans of ates tempearutre
        # - temperature change ates continues is equal to temperature loss and temperature change charging
        #
        # - only heatpump or heat exchanger is in operation, soly for charging/discharging ates
        # - if ates is charging (heat_ates>0), hex is enabled. if ates is discharging (heat<0), hp is enabled
        # - Discharge heat and mass flow is corresponding to the temperature regime (flow rate remains the same, but heat changes)

        #TODO: still have to add checks:
        # - temperature loss>= relation of temperature loss
        # - temperature addition charging
        # - heat loss ates>= relation of heat loss

        tol = 1e-6

        # demand_matching_test(solution, results)
        # energy_conservation_test(solution, results)
        # # heat_to_discharge_test(solution, results)

        ates_charging = results['Pipe1__flow_direct_var'] #=1 if charging
        ates_temperature = results['ATES_cb47.Temperature_ates']
        ates_temperature_disc = results['ATES_cb47__temperature_ates_disc']
        carrier_temperature = results['41770304791669983859190_temperature']
        temperature_regimes = solution.temperature_regimes(41770304791669983859190)

        ates_temperature_loss = results['ATES_cb47.Temperature_loss']
        ates_temperature_change_charging = results['ATES_cb47.Temperature_change_charging']

        heat_pump_sec = results['HeatPump_7f2c.Secondary_heat']
        heat_ates = results['ATES_cb47.Heat_ates']
        heat_loss_ates = results['ATES_cb47.Heat_loss']
        ates_stored_heat = results['ATES_cb47.Stored_heat']
        hex_disabled = results['HeatExchange_32ba__disabled']
        hp_disabled = results['HeatPump_7f2c__disabled']

        geo_source = results['GeothermalSource_4e5b.Heat_source']
        objective = solution.objective(0)

        objective_calc = results['GeothermalSource_4e5b__variable_operational_cost'] + sum(parameters['HeatPump_7f2c.variable_operational_cost_coefficient']*results['HeatPump_7f2c.Power_elec'][1:]*(times[
                                                                                                               1:]-times[:-1])/3600)

        feasibility = solution.solver_stats['return_status']

        self.assertTrue((feasibility=="Optimal"))

        np.testing.assert_array_less(ates_temperature_disc-tol, ates_temperature)
        np.testing.assert_allclose(ates_temperature_disc, sum([results[f"ATES_cb47__temperature_disc_{temp}"]*temp for temp in temperature_regimes]))
        np.testing.assert_allclose((1-ates_charging)*ates_temperature_disc, (1-ates_charging)*carrier_temperature)
        np.testing.assert_allclose(ates_temperature[1:]-ates_temperature[:-1], (ates_temperature_change_charging[1:] - ates_temperature_loss[1:])*(times[1:]-times[:-1]), atol=tol)

        np.testing.assert_allclose(heat_ates[1:]-heat_loss_ates[1:], (ates_stored_heat[1:]-ates_stored_heat[:-1])/(times[1:]-times[:-1]), atol=tol)
        np.testing.assert_array_less(heat_pump_sec, geo_source)

        #array less then because ates charging boolean can be either 0 or 1 when there is no flow, or just flow to compensate the heatloss
        np.testing.assert_array_less(np.ones(len(hex_disabled))-tol, hex_disabled+hp_disabled)
        np.testing.assert_array_less(ates_charging-tol, hp_disabled)
        np.testing.assert_array_less(ates_charging-tol, 1-hex_disabled)











