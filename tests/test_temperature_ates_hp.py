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
        #
        # - Discharge heat and mass flow is corresponding to the temperature regime (flow rate remains the same, but heat changes)
        ates_charging = results['Pipe1__flow_direct_var'] #=1 if charging
        ates_temperature = results['ATES_cb47.Temperature_ates']
        ates_temperature_disc = results['ATES_cb47__temperature_ates_disc']
        carrier_temperature = results['41770304791669983859190_temperature']

        ates_temperature_loss = results['ATES_cb47.Temperature_loss']
        ates_temperature_change_charging = results['ATES_cb47.Temperature_change_charging']
        ates_stored_heat = results['ATES_cb47.Stored_heat']
        ates_max_heat = bounds['ATES_cb47.Stored_heat'][1]

        # t_35 = results['ATES_cb47__temperature_disc_35.0']
        # t_40 = results['ATES_cb47__temperature_disc_40.0']
        t_50 = results['ATES_cb47__temperature_disc_50.0']
        t_70 = results['ATES_cb47__temperature_disc_70.0']

        heat_demand = results['HeatingDemand_1.Heat_demand']
        heat_pump_sec = results['HeatPump_7f2c.Secondary_heat']
        heat_pump_prim = results['HeatPump_7f2c.Primary_heat']
        heat_ates = results['ATES_cb47.Heat_ates']
        heat_ex_prim = results['HeatExchange_32ba.Primary_heat']
        heat_ex_sec = results['HeatExchange_32ba.Secondary_heat']
        heat_loss_ates = results['ATES_cb47.Heat_loss']

        geo_source = results['GeothermalSource_4e5b.Heat_source']
        # peak_source = results['GenericProducer_4dfe.Heat_source']
        objective = solution.objective(0)

        feasilibity = solution.solver_stats['return_status']


        np.testing.assert_array_less(ates_temperature_disc, ates_temperature)
        np.testing.assert_allclose(solution.get_timeseries(f"HeatingDemand_1.target_heat_demand").values[0:len(heat_demand)]- heat_demand)
        # np.testing.assert_allclose(t_40*40+t_50*50+t_70*70, ates_temperature_disc)

        np.testing.assert_allclose((1-ates_charging)*ates_temperature_disc, (1-ates_charging)*carrier_temperature)










