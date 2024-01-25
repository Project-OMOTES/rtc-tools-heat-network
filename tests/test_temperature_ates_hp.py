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

        results['ATES_cb47__temperature_disc_50.0']
        results['ATES_cb47__temperature_disc_70.0']

        np.testing.assert_array_less(ates_temperature_disc, ates_temperature)
        np.testing.assert_allclose((1-ates_charging)*ates_temperature_disc, (1-ates_charging)*carrier_temperature)










