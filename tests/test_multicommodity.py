from pathlib import Path
from unittest import TestCase

import numpy as np


from rtctools.util import run_optimization_problem


class TestMultiCommodityHeatPump(TestCase):
    """Test to verify that the optimisation problem can handle multicommodity problems, relating
    electricity and heat"""

    def test_heat_pump_elec_min_heat(self):
        """Test to verify the optimisation of minimisation of the heat_source used, and thus
        exploiting the heatpump as much as possible, and minimum use of heat source at secondary
        side, this heat source should have zero heat production."""
        import models.unit_cases_electricity.heat_pump_elec.src.run_hp_elec as run_hp_elec
        from models.unit_cases_electricity.heat_pump_elec.src.run_hp_elec import HeatProblem2

        v_min = 230
        i_max = 142
        cop = 4

        base_folder = Path(run_hp_elec.__file__).resolve().parent.parent

        solution = run_optimization_problem(HeatProblem2, base_folder=base_folder)
        results = solution.extract_results()

        heatsource_prim = results["ResidualHeatSource_61b8.Heat_source"]
        heatsource_sec = results["ResidualHeatSource_aec9.Heat_source"]
        heatpump_power = results["GenericConversion_3d3f.Power_elec"]
        heatpump_heat_prim = results["GenericConversion_3d3f.Primary_heat"]
        heatpump_heat_sec = results["GenericConversion_3d3f.Secondary_heat"]
        heatdemand_sec = results["HeatingDemand_18aa.Heat_demand"]
        heatdemand_prim = results["HeatingDemand_3322.Heat_demand"]
        elec_prod_power = results["ElectricityProducer_ac2e.ElectricityOut.Power"]

        heatdemand_prim_target = solution.get_timeseries(
            "HeatingDemand_3322.target_heat_demand"
        ).values
        heatdemand_sec_target = solution.get_timeseries(
            "HeatingDemand_18aa.target_heat_demand"
        ).values

        heatpump_voltage = results["GenericConversion_3d3f.ElectricityIn.V"]
        heatpump_current = results["GenericConversion_3d3f.ElectricityIn.I"]

        # first check if demands are met
        np.testing.assert_allclose(heatdemand_sec_target, heatdemand_sec)
        np.testing.assert_allclose(heatdemand_prim_target, heatdemand_prim)

        # check that heatpump is providing more energy to secondary side than demanded
        np.testing.assert_array_less(heatdemand_sec - heatpump_heat_sec, 0)
        # check that producer is providing more energy to heatpump and primary demand
        np.testing.assert_array_less(heatdemand_prim - (heatsource_prim - heatpump_heat_prim), 0)
        # check that secondary producer does not provide heat
        np.testing.assert_allclose(heatsource_sec, np.zeros(len(heatsource_sec)))

        # check that heatpumppower*COP==secondaryheat heatpump
        np.testing.assert_allclose(heatpump_power * cop, heatpump_heat_sec)
        # check power consumption with current and voltage heatpump
        np.testing.assert_allclose(heatpump_power, heatpump_current * heatpump_voltage)
        np.testing.assert_array_less(heatpump_power, elec_prod_power)
        # check if current and voltage limits are satisfied
        np.testing.assert_array_less(heatpump_current, i_max * np.ones(len(heatpump_current)))
        np.testing.assert_allclose(v_min * np.ones(len(heatpump_voltage)), heatpump_voltage)

    def test_heat_pump_elec_min_heat_curr_limit(self):
        """Test to verify the optimisation of minimisation of the heat_source used, however due to
        limitations in the electricity transport through the cables, the power and thus the heat
        produced at the heatpump is limited, resulting in heat production by the secondary
        heatsource, e.g. the heat produced by this asset is not 0."""
        import models.unit_cases_electricity.heat_pump_elec.src.run_hp_elec as run_hp_elec
        from models.unit_cases_electricity.heat_pump_elec.src.run_hp_elec import HeatProblem

        base_folder = Path(run_hp_elec.__file__).resolve().parent.parent

        v_min = 230
        i_max = 142
        cop = 4

        solution = run_optimization_problem(HeatProblem, base_folder=base_folder)
        results = solution.extract_results()

        heatsource_prim = results["ResidualHeatSource_61b8.Heat_source"]
        heatsource_sec = results["ResidualHeatSource_aec9.Heat_source"]
        heatpump_power = results["GenericConversion_3d3f.Power_elec"]
        heatpump_heat_prim = results["GenericConversion_3d3f.Primary_heat"]
        heatpump_heat_sec = results["GenericConversion_3d3f.Secondary_heat"]
        heatdemand_sec = results["HeatingDemand_18aa.Heat_demand"]
        heatdemand_prim = results["HeatingDemand_3322.Heat_demand"]
        elec_prod_power = results["ElectricityProducer_ac2e.ElectricityOut.Power"]

        heatdemand_prim_target = solution.get_timeseries(
            "HeatingDemand_3322.target_heat_demand"
        ).values
        heatdemand_sec_target = solution.get_timeseries(
            "HeatingDemand_18aa.target_heat_demand"
        ).values

        heatpump_voltage = results["GenericConversion_3d3f.ElectricityIn.V"]
        heatpump_current = results["GenericConversion_3d3f.ElectricityIn.I"]

        np.testing.assert_allclose(heatdemand_sec_target, heatdemand_sec)
        np.testing.assert_allclose(heatdemand_prim_target, heatdemand_prim)

        # check that heatpump isnot providing enough energy to secondary side for demanded
        np.testing.assert_array_less(
            np.zeros(len(heatdemand_sec)), heatdemand_sec - heatpump_heat_sec
        )
        np.testing.assert_array_less(
            heatdemand_sec - (heatpump_heat_sec + heatsource_sec), np.zeros(len(heatdemand_sec))
        )
        # check that heatpump is limited by electric transport power limitations:
        np.testing.assert_allclose(heatpump_power, i_max * v_min * np.ones(len(heatpump_power)))
        # check that prim producer is providing more energy to heatpump and primary demand
        np.testing.assert_array_less(heatdemand_prim - (heatsource_prim - heatpump_heat_prim), 0)
        # check that heatpumppower*COP==secondaryheat heatpump
        np.testing.assert_allclose(heatpump_power * cop, heatpump_heat_sec)
        # check power consumption with current and voltage heatpump
        np.testing.assert_allclose(heatpump_power, heatpump_current * heatpump_voltage)
        np.testing.assert_array_less(heatpump_power, elec_prod_power)
        # check if current and voltage limits are satisfied
        np.testing.assert_allclose(heatpump_current, i_max * np.ones(len(heatpump_current)))
        np.testing.assert_allclose(v_min * np.ones(len(heatpump_voltage)), heatpump_voltage)
        # TODO: currently connecting pipes at HPs can not be disabled, these don't have the
        # functionality as this causes other problems with HP tests, have to adjust this later.
        # This option would be added/changed in asset_to_component_base

    def test_heat_pump_elec_min_elec(self):
        """Test to verify the optimisation of minimisation of the electricity power used, and thus
        exploiting the heatpump only for heat that can not directly be covered by other sources as
        possible."""
        import models.unit_cases_electricity.heat_pump_elec.src.run_hp_elec as run_hp_elec
        from models.unit_cases_electricity.heat_pump_elec.src.run_hp_elec import (
            ElectricityProblem,
        )

        base_folder = Path(run_hp_elec.__file__).resolve().parent.parent

        solution = run_optimization_problem(ElectricityProblem, base_folder=base_folder)
        results = solution.extract_results()

        heatsource_prim = results["ResidualHeatSource_61b8.Heat_source"]
        # heatsource_sec = results["ResidualHeatSource_aec9.Heat_source"]
        heatpump_power = results["GenericConversion_3d3f.Power_elec"]
        heatpump_heat_prim = results["GenericConversion_3d3f.Primary_heat"]
        heatpump_heat_sec = results["GenericConversion_3d3f.Secondary_heat"]
        heatpump_disabled = results["GenericConversion_3d3f__disabled"]
        heatdemand_sec = results["HeatingDemand_18aa.Heat_demand"]
        heatdemand_prim = results["HeatingDemand_3322.Heat_demand"]
        elec_prod_power = results["ElectricityProducer_ac2e.ElectricityOut.Power"]
        # pipe_sec_out_hp_disconnected = results["Pipe_408e__is_disconnected"]

        heatdemand_prim_target = solution.get_timeseries(
            "HeatingDemand_3322.target_heat_demand"
        ).values
        heatdemand_sec_target = solution.get_timeseries(
            "HeatingDemand_18aa.target_heat_demand"
        ).values

        np.testing.assert_allclose(heatdemand_sec_target, heatdemand_sec)
        np.testing.assert_allclose(heatdemand_prim_target, heatdemand_prim)

        # check that heatpump is not used:
        np.testing.assert_allclose(heatpump_power, np.zeros(len(heatpump_power)))
        np.testing.assert_allclose(heatpump_heat_sec, np.zeros(len(heatpump_heat_sec)))
        np.testing.assert_allclose(heatpump_heat_prim, np.zeros(len(heatpump_heat_prim)))

        np.testing.assert_allclose(elec_prod_power, np.zeros(len(heatpump_heat_prim)))
        np.testing.assert_allclose(heatpump_disabled, np.ones(len(heatpump_heat_prim)))

        # check that prim producer is providing more energy to heatpump and primary demand
        np.testing.assert_array_less(heatdemand_prim - (heatsource_prim - heatpump_heat_prim), 0)
