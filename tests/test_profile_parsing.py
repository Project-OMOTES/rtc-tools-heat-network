import unittest
from pathlib import Path
from typing import Optional

import esdl
import numpy as np
import pandas as pd

from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import InfluxDBProfileReader, ProfileReaderFromFile
from rtctools_heat_network.workflows import EndScenarioSizingStagedHIGHS


class MockInfluxDBProfileReader(InfluxDBProfileReader):

    def __init__(self, energy_system: esdl.EnergySystem,
                 file_path: Optional[Path]):
        super().__init__(energy_system, file_path)
        self._loaded_profiles = pd.read_csv(file_path, index_col="DateTime",
                                            parse_dates=True)

    def _load_profile_timeseries_from_database(self, profile: esdl.InfluxDBProfile) -> pd.Series:
        return self._loaded_profiles[profile.id]


class TestProfileLoading(unittest.TestCase):
    def test_loading_from_influx(self):
        import models.unit_cases.case_1a.src.run_1a as run_1a

        base_folder = Path(run_1a.__file__).resolve().parent.parent
        model_folder = base_folder / "model"
        input_folder = base_folder / "input"
        problem = EndScenarioSizingStagedHIGHS(
            esdl_parser=ESDLFileParser,
            base_folder=base_folder,
            model_folder=model_folder,
            input_folder=input_folder,
            esdl_file_name="1a_with_influx_profiles.esdl",
            # profile_reader=MockInfluxDBProfileReader,
            # input_timeseries_file="influx_mock.csv"
        )
        problem.pre()

        expected_array = np.array([1.0e8] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries(
            "WindPark_7f14.maximum_electricity_source").values)

        expected_array = np.array([1.0] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries(
            "elec.price_profile").values)

        expected_array = np.array([1.0e6] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries(
            "gas.price_profile").values)

    def test_loading_from_csv(self):
        import models.unit_cases_electricity.electrolyzer.src.example as example
        from models.unit_cases_electricity.electrolyzer.src.example import \
            MILPProblem

        base_folder = Path(example.__file__).resolve().parent.parent
        model_folder = base_folder / "model"
        input_folder = base_folder / "input"
        problem = MILPProblem(
            esdl_parser=ESDLFileParser,
            base_folder=base_folder,
            model_folder=model_folder,
            input_folder=input_folder,
            esdl_file_name="h2.esdl",
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries.csv"
        )
        problem.pre()

        expected_array = np.array([1.0e8] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries("WindPark_7f14.maximum_electricity_source").values)

        expected_array = np.array([1.0] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries("elec.price_profile").values)

        expected_array = np.array([1.0e6] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries("gas.price_profile").values)

    def test_loading_from_xml(self):
        import models.basic_source_and_demand.src.heat_comparison as heat_comparison
        from models.basic_source_and_demand.src.heat_comparison import HeatESDL
        base_folder = Path(heat_comparison.__file__).resolve().parent.parent
        model_folder = base_folder / "model"
        input_folder = base_folder / "input"
        problem = HeatESDL(
            esdl_parser=ESDLFileParser,
            base_folder=base_folder,
            model_folder=model_folder,
            input_folder=input_folder,
            esdl_file_name="model.esdl",
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries.xml"
        )
        problem.pre()

        expected_array = np.array([1.5e5] * 16 + [1.0e5] * 13 + [0.5e5] * 16)
        np.testing.assert_equal(expected_array, problem.get_timeseries("demand.target_heat_demand").values)

    def test_loading_from_csv_with_influx_profiles_given(self):
        import models.unit_cases_electricity.electrolyzer.src.example as example
        from models.unit_cases_electricity.electrolyzer.src.example import MILPProblem

        base_folder = Path(example.__file__).resolve().parent.parent
        model_folder = base_folder / "model"
        input_folder = base_folder / "input"
        problem = MILPProblem(
            esdl_parser=ESDLFileParser,
            base_folder=base_folder,
            model_folder=model_folder,
            input_folder=input_folder,
            esdl_file_name="h2_profiles_added_dummy_values.esdl",
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries.csv"
        )
        problem.pre()

        expected_array = np.array([1.0e8] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries(
            "WindPark_7f14.maximum_electricity_source").values)

        expected_array = np.array([1.0] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries(
            "elec.price_profile").values)

        expected_array = np.array([1.0e6] * 3)
        np.testing.assert_equal(expected_array, problem.get_timeseries(
            "gas.price_profile").values)

if __name__ == '__main__':
    unittest.main()
