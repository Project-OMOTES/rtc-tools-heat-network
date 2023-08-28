import base64
import datetime
import logging
import xml.etree.ElementTree as ET  # noqa: N817
from datetime import timedelta
from pathlib import Path
from typing import Dict, Union

import esdl
from esdl.esdl_handler import EnergySystemHandler

import numpy as np

import pandas as pd

from pyecore.resources import ResourceSet

import rtctools.data.pi as pi
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.io_mixin import IOMixin

from rtctools_heat_network.esdl.asset_to_component_base import _AssetToComponentBase
from rtctools_heat_network.esdl.edr_pipe_class import EDRPipeClass
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.influxdb.profile import parse_esdl_profiles
from rtctools_heat_network.modelica_component_type_mixin import (
    ModelicaComponentTypeMixin,
)
from rtctools_heat_network.pipe_class import PipeClass
from rtctools_heat_network.pycml.pycml_mixin import PyCMLMixin
from rtctools_heat_network.qth_mixin import QTHMixin

from .common import Asset
from .esdl_heat_model import ESDLHeatModel
from .esdl_qth_model import ESDLQTHModel

logger = logging.getLogger("rtctools_heat_network")


ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}
DEFAULT_START_TIMESTAMP = "2017-01-01T00:00:00+00:00"
DEFAULT_END_TIMESTAMP = "2018-01-01T00:00:00+00:00"


class _ESDLInputException(Exception):
    pass


class ESDLMixin(
    ModelicaComponentTypeMixin,
    IOMixin,
    PyCMLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    esdl_run_info_path: Path = None

    esdl_pi_validate_timeseries = False

    esdl_pi_input_data_config = None
    esdl_pi_output_data_config = None

    __max_supply_temperature = None

    # TODO: remove this once ESDL allows specifying a minimum pipe size for an optional pipe.
    __minimum_pipe_size_name = "DN150"

    def __init__(self, *args, **kwargs):
        esdl_string = kwargs.get("esdl_string", None)
        if esdl_string is not None:
            esdl_path = None
            self.esdl_string = base64.b64decode(esdl_string).decode("utf-8")
            self.__run_info = None
        else:
            self.esdl_string = None
            self.esdl_run_info_path = kwargs.get(
                "esdl_run_info_path", Path(kwargs["input_folder"]) / "RunInfo.xml"
            )

            if not self.esdl_pi_input_data_config:
                self.esdl_pi_input_data_config = _ESDLInputDataConfig

            if not self.esdl_pi_output_data_config:
                self.esdl_pi_output_data_config = _ESDLOutputDataConfig

            self.__run_info = _RunInfoReader(self.esdl_run_info_path)
            esdl_path = self.__run_info.esdl_file

        self.__esdl_assets, self._profiles, self.__esdl_carriers = _esdl_to_assets(
            self.esdl_string, esdl_path
        )

        if self.__run_info is not None and self.__run_info.parameters_file is not None:
            self.__esdl_assets = _overwrite_parameters(
                self.__run_info.parameters_file, self.__esdl_assets
            )

        # This way we allow users to adjust the parsed ESDL assets
        assets = self.esdl_assets

        # Although we work with the names, the FEWS import data uses the component IDs
        self.__timeseries_id_map = {a.id: a.name for a in assets.values()}

        if isinstance(self, HeatMixin):
            self.__model = ESDLHeatModel(assets, **self.esdl_heat_model_options())
        else:
            assert isinstance(self, QTHMixin)

            # Maximum supply temperature is very network dependent, so it is
            # hard to choose a default. Therefore, we look at the global
            # properties instead and add 10 degrees on top.
            global_supply_temperatures = [
                c["supplyTemperature"]
                for a in assets.values()
                for c in a.global_properties["carriers"].values()
            ]
            max_global_supply = max(x for x in global_supply_temperatures if np.isfinite(x))

            attribute_temperatures = [
                a.attributes.get("maxTemperature", -np.inf) for a in assets.values()
            ]
            max_attribute = max(x for x in attribute_temperatures if np.isfinite(x))

            self.__max_supply_temperature = max(max_global_supply, max_attribute) + 10.0

            self.__model = ESDLQTHModel(assets, **self.esdl_qth_model_options())

        root_logger = logging.getLogger("")

        if self.__run_info is not None and self.__run_info.output_diagnostic_file:
            # Add stream handler if it does not already exist.
            if not logger.hasHandlers() and not any(
                (isinstance(h, logging.StreamHandler) for h in logger.handlers)
            ):
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            # Add pi.DiagHandler. Only add if it doesn't already exist.
            if not any((isinstance(h, pi.DiagHandler) for h in root_logger.handlers)):
                basename = self.__run_info.output_diagnostic_file.stem
                folder = self.__run_info.output_diagnostic_file.parent

                handler = pi.DiagHandler(folder, basename)
                root_logger.addHandler(handler)

        if self.__run_info is not None:
            self._input_timeseries_file = self.__run_info.input_timeseries_file
            self.__output_timeseries_file = self.__run_info.output_timeseries_file

        self._override_pipe_classes = dict()
        self.override_pipe_classes()

        self.name_to_esdl_id_map = dict()

        super().__init__(*args, **kwargs)

    def pre(self):
        super().pre()
        for esdl_id, esdl_asset in self.esdl_assets.items():
            if esdl_asset.name in self.name_to_esdl_id_map:
                raise RuntimeWarning(
                    f"Found multiple ESDL assets with name {esdl_asset.name} in the "
                    f"input ESDL. This is not supported in the optimization."
                )
            self.name_to_esdl_id_map[esdl_asset.name] = esdl_id

    def override_pipe_classes(self):
        maximum_velocity = self.heat_network_options()["maximum_velocity"]

        no_pipe_class = PipeClass("None", 0.0, 0.0, (0.0, 0.0), 0.0)
        pipe_classes = [
            EDRPipeClass.from_edr_class(name, edr_class_name, maximum_velocity)
            for name, edr_class_name in _AssetToComponentBase.STEEL_S1_PIPE_EDR_ASSETS.items()
        ]

        # We assert the pipe classes are monotonically increasing in size
        assert np.all(np.diff([pc.inner_diameter for pc in pipe_classes]) > 0)

        for asset in self.esdl_assets.values():
            if asset.asset_type == "Pipe":
                p = asset.name
                if self.is_cold_pipe(p):
                    continue

                if asset.attributes["state"].name == "OPTIONAL":
                    c = self._override_pipe_classes[p] = []
                    c.append(no_pipe_class)

                    min_size = self.__minimum_pipe_size_name
                    min_size_idx = [
                        idx for idx, pipe in enumerate(pipe_classes) if pipe.name == min_size
                    ]
                    assert len(min_size_idx) == 1
                    min_size_idx = min_size_idx[0]

                    max_size = asset.attributes["diameter"].name

                    max_size_idx = [
                        idx for idx, pipe in enumerate(pipe_classes) if pipe.name == max_size
                    ]
                    assert len(max_size_idx) == 1
                    max_size_idx = max_size_idx[0]

                    if max_size_idx < min_size_idx:
                        logger.warning(
                            f"{p} has an upper DN size smaller than the used minimum size "
                            f"of {self.__minimum_pipe_size_name}, choose at least "
                            f"{self.__minimum_pipe_size_name}"
                        )
                    elif min_size_idx == max_size_idx:
                        c.append(pipe_classes[min_size_idx])
                    else:
                        c.extend(pipe_classes[min_size_idx : max_size_idx + 1])
                elif asset.attributes["state"].name == "DISABLED":
                    c = self._override_pipe_classes[p] = []
                    c.append(no_pipe_class)

    @property
    def esdl_assets(self):
        return self.__esdl_assets

    @property
    def esdl_carriers(self):
        return self.__esdl_carriers

    @property
    def esdl_asset_id_to_name_map(self):
        return self.__timeseries_id_map.copy()

    @property
    def esdl_asset_name_to_id_map(self):
        return dict(zip(self.__timeseries_id_map.values(), self.__timeseries_id_map.keys()))

    def get_asset_from_asset_name(self, asset_name: str) -> esdl.Asset:
        asset_id = self.esdl_asset_name_to_id_map[asset_name]
        return self.esdl_assets[asset_id]

    def esdl_heat_model_options(self) -> Dict:
        heat_network_options = self.heat_network_options()
        v_nominal = heat_network_options["estimated_velocity"]
        v_max = heat_network_options["maximum_velocity"]
        return dict(v_nominal=v_nominal, v_max=v_max)

    def esdl_qth_model_options(self) -> Dict:
        heat_network_options = self.heat_network_options()
        kwargs = {}
        kwargs["v_nominal"] = heat_network_options["estimated_velocity"]
        kwargs["v_max"] = heat_network_options["maximum_velocity"]
        if self.__max_supply_temperature is not None:
            kwargs["maximum_temperature"] = self.__max_supply_temperature
        return dict(**kwargs)

    def is_hot_pipe(self, pipe: str) -> bool:
        return not self.is_cold_pipe(pipe)

    def is_cold_pipe(self, pipe: str) -> bool:
        return pipe.endswith("_ret")

    def hot_to_cold_pipe(self, pipe: str):
        return f"{pipe}_ret"

    def cold_to_hot_pipe(self, pipe: str):
        return pipe[:-4]

    def pycml_model(self):
        return self.__model

    def read(self):
        super().read()

        if self._profiles:
            datetimes = None
            for id, profile in self._profiles.items():
                asset_name = self.esdl_asset_id_to_name_map[id]
                asset = next(a for a in self.esdl_assets.values() if a.name == asset_name)
                ports = []
                ports.extend(asset.in_ports) if asset.in_ports is not None else ports
                ports.extend(asset.out_ports) if asset.out_ports is not None else ports
                if isinstance(ports[0].carrier, esdl.HeatCommodity):
                    commodity = "heat"
                elif isinstance(ports[0].carrier, esdl.ElectricityCommodity):
                    commodity = "electricity"
                elif isinstance(ports[0].carrier, esdl.GasCommodity):
                    commodity = "gas"
                variable = f"{asset_name}.target_{commodity}_demand"
                values = profile.values
                flat_list = []
                for sublist in values:
                    for item in sublist:
                        if not np.isnan(item):
                            flat_list.append(item)
                        else:
                            if len(flat_list) > 0:
                                logger.warning(
                                    f"Found NaN value in profile for {variable},  "
                                    f"using value of previous timestep"
                                )
                                flat_list.append(flat_list[-1])
                            else:
                                logger.error(
                                    f"Found NaN value as first value in profile for "
                                    f"{variable}, using 0 instead"
                                )
                                flat_list.append(0.0)
                if datetimes is None:
                    datetimes = profile.index.tz_convert(None).to_pydatetime()
                elif len(profile.index.tz_convert(None).to_pydatetime()) != len(datetimes):
                    logger.error(f"Unequal profile lengths for {variable}")
                for ensemble_member in range(self.ensemble_size):
                    self.io.set_timeseries(
                        variable, datetimes, flat_list, ensemble_member=ensemble_member
                    )
                if self.io.reference_datetime is None:
                    self.io.reference_datetime = (profile.index.tz_convert(None).to_pydatetime())[0]
        if not hasattr(self, "_input_timeseries_file"):
            demand_assets = [
                asset for asset in self.esdl_assets.values() if asset.asset_type == "HeatingDemand"
            ]
            try:
                datetimes = self.io.datetimes
            except AttributeError:
                logger.warning(
                    f"No profiles provided for demands, could not infer the period over "
                    f"which to optimize, using the period between "
                    f"{DEFAULT_START_TIMESTAMP} and {DEFAULT_END_TIMESTAMP}."
                )
                start_time = datetime.datetime.fromisoformat(DEFAULT_START_TIMESTAMP)
                end_time = datetime.datetime.fromisoformat(DEFAULT_END_TIMESTAMP)
                datetimes = pd.date_range(start=start_time, end=end_time, freq="H").to_pydatetime()
                if self.io.reference_datetime is None:
                    self.io.reference_datetime = start_time
            for demand in demand_assets:
                if self._profiles and demand.name in self._profiles:
                    continue
                logger.warning(
                    f"No demand profile specified for {demand} in an influxdb "
                    f"profile and no file provided with profiles. Using the asset "
                    f"power instead to generate a constant profile, with default "
                    f"asset power of 0.0"
                )
                demand_power = demand.attributes["power"]
                profile = [demand_power] * len(datetimes)
                for ensemble_member in range(self.ensemble_size):
                    self.io.set_timeseries(
                        variable=f"{demand.name}.target_heat_demand",
                        datetimes=datetimes,
                        values=profile,
                        ensemble_member=ensemble_member,
                        check_duplicates=True,
                    )
        else:
            if self._input_timeseries_file is None:
                return

            input_timeseries_file = Path(self._input_timeseries_file)
            assert input_timeseries_file.is_absolute()
            assert input_timeseries_file.suffix == ".xml" or input_timeseries_file.suffix == ".csv"

            if input_timeseries_file.suffix == ".xml":
                self.read_xml(input_timeseries_file)
            elif input_timeseries_file.suffix == ".csv":
                self.read_csv(input_timeseries_file)

    def read_csv(self, input_timeseries_file):
        csv_data = pd.read_csv(input_timeseries_file)
        try:
            timeseries_import_times = [
                datetime.datetime.strptime(entry.replace("Z", ""), "%Y-%m-%d %H:%M:%S")
                for entry in csv_data["DateTime"].to_numpy()
            ]
        except ValueError:
            try:
                timeseries_import_times = [
                    datetime.datetime.strptime(entry.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    for entry in csv_data["DateTime"].to_numpy()
                ]
            except ValueError:
                try:
                    timeseries_import_times = [
                        datetime.datetime.strptime(entry.replace("Z", ""), "%d-%m-%Y %H:%M")
                        for entry in csv_data["DateTime"].to_numpy()
                    ]
                except ValueError:
                    logger.error("Date time string is not in supported format")

        self.io.reference_datetime = timeseries_import_times[0]
        for ensemble_member in range(self.ensemble_size):
            for demand in self.heat_network_components.get("demand", []):
                try:
                    values = csv_data[f"{demand}"].to_numpy()
                    self.io.set_timeseries(
                        demand + ".target_heat_demand",
                        timeseries_import_times,
                        values,
                        ensemble_member,
                    )
                except KeyError:
                    pass
            for source in self.heat_network_components.get("source", []):
                try:
                    values = csv_data[f"{source.replace(' ', '')}"].to_numpy()
                    self.io.set_timeseries(
                        source + ".target_heat_source",
                        timeseries_import_times,
                        values,
                        ensemble_member,
                    )
                except KeyError:
                    pass
            for demand in self.heat_network_components.get("electricity_demand", []):
                try:
                    values = csv_data[
                        f"{demand.replace(' ', '')}.target_electricity_demand"
                    ].to_numpy()
                    self.io.set_timeseries(
                        demand + ".target_electricity_demand",
                        timeseries_import_times,
                        values,
                        ensemble_member,
                    )
                except KeyError:
                    pass
            for source in self.heat_network_components.get("electricity_source", []):
                try:
                    values = csv_data[
                        f"{source.replace(' ', '')}.target_electricity_source"
                    ].to_numpy()
                    self.io.set_timeseries(
                        source + ".target_electricity_source",
                        timeseries_import_times,
                        values,
                        ensemble_member,
                    )
                except KeyError:
                    pass
            for demand in self.heat_network_components.get("gas_demand", []):
                try:
                    values = csv_data[f"{demand.replace(' ', '')}.target_gas_demand"].to_numpy()
                    self.io.set_timeseries(
                        demand + ".target_gas_demand",
                        timeseries_import_times,
                        values,
                        ensemble_member,
                    )
                except KeyError:
                    pass
            for source in self.heat_network_components.get("gas_source", []):
                try:
                    values = csv_data[f"{source.replace(' ', '')}.target_gas_source"].to_numpy()
                    self.io.set_timeseries(
                        source + ".target_gas_source",
                        timeseries_import_times,
                        values,
                        ensemble_member,
                    )
                except KeyError:
                    pass

    def read_xml(self, input_timeseries_file):
        timeseries_import_basename = input_timeseries_file.stem
        input_folder = input_timeseries_file.parent

        try:
            self.__timeseries_import = pi.Timeseries(
                self.esdl_pi_input_data_config(
                    self.__timeseries_id_map, self.heat_network_components.copy()
                ),
                input_folder,
                timeseries_import_basename,
                binary=False,
                pi_validate_times=self.esdl_pi_validate_timeseries,
            )
        except IOError:
            raise Exception(
                "ESDLMixin: {}.xml not found in {}.".format(
                    timeseries_import_basename, input_folder
                )
            )

        # Convert timeseries timestamps to seconds since t0 for internal use
        timeseries_import_times = self.__timeseries_import.times

        # Offer input timeseries to IOMixin
        self.io.reference_datetime = self.__timeseries_import.forecast_datetime

        for ensemble_member in range(self.__timeseries_import.ensemble_size):
            for variable, values in self.__timeseries_import.items(ensemble_member):
                self.io.set_timeseries(variable, timeseries_import_times, values, ensemble_member)

    def write(self):
        super().write()

        if getattr(self, "__output_timeseries_file", None) is None:
            return

        output_timeseries_file = Path(self.__output_timeseries_file)
        assert output_timeseries_file.is_absolute()
        assert output_timeseries_file.suffix == ".xml"

        timeseries_export_basename = output_timeseries_file.stem
        output_folder = output_timeseries_file.parent

        try:
            self.__timeseries_export = pi.Timeseries(
                self.esdl_pi_output_data_config(self.__timeseries_id_map),
                output_folder,
                timeseries_export_basename,
                binary=False,
                pi_validate_times=self.esdl_pi_validate_timeseries,
            )
        except IOError:
            raise Exception(
                "ESDLMixin: {}.xml not found in {}.".format(
                    timeseries_export_basename, output_folder
                )
            )

        # Get time stamps
        times = self.times()
        if len(set(times[1:] - times[:-1])) == 1:
            dt = timedelta(seconds=times[1] - times[0])
        else:
            dt = None

        output_keys = [k for k, v in self.__timeseries_export.items()]

        # Start of write output
        # Write the time range for the export file.
        self.__timeseries_export.times = [
            self.__timeseries_import.times[self.__timeseries_import.forecast_index]
            + timedelta(seconds=s)
            for s in times
        ]

        # Write other time settings
        self.__timeseries_export.forecast_datetime = self.__timeseries_import.forecast_datetime
        self.__timeseries_export.dt = dt
        self.__timeseries_export.timezone = self.__timeseries_import.timezone

        # Write the ensemble properties for the export file.
        if self.ensemble_size > 1:
            self.__timeseries_export.contains_ensemble = True
        self.__timeseries_export.ensemble_size = self.ensemble_size
        self.__timeseries_export.contains_ensemble = self.ensemble_size > 1

        # Start looping over the ensembles for extraction of the output values.
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            # For all variables that are output variables the values are
            # extracted from the results.
            for variable in output_keys:
                try:
                    values = results[variable]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times,
                            self.times(variable),
                            values,
                            self.interpolation_method(variable),
                        )
                except KeyError:
                    try:
                        ts = self.get_timeseries(variable, ensemble_member)
                        if len(ts.times) != len(times):
                            values = self.interpolate(times, ts.times, ts.values)
                        else:
                            values = ts.values
                    except KeyError:
                        logger.warning(
                            "ESDLMixin: Output requested for non-existent variable {}. "
                            "Will not be in output file.".format(variable)
                        )
                        continue

                self.__timeseries_export.set(variable, values, ensemble_member=ensemble_member)

        # Write output file to disk
        self.__timeseries_export.write()


class _ESDLInputDataConfig:
    def __init__(self, id_map, heat_network_components):
        # TODO: change naming source and demand to heat_source and heat_demand throughout code
        self.__id_map = id_map
        self._sources = set(heat_network_components.get("source", []))
        self._demands = set(heat_network_components.get("demand", []))
        self._electricity_sources = set(heat_network_components.get("electricity_source", []))
        self._electricity_demands = set(heat_network_components.get("electricity_demand", []))
        self._gas_sources = set(heat_network_components.get("gas_source", []))
        self._gas_demands = set(heat_network_components.get("gas_demand", []))

    def variable(self, pi_header):
        location_id = pi_header.find("pi:locationId", ns).text

        try:
            component_name = self.__id_map[location_id]
        except KeyError:
            parameter_id = pi_header.find("pi:parameterId", ns).text
            qualifiers = pi_header.findall("pi:qualifierId", ns)
            qualifier_ids = ":".join(q.text for q in qualifiers)
            return f"{location_id}:{parameter_id}:{qualifier_ids}"

        if component_name in self._demands:
            suffix = ".target_heat_demand"
        elif component_name in self._sources:
            suffix = ".target_heat_source"
        elif component_name in self._electricity_demands:
            suffix = ".target_electricity_demand"
        elif component_name in self._electricity_sources:
            suffix = ".target_electricity_source"
        elif component_name in self._gas_demands:
            suffix = ".target_gas_demand"
        elif component_name in self._gas_sources:
            suffix = ".target_gas_source"
        else:
            logger.warning(
                f"Could not identify '{component_name}' as either source or demand. "
                f"Using neutral suffix '.target_heat' for its heat timeseries."
            )
            suffix = ".target_heat"

        # Note that the qualifier id (if any specified) refers to the profile
        # element of the respective ESDL asset->in_port. For now we just
        # assume that only heat demand timeseries are set in the XML file.
        return f"{component_name}{suffix}"

    def pi_variable_ids(self, variable):
        raise NotImplementedError

    def parameter(self, parameter_id, location_id=None, model_id=None):
        raise NotImplementedError

    def pi_parameter_ids(self, parameter):
        raise NotImplementedError


class _ESDLOutputDataConfig:
    def __init__(self, id_map):
        self.__id_map = id_map

    def variable(self, pi_header):
        location_id = pi_header.find("pi:locationId", ns).text
        parameter_id = pi_header.find("pi:parameterId", ns).text

        component_name = self.__id_map[location_id]

        return f"{component_name}.{parameter_id}"

    def pi_variable_ids(self, variable):
        raise NotImplementedError

    def parameter(self, parameter_id, location_id=None, model_id=None):
        raise NotImplementedError

    def pi_parameter_ids(self, parameter):
        raise NotImplementedError


class _RunInfoReader:
    def __init__(self, filepath: Union[str, Path]):
        filepath = Path(filepath).resolve()

        root = ET.parse(filepath).getroot()

        # If the workDir is not absolute, we take it relative to the folder in
        # which the RunInfo file is.
        work_dir = Path(root.findtext("pi:workDir", namespaces=ns))
        if not work_dir.is_absolute():
            work_dir = filepath.parent / work_dir

        self.esdl_file = Path(root.find("pi:properties", ns)[0].attrib["value"])
        if not self.esdl_file.is_absolute():
            self.esdl_file = work_dir / self.esdl_file

        self.parameters_file = root.findtext("pi:inputParameterFile", namespaces=ns)
        if self.parameters_file is not None:
            self.parameters_file = Path(self.parameters_file)
            if not self.parameters_file.is_absolute():
                self.parameters_file = work_dir / self.parameters_file

        try:
            self.input_timeseries_file = Path(root.findall("pi:inputTimeSeriesFile", ns)[0].text)
            if not self.input_timeseries_file.is_absolute():
                self.input_timeseries_file = work_dir / self.input_timeseries_file
        except IndexError:
            self.input_timeseries_file = None

        try:
            self.output_timeseries_file = Path(root.findall("pi:outputTimeSeriesFile", ns)[0].text)
            if not self.output_timeseries_file.is_absolute():
                self.output_timeseries_file = work_dir / self.output_timeseries_file
        except IndexError:
            self.output_timeseries_file = None

        try:
            self.output_diagnostic_file = Path(root.findall("pi:outputDiagnosticFile", ns)[0].text)
            if not self.output_diagnostic_file.is_absolute():
                self.output_diagnostic_file = work_dir / self.output_diagnostic_file
        except IndexError:
            self.output_diagnostic_file = None


def _overwrite_parameters(parameters_file, assets):
    paramroot = ET.parse(parameters_file).getroot()
    groups = paramroot.findall("pi:group", ns)

    for parameter in groups:
        id_ = parameter.attrib["id"]
        param_name = parameter[0].attrib["id"]
        param_value = parameter[0][0].text

        asset = assets[id_]
        type_ = type(asset.attributes[param_name])
        asset.attributes[param_name] = type_(param_value)

    return assets


def _esdl_to_assets(esdl_string, esdl_path: Union[Path, str]):
    if esdl_string is None:
        # correct profile attribute
        esdl.ProfileElement.from_.name = "from"
        setattr(esdl.ProfileElement, "from", esdl.ProfileElement.from_)

        # using esdl as resourceset
        rset_existing = ResourceSet()

        # read esdl energy system
        resource_existing = rset_existing.get_resource(str(esdl_path))
        created_energy_system = resource_existing.contents[0]

        esdl_model = created_energy_system
    else:
        esh = EnergySystemHandler()
        esdl_model = esh.load_from_string(esdl_string)

    # global properties
    global_properties = {}
    global_properties["carriers"] = {}
    id_to_idnumber_map = {}

    for x in esdl_model.energySystemInformation.carriers.carrier.items:
        if isinstance(x, esdl.esdl.HeatCommodity):
            if x.supplyTemperature != 0.0 and x.returnTemperature == 0.0:
                type_ = "supply"
            elif x.returnTemperature != 0.0 and x.supplyTemperature == 0.0:
                type_ = "return"
            else:
                type_ = "none"
            if x.id not in id_to_idnumber_map:
                number_list = [int(s) for s in x.id if s.isdigit()]
                number = ""
                for nr in number_list:
                    number = number + str(nr)
                if type_ == "return":
                    number = number + "000"
                id_to_idnumber_map[x.id] = int(number)

            global_properties["carriers"][x.id] = dict(
                name=x.name.replace("_ret", ""),
                id=x.id,
                id_number_mapping=id_to_idnumber_map[x.id],
                supplyTemperature=x.supplyTemperature,
                returnTemperature=x.returnTemperature,
                __rtc_type=type_,
            )

    # For now, we only support networks with two carries; one hot, one cold.
    # When this no longer holds, carriers either have to specify both the
    # supply and return temperature (instead of one being 0.0), or we have to
    # pair them up.
    if (len(global_properties["carriers"]) % 2) != 0:
        _ESDLInputException(
            "Odd number of carriers specified, please use model with dedicated supply and return "
            "carriers. Every hydraulically coupled system should have one carrier for the supply "
            "side and one for the return side"
        )

    for c in global_properties["carriers"].values():
        supply_temperature = next(
            x["supplyTemperature"]
            for x in global_properties["carriers"].values()
            if x["supplyTemperature"] != 0.0 and x["name"] == c["name"]
        )
        return_temperature = next(
            x["returnTemperature"]
            for x in global_properties["carriers"].values()
            if x["returnTemperature"] != 0.0 and x["name"] == c["name"]
        )
        c["supplyTemperature"] = supply_temperature
        c["returnTemperature"] = return_temperature

    for x in esdl_model.energySystemInformation.carriers.carrier.items:
        if isinstance(x, esdl.esdl.ElectricityCommodity):
            global_properties["carriers"][x.id] = dict(
                name=x.name,
                voltage=x.voltage,
            )

    assets = {}

    # Component ids are unique, but we require component names to be unique as well.
    component_names = set()

    # loop through assets
    for el in esdl_model.eAllContents():
        if isinstance(el, esdl.Asset):
            if hasattr(el, "name") and el.name:
                el_name = el.name
            else:
                el_name = el.id

            if "." in el_name:
                # Dots indicate hierarchy, so would be very confusing
                raise ValueError(f"Dots in component names not supported: '{el_name}'")

            if el_name in component_names:
                raise Exception(f"Asset names have to be unique: '{el_name}' already exists")
            else:
                component_names.add(el_name)

            # For some reason `esdl_element.assetType` is `None`, so use the class name
            asset_type = el.__class__.__name__

            # Every asset should at least have a port to be connected to another asset
            assert len(el.port) >= 1

            in_ports = None
            out_ports = None
            for port in el.port:
                if isinstance(port, esdl.InPort):
                    if in_ports is None:
                        in_ports = [port]
                    else:
                        in_ports.append(port)
                elif isinstance(port, esdl.OutPort):
                    if out_ports is None:
                        out_ports = [port]
                    else:
                        out_ports.append(port)
                else:
                    _ESDLInputException(f"The port for {el_name} is neither an IN or OUT port")

            # Note that e.g. el.__dict__['length'] does not work to get the length of a pipe.
            # We therefore built this dict ourselves using 'dir' and 'getattr'
            attributes = {k: getattr(el, k) for k in dir(el)}
            assets[el.id] = Asset(
                asset_type,
                el.id,
                el_name,
                in_ports,
                out_ports,
                attributes,
                global_properties,
            )
    profiles = parse_esdl_profiles(esdl_model)

    return assets, profiles, global_properties["carriers"]
