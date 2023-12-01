import datetime
import logging
import xml.etree.ElementTree as ET  # noqa: N817
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import esdl

import numpy as np

import pandas as pd

import rtctools.data.pi as pi
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.io_mixin import IOMixin

from rtctools_heat_network.esdl.asset_to_component_base import _AssetToComponentBase
from rtctools_heat_network.esdl.edr_pipe_class import EDRPipeClass
from rtctools_heat_network.esdl.profile_parser import BaseProfileReader, InfluxDBProfileReader
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.modelica_component_type_mixin import (
    ModelicaComponentTypeMixin,
)
from rtctools_heat_network.pipe_class import PipeClass
from rtctools_heat_network.pycml.pycml_mixin import PyCMLMixin
from rtctools_heat_network.qth_not_maintained.qth_mixin import QTHMixin

from .esdl_heat_model import ESDLHeatModel
from .esdl_model_base import _ESDLModelBase
from .esdl_qth_model import ESDLQTHModel
from .esdl_parser import ESDLStringParser

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
    """
    This class is used to be able to parse an ESDL file and utilize the definition of the energy
    system in that file. Furthermore, it contains functionality to extract profiles specified like
    for example demand profiles.
    """

    esdl_run_info_path: Path = None

    esdl_pi_validate_timeseries: bool = False

    __max_supply_temperature: Optional[float] = None

    # TODO: remove this once ESDL allows specifying a minimum pipe size for an optional pipe.
    __minimum_pipe_size_name: str = "DN150"

    _profile_reader: BaseProfileReader

    def __init__(self, *args, **kwargs) -> None:
        """
        In this __init__ function we do the parsing of the esdl file based on either a string which
        is provided or read it in from a provided file name.

        We put the assets, profiles and carriers in attributes of the class to later instantiate
        the PyCML objects and write the desired time-series.

        We set file locations for the input files and for the diagnostic file.

        We create a dict with all possible pipe classes for the optional pipes to later add them
        to the optimization problem. This is done in this Mixin as we here use the information of
        the EDR database which is linked to ESDL and the Mapeditor.

        Parameters
        ----------
        args : none
        kwargs : esdl_string or esdl_file_name must be provided
        """

        esdl_parser_class = kwargs.get("esdl_parser", ESDLStringParser)
        esdl_string = kwargs.get("esdl_string", None)
        molder_folder = kwargs.get("model_folder")
        esdl_file_name = kwargs.get("esdl_file_name", None)
        esdl_path = None
        if esdl_file_name is not None:
            esdl_path = Path(molder_folder) / esdl_file_name

        # TODO: discuss if this is correctly located here and why the reading of profiles is then
        #  in the read function?
        esdl_parser = esdl_parser_class(esdl_string=esdl_string, esdl_path=esdl_path)
        self.__esdl_assets = esdl_parser.get_assets()
        self.__esdl_carriers = esdl_parser.get_carrier_properties()
        self.__esdl_model = esdl_parser.get_esdl_model()

        profile_reader_class = kwargs.get("profile_reader", InfluxDBProfileReader)
        input_file_path = kwargs.get("input_timeseries_file", None)
        self._profile_reader = profile_reader_class(
            energy_system=self.__esdl_model, file_path=input_file_path)

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
                c["temperature"]
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

        self._override_pipe_classes = dict()
        self.override_pipe_classes()

        self.name_to_esdl_id_map = dict()

        super().__init__(*args, **kwargs)

    def pre(self) -> None:
        """
        In this pre method we create a dict with a mapping between the esdl id and the name. We
        also check that every asset has an unique name, which is needed for us to create unique
        variable names.

        Returns
        -------
        None
        """
        super().pre()
        for esdl_id, esdl_asset in self.esdl_assets.items():
            if esdl_asset.name in self.name_to_esdl_id_map:
                raise RuntimeWarning(
                    f"Found multiple ESDL assets with name {esdl_asset.name} in the "
                    f"input ESDL. This is not supported in the optimization."
                )
            self.name_to_esdl_id_map[esdl_asset.name] = esdl_id

    def override_pipe_classes(self) -> None:
        """
        In this method we populate the _override_pipe_classes dict, which gives a list of possible
        pipe classes for every pipe. We do this only when a pipe has the state OPTIONAL. We use the
        EDR pipe classes. We assume that it is possible to remove a pipe PipeClass None, but also
        that there is a minimum layed pipe size of DN150 to limit the search space. This seems
        reasonable as we focus upon regional and primary networks.

        Returns
        -------
        None
        """
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
    def esdl_assets(self) -> Dict:
        """
        property method to retrieve the esdl assets which are a private attribute of the class.

        Returns
        -------
        A dict of the esdl assets with their properties
        """
        return self.__esdl_assets

    @property
    def esdl_carriers(self) -> Dict:
        """
        property method to retrieve the esdl carriers which are a private attribute of the class.

        Returns
        -------
        A dict with the id of the carrier and the attributes in the value
        """
        return self.__esdl_carriers

    @property
    def esdl_asset_id_to_name_map(self) -> Dict:
        """
        A map between the id and the name of an asset. Very bad naming of the attribute...

        Returns
        -------
        A dict with the id to name map.
        """
        return self.__timeseries_id_map.copy()

    @property
    def esdl_asset_name_to_id_map(self) -> Dict:
        """
        A map between the name and the id of an asset. Very bad naming of the attribute...

        Returns
        -------
        A dict with the name to id map.
        """
        return dict(zip(self.__timeseries_id_map.values(), self.__timeseries_id_map.keys()))

    def get_asset_from_asset_name(self, asset_name: str) -> esdl.Asset:
        """
        This function returns the esdl asset with its properties based on the name you provide

        Parameters
        ----------
        asset_name : string with the asset name of the esdl asset.

        Returns
        -------
        The esdl asset with its attributes and global properties
        """

        asset_id = self.esdl_asset_name_to_id_map[asset_name]
        return self.esdl_assets[asset_id]

    def esdl_heat_model_options(self) -> Dict:
        """
        function to spedifically return the needed HeatMixin options needed for the conversion
        from ESDL to pycml. This case velocities used to set nominals and caps on the heat.

        Returns
        -------
        dict with estimated and maximum velocity
        """
        heat_network_options = self.heat_network_options()
        v_nominal = heat_network_options["estimated_velocity"]
        v_max = heat_network_options["maximum_velocity"]
        return dict(v_nominal=v_nominal, v_max=v_max)

    def esdl_qth_model_options(self) -> Dict:
        """
        function to spedifically return the needed HeatMixin options needed for the conversion
        from ESDL to pycml. This case velocities used to set nominals and caps on the heat.

        Returns
        -------
        dict with estimated and maximum velocity
        """
        heat_network_options = self.heat_network_options()
        kwargs = {}
        kwargs["v_nominal"] = heat_network_options["estimated_velocity"]
        kwargs["v_max"] = heat_network_options["maximum_velocity"]
        if self.__max_supply_temperature is not None:
            kwargs["maximum_temperature"] = self.__max_supply_temperature
        return dict(**kwargs)

    def is_hot_pipe(self, pipe: str) -> bool:
        """
        To check if a pipe is part of the "supply" network.

        Parameters
        ----------
        pipe : string with name of the pipe

        Returns
        -------
        Returns true if the pipe is in the supply network thus not ends with "_ret"
        """
        return True if pipe not in self.cold_pipes else False

    def is_cold_pipe(self, pipe: str) -> bool:
        """
        To check if a pipe is part of the "return" network. Note we only assign to the return
        network if it has a dedicated hot pipe.

        Parameters
        ----------
        pipe : string with name of the pipe

        Returns
        -------
        Returns true if the pipe is in the return network thus ends with "_ret"
        """
        return pipe.endswith("_ret")

    def hot_to_cold_pipe(self, pipe: str) -> str:
        """
        To get the name of the respective cold pipe. Note hot pipes do not automatically have a
        dedicated return pipe in case of different supply and return topologies and/or temperature
        cascading. This function should only be called if the cold pipe exists.

        Parameters
        ----------
        pipe : string with hot pipe name.

        Returns
        -------
        string with the associated return pipe name.
        """
        return f"{pipe}_ret"

    def cold_to_hot_pipe(self, pipe: str) -> str:
        """
        To get the name of the respective hot pipe. Note hot pipes do not automatically have a
        dedicated return pipe in case of different supply and return topologies and/or temperature
        cascading.

        Parameters
        ----------
        pipe : string with cold pipe name.

        Returns
        -------
        string with the associated hot pipe name.
        """
        return pipe[:-4]

    def pycml_model(self) -> _ESDLModelBase:
        """
        Function to get the model description.

        Returns
        -------
        Returns the pycml model object.
        """
        return self.__model

    def read(self) -> None:
        """
        In this read function we read the relevant time-series and write them to the io object for
        later use. We read and write the demand and production profiles. These profiles can either
        be specified in the esdl file referring to an InfluxDB profile, or be specified in a csv
        file in this case we rely on the user to give the csv file in the runinfo.xml.

        Returns
        -------
        None
        """
        # TODO: fix this whole function and fix docstring
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
        elif not hasattr(self, "_input_timeseries_file"):
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
                if self._profiles and self.esdl_asset_name_to_id_map[demand.name] in self._profiles:
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

    def read_csv(self, input_timeseries_file: str) -> None:
        """
        This function reads profiles from a csv and writes them to the io attribute for later use.
        We assume that the csv file has a column with "DateTime" header name in which the datetimes
        are specified for the demand. For most practical workflows we assume hourly resolution data.

        Parameters
        ----------
        input_timeseries_file : str of filepath of the csv

        Returns
        -------
        None
        """
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

    def read_xml(self, input_timeseries_file: str) -> None:
        """
        This function reads profiles from a xml and writes them to the io attribute for later use.
        This method still works but is no longer maintained.

        Parameters
        ----------
        input_timeseries_file : str of filepath of the xml

        Returns
        -------
        None
        """
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

    def write(self) -> None:
        """
        This function comes from legacy with CF in the WarmingUP time. It was used to write out a
        xml file with in that the timeseries output of some specified types of assets. This method
        works but is no longer maintained.

        Returns
        -------
        None
        """
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


# class _ESDLInputDataConfig:
#     def __init__(self, id_map, heat_network_components):
#         # TODO: change naming source and demand to heat_source and heat_demand throughout code
#         self.__id_map = id_map
#         self._sources = set(heat_network_components.get("source", []))
#         self._demands = set(heat_network_components.get("demand", []))
#         self._electricity_sources = set(heat_network_components.get("electricity_source", []))
#         self._electricity_demands = set(heat_network_components.get("electricity_demand", []))
#         self._gas_sources = set(heat_network_components.get("gas_source", []))
#         self._gas_demands = set(heat_network_components.get("gas_demand", []))
#
#     def variable(self, pi_header):
#         location_id = pi_header.find("pi:locationId", ns).text
#
#         try:
#             component_name = self.__id_map[location_id]
#         except KeyError:
#             parameter_id = pi_header.find("pi:parameterId", ns).text
#             qualifiers = pi_header.findall("pi:qualifierId", ns)
#             qualifier_ids = ":".join(q.text for q in qualifiers)
#             return f"{location_id}:{parameter_id}:{qualifier_ids}"
#
#         if component_name in self._demands:
#             suffix = ".target_heat_demand"
#         elif component_name in self._sources:
#             suffix = ".target_heat_source"
#         elif component_name in self._electricity_demands:
#             suffix = ".target_electricity_demand"
#         elif component_name in self._electricity_sources:
#             suffix = ".target_electricity_source"
#         elif component_name in self._gas_demands:
#             suffix = ".target_gas_demand"
#         elif component_name in self._gas_sources:
#             suffix = ".target_gas_source"
#         else:
#             logger.warning(
#                 f"Could not identify '{component_name}' as either source or demand. "
#                 f"Using neutral suffix '.target_heat' for its heat timeseries."
#             )
#             suffix = ".target_heat"
#
#         # Note that the qualifier id (if any specified) refers to the profile
#         # element of the respective ESDL asset->in_port. For now we just
#         # assume that only heat demand timeseries are set in the XML file.
#         return f"{component_name}{suffix}"
#
#     def pi_variable_ids(self, variable):
#         raise NotImplementedError
#
#     def parameter(self, parameter_id, location_id=None, model_id=None):
#         raise NotImplementedError
#
#     def pi_parameter_ids(self, parameter):
#         raise NotImplementedError
#
#
# class _ESDLOutputDataConfig:
#     def __init__(self, id_map):
#         self.__id_map = id_map
#
#     def variable(self, pi_header):
#         location_id = pi_header.find("pi:locationId", ns).text
#         parameter_id = pi_header.find("pi:parameterId", ns).text
#
#         component_name = self.__id_map[location_id]
#
#         return f"{component_name}.{parameter_id}"
#
#     def pi_variable_ids(self, variable):
#         raise NotImplementedError
#
#     def parameter(self, parameter_id, location_id=None, model_id=None):
#         raise NotImplementedError
#
#     def pi_parameter_ids(self, parameter):
#         raise NotImplementedError
#
#
class _ESDLInputDataConfig:
    """
    This class is used to specify naming standard for input data, specifically for demand and
    production profiles.
    """

    def __init__(self, id_map: Dict, heat_network_components: Dict) -> None:
        # TODO: change naming source and demand to heat_source and heat_demand throughout code
        self.__id_map = id_map
        self._sources = set(heat_network_components.get("source", []))
        self._demands = set(heat_network_components.get("demand", []))
        self._electricity_sources = set(heat_network_components.get("electricity_source", []))
        self._electricity_demands = set(heat_network_components.get("electricity_demand", []))
        self._gas_sources = set(heat_network_components.get("gas_source", []))
        self._gas_demands = set(heat_network_components.get("gas_demand", []))

    def variable(self, pi_header: Any) -> str:
        """
        Old function not maintained anymore from WarmingUp times. The input xml file would specify
        the id of the asset for which a time-series was given. This function would return the name
        we use in our framework for that same variable.

        Parameters
        ----------
        pi_header : the xml header element in which the id of the asset is specified

        Returns
        -------
        string with the name of the timeseries name.
        """
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
