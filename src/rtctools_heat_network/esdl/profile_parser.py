import datetime
import logging
import sys
from typing import Dict, Optional, Set
from collections import defaultdict

from rtctools.data.storage import DataStore
import rtctools.data.pi

from rtctools_heat_network.esdl.common import Asset

import esdl
from esdl.profiles.influxdbprofilemanager import ConnectionSettings
from esdl.profiles.influxdbprofilemanager import InfluxDBProfileManager
from esdl.units.conversion import ENERGY_IN_J, POWER_IN_W, convert_to_unit

import pandas as pd
import numpy as np

from pathlib import Path


logger = logging.getLogger()

influx_cred_map = {"wu-profiles.esdl-beta.hesi.energy:443": ("warmingup", "warmingup")}


class _ProfileParserException(Exception):
    pass

class BaseProfileReader:
    # _energy_system: esdl.EnergySystem
    # _file_path: Optional[Path]
    # _profiles: Dict[int, Dict[str, np.ndarray]]
    # _reference_datetimes: Optional[pd.DatetimeIndex]
    component_type_to_var_name_map: dict = {
        "demand": ".target_heat_demand",
        "source": ".target_heat_source",
        "electricity_demand": ".target_electricity_demand",
        "electricity_source": ".target_electricity_source",
        "gas_demand": ".target_gas_demand",
        "gas_source": ".target_gas_source",
    }

    def __init__(self, energy_system: esdl.EnergySystem, file_path: Optional[Path]):
        self._profiles: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
        self._energy_system: esdl.EnergySystem = energy_system
        self._file_path: Optional[Path] = file_path
        self._reference_datetimes: Optional[pd.DatetimeIndex] = None

    def read_profiles(self, io: DataStore, heat_network_components: Dict[str, Set[str]],
                      esdl_asset_id_to_name_map: Dict[str, str],
                      ensemble_size: int,
                      esdl_assets: Dict[str, Asset]) -> None:
        """
        This function takes a datastore and a dictionary of heat network components and loads a
        profile for each demand and source in the provided heat network components into the
        datastore. If no profile is available the following happens:
        - for sources, no target profile is set
        - for demands a default profile is loaded equal to the power of the demand asset
        Note that at least one profile must be provided to determine the start and end times of the
        optimization horizon.

        Parameters
        ----------
        io : Datastore in which the profiles will be saved
        heat_network_components :   Dictionary of the components of the network, should
                                    contain at least every component for which a profile
                                    needs to be loaded
        esdl_asset_id_to_name_map : Dictionary that maps asset ids to asset names,
                                    this is required when reading from an XML
        ensemble_size :     Integer denoting the size of the set of scenarios to
                            optimize. Currently only XML inputs support loading a
                            different profile for different ensemble members
        esdl_assets : Dictionary mapping asset IDs to loaded ESDL assets

        Returns
        -------
        None
        """
        self._load_profiles_from_source(heat_network_components=heat_network_components,
                                        esdl_asset_id_to_name_map=esdl_asset_id_to_name_map,
                                        ensemble_size=ensemble_size)

        try:
            io.reference_datetime = self._reference_datetimes[0]
        except AttributeError:
            raise RuntimeError(f"No profiles were provided so no timeframe for the profiles "
                               f"could be deduced")

        esdl_asset_names_to_ids = dict(zip(esdl_asset_id_to_name_map.values(),
                                           esdl_asset_id_to_name_map.keys()))

        for ensemble_member in range(ensemble_size):
            for component_type, var_name in self.component_type_to_var_name_map.items():
                for component in heat_network_components.get(component_type, []):
                    profile = self._profiles[ensemble_member].get(component + var_name, None)
                    if profile is not None:
                        values = profile
                    else:
                        if not "demand" in component_type:
                            # We don't set a default profile for source targets
                            continue
                        logger.warning(f"No profile provided for {component=} and "
                                       f"{ensemble_member=}, using the assets power value instead")
                        asset_power = \
                            esdl_assets[esdl_asset_names_to_ids[component]].attributes["power"]
                        values = np.array([asset_power] * len(self._reference_datetimes))

                    io.set_timeseries(
                        variable = component + var_name,
                        datetimes=self._reference_datetimes,
                        values = values,
                        ensemble_member=ensemble_member
                    )


    def _load_profiles_from_source(self, heat_network_components: Dict[str, Set[str]],
                                   esdl_asset_id_to_name_map: Dict[str, str],
                                   ensemble_size: int) -> None:
        """
        This function must be implemented by the child. It must load the available
        profiles for demands and sources from the correct source and saves them in the _profiles
        attribute. It must also set the _reference_datetime_index attribute to the correct
        index to be used in the DataStore when loading the profiles

        Parameters
        ----------
        heat_network_components :   Dictionary of the components of the network, should
                                    contain at least every component for which a profile
                                    needs to be loaded
        esdl_asset_id_to_name_map : Dictionary that maps asset ids to asset names,
                                    this is required when reading from an XML
        ensemble_size :     Integer denoting the size of the set of scenarios to
                            optimize. Currently only XML inputs support loading a
                            different profile for different ensemble members

        Returns
        -------
        None
        """
        raise NotImplementedError


class InfluxDBProfileReader(BaseProfileReader):

    asset_type_to_variable_name_conversion = {
        esdl.esdl.HeatingDemand: ".target_heat_demand",
        esdl.esdl.HeatProducer: ".target_heat_source",
        esdl.esdl.ElectricityDemand: ".target_electricity_demand",
        esdl.esdl.ElectricityProducer: ".target_electricity_source",
        esdl.esdl.GasDemand: ".target_gas_demand",
        esdl.esdl.GasProducer: ".target_gas.source",
    }

    def __init__(self, energy_system: esdl.EnergySystem, file_path: Optional[Path]):
        super().__init__(energy_system=energy_system, file_path=file_path)

    def _load_profiles_from_source(self, heat_network_components: Dict[str, Set[str]],
                                   esdl_asset_id_to_name_map: Dict[str, str],
                                   ensemble_size: int) -> None:
        profiles: Dict[str, np.ndarray] = dict()
        logger.info("Reading profiles from InfluxDB")
        self._reference_datetimes = None
        for profile in [x for x in self._energy_system.eAllContents()
                        if isinstance(x, esdl.InfluxDBProfile)]:
            series = self._load_profile_timeseries_from_database(profile=profile)
            self._check_profile_time_series(profile_time_series=series, profile=profile)
            if self._reference_datetimes is None:
                # TODO: since the previous function ensures it's a date time index, I'm not sure
                #  how to get rid of this type checking warning
                self._reference_datetimes = series.index
            else:
                if not all(series.index == self._reference_datetimes):
                    raise RuntimeError(f"Obtained a profile for asset {profile.field} with a "
                                       f"timeseries index that doesn't match the timeseries of "
                                       f"other assets. Please ensure that the profile that is "
                                       f"specified to be loaded for each asset covers exactly the "
                                       f"same timeseries. "
                                       )
            converted_dataframe = self._convert_profile_to_correct_unit(
                profile_time_series=series, profile=profile)

            asset = profile.eContainer().energyasset
            try:
                variable_suffix = self.asset_type_to_variable_name_conversion[type(asset)]
            except KeyError:
                raise RuntimeError(f"The asset {profile.field} is of type {type(asset)} which is "
                                   f"currently not supported to have a profile to be loaded "
                                   f"from the database.")
            profiles[asset.name + variable_suffix] = converted_dataframe * profile.multiplier

        for idx in range(ensemble_size):
            self._profiles[idx]= profiles.copy()

    @staticmethod
    def _load_profile_timeseries_from_database(profile: esdl.InfluxDBProfile) -> pd.Series:
        """
        Function to load the profiles from an InfluxDB. Returns a timeseries with the data for
        the asset.

        Parameters
        ----------
        profile : Input InfluxDBProfile for the asset in the ESDL for which a profile should be read

        Returns
        -------
        A pandas Series of the profile for the asset.
        """
        profile_host = profile.host

        ssl_setting = False
        if "https" in profile_host:
            profile_host = profile_host[8:]
            ssl_setting = True
        elif "http" in profile_host:
            profile_host = profile_host[7:]
        if profile.port == 443:
            ssl_setting = True
        influx_host = "{}:{}".format(profile_host, profile.port)

        # TODO: remove hard-coded database credentials, should probably be read from a settings file
        if influx_host in influx_cred_map:
            (username, password) = influx_cred_map[influx_host]
        else:
            username = None
            password = None

        conn_settings = ConnectionSettings(
            host=profile.host,
            port=profile.port,
            username=username,
            password=password,
            database=profile.database,
            ssl=ssl_setting,
            verify_ssl=ssl_setting,
        )
        time_series_data = InfluxDBProfileManager(conn_settings)

        time_series_data.load_influxdb(
            '"' + profile.measurement + '"',
            [profile.field],
            profile.startDate,
            profile.endDate,
        )

        for x in time_series_data.profile_data_list:
            if len(x) != 2:
                raise RuntimeError(f"InfluxDB profile currently only supports parsing exactly one "
                                   f"profile for each asset")

        index = pd.DatetimeIndex(data=[x[0] for x in time_series_data.profile_data_list])
        data = [x[1] for x in time_series_data.profile_data_list]

        return pd.Series(data=data, index=index)

    @staticmethod
    def _check_profile_time_series(profile_time_series: pd.Series,
                                   profile: esdl.InfluxDBProfile) -> None:
        """
        Function that checks if the loaded profile matches what was expected

        Parameters
        ----------
        profile_time_series : the pandas Series of the profile obtained for the profile.
        profile : the InfluxDBProfile used to obtain the time series

        Returns
        -------
        None
        """
        if profile_time_series.index[0] != profile.startDate:
            raise RuntimeError(
                f"The user input profile start datetime: {profile.startDate} does not match the"
                f" start date in the database: {profile_time_series.index[0]} for asset: "
                f"{profile.field}"
            )
        if profile_time_series.index[-1] != profile.endDate:
            raise RuntimeError(
                f"The user input profile end datetime: {profile.endDate} does not match the end"
                f" datetime in the database: {profile_time_series.index[-1]} for asset: "
                f"{profile.field}")


        # Error check: ensure that the profile data has a time resolution of 3600s (1hour) as
        # expected
        for d1, d2 in zip(profile_time_series.index, profile_time_series.index[1:]):
            if d2 - d1 != pd.Timedelta(hours=1):
                raise RuntimeError(
                    f"The timestep for variable {profile.field} between {d1} and {d2} isn't "
                    f"exactly 1 hour"
                )

    @staticmethod
    def _convert_profile_to_correct_unit(profile_time_series: pd.Series, profile) -> pd.Series:
        """
        Conversion function to change the values in the provided series to the correct unit

        Parameters
        ----------
        profile_time_series: the time series obtained for the provided profile.
        profile: the profile which was used to obtain the series.

        Returns
        -------
        A pandas Series with the same index as the provided profile_time_series and with all values
        converted to either Watt or Joules, depending on the quantity used in the profile.
        """
        profile_quantity = profile.profileQuantityAndUnit.reference.physicalQuantity
        if profile_quantity == esdl.PhysicalQuantityEnum.POWER:
            target_unit = POWER_IN_W
        elif profile_quantity == esdl.PhysicalQuantityEnum.ENERGY:
            target_unit = ENERGY_IN_J
        else:
            raise RuntimeError(
                f"The user input profile currently only supports loading profiles containing "
                f"either power or energy values, not {profile_quantity}."
            )
        return profile_time_series.apply(func=lambda x: convert_to_unit(
            value=x, source_unit=profile.profileQuantityAndUnit, target_unit=target_unit
        ))


class ProfileReaderFromFile(BaseProfileReader):

    def __init__(self, energy_system: esdl.EnergySystem, file_path: Path):
        super().__init__(energy_system=energy_system, file_path=file_path)

    def _load_profiles_from_source(self, heat_network_components: Dict[str, Set[str]],
                                   esdl_asset_id_to_name_map: Dict[str, str],
                                   ensemble_size: int) -> None:
        if self._file_path.suffix == ".xml":
            self._load_xml(heat_network_components=heat_network_components,
                                  esdl_asset_id_to_name_map=esdl_asset_id_to_name_map)
        elif self._file_path.suffix == ".csv":
            self._load_csv(heat_network_components=heat_network_components,
                                  ensemble_size=ensemble_size)
        else:
            raise _ProfileParserException(f"Unsupported profile file extension "
                                          f"{self._file_path.suffix}")

    def _load_csv(self, heat_network_components: Dict[str, Set[str]], ensemble_size: int):
        data = pd.read_csv(self._file_path)
        try:
            timeseries_import_times = [
                datetime.datetime.strptime(entry.replace("Z", ""), "%Y-%m-%d %H:%M:%S")
                for entry in data["DateTime"].to_numpy()
            ]
        except ValueError:
            try:
                timeseries_import_times = [
                    datetime.datetime.strptime(entry.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    for entry in data["DateTime"].to_numpy()
                ]
            except ValueError:
                try:
                    timeseries_import_times = [
                        datetime.datetime.strptime(entry.replace("Z", ""), "%d-%m-%Y %H:%M")
                        for entry in data["DateTime"].to_numpy()
                    ]
                except ValueError:
                    raise _ProfileParserException("Date time string is not in a supported format")

        self._reference_datetimes = timeseries_import_times
        for ensemble_member in range(ensemble_size):
            for component_type, var_name in self.component_type_to_var_name_map.items():
                for component_name in heat_network_components.get(component_type, []):
                    try:
                        values = data[f"{component_name.replace(' ', '')}"].to_numpy()
                    except KeyError:
                        pass
                    else:
                        self._profiles[ensemble_member][component_name + var_name] = values

    def _load_xml(self, heat_network_components, esdl_asset_id_to_name_map):
        timeseries_import_basename = self._file_path.stem
        input_folder = self._file_path.parent

        try:
            data = rtctools.data.pi.Timeseries(
                _ESDLInputDataConfig(
                    esdl_asset_id_to_name_map, heat_network_components
                ),
                input_folder,
                timeseries_import_basename,
                binary=False,
                pi_validate_times=False,
            )
        except IOError:
            raise Exception(
                "ESDLMixin: {}.xml not found in {}.".format(
                    timeseries_import_basename, input_folder
                )
            )

        # Convert timeseries timestamps to seconds since t0 for internal use
        self._reference_datetimes = data.times

        # Offer input timeseries to IOMixin
        for ensemble_member in range(data.ensemble_size):
            for variable, values in data.items(ensemble_member):
                self._profiles[ensemble_member][variable] = values


class _ESDLInputDataConfig:
    ns: dict = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}
    # __id_map: Dict[str, str]
    # _sources: Set
    # _demands: Set
    # _electricity_sources: Set
    # _electricity_demands: Set
    # _gas_sources: Set
    # _gas_demands: Set

    def __init__(self, id_map: Dict[str, str], heat_network_components: Dict[str, Set[str]]):
        # TODO: change naming source and demand to heat_source and heat_demand throughout code
        self.__id_map: Dict[str, str] = id_map
        self._sources: Set = set(heat_network_components.get("source", []))
        self._demands: Set = set(heat_network_components.get("demand", []))
        self._electricity_sources: Set = set(heat_network_components.get("electricity_source", []))
        self._electricity_demands: Set = set(heat_network_components.get("electricity_demand", []))
        self._gas_sources: Set = set(heat_network_components.get("gas_source", []))
        self._gas_demands: Set = set(heat_network_components.get("gas_demand", []))

    def variable(self, pi_header):
        location_id = pi_header.find("pi:locationId", self.ns).text

        try:
            component_name = self.__id_map[location_id]
        except KeyError:
            parameter_id = pi_header.find("pi:parameterId", self.ns).text
            qualifiers = pi_header.findall("pi:qualifierId", self.ns)
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
