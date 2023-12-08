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
from esdl.units.conversion import ENERGY_IN_J, POWER_IN_W

import pandas as pd
import numpy as np

from pathlib import Path


logger = logging.getLogger()

influx_cred_map = {"wu-profiles.esdl-beta.hesi.energy:443": ("warmingup", "warmingup")}


class _ProfileParserException(Exception):
    pass

class BaseProfileReader:
    _energy_system: esdl.EnergySystem
    _file_path: Optional[Path]
    _profiles: Dict[int, Dict[str, np.ndarray]]
    _reference_datetimes = pd.DatetimeIndex
    component_type_to_var_name_map: dict = {
        "demand": ".target_heat_demand",
        "source": ".target_heat_source",
        "electricity_demand": ".target_electricity_demand",
        "electricity_source": ".target_electricity_source",
        "gas_demand": ".target_gas_demand",
        "gas_source": ".target_gas_source",
    }

    def __init__(self, energy_system: esdl.EnergySystem, file_path: Optional[Path]):
        self._profiles = defaultdict(dict)
        self._energy_system = energy_system
        self._file_path = file_path

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

        :param io:                          Datastore in which the profiles will be saved
        :param heat_network_components:     Dictionary of the components of the network, should
                                            contain at least every component for which a profile
                                            needs to be loaded
        :param esdl_asset_id_to_name_map    Dictionary that maps asset ids to asset names,
                                            this is required when reading from an XML
        :param ensemble_size                Integer denoting the size of the set of scenarios to
                                            optimize. Currently only XML inputs support loading a
                                            different profile for different ensemble members
        :param esdl_assets                  Dictionary mapping asset IDs to loaded ESDL assets
        """
        self._load_profiles_from_source(heat_network_components=heat_network_components,
                                        esdl_asset_id_to_name_map=esdl_asset_id_to_name_map,
                                        ensemble_size=ensemble_size)

        io.reference_datetime = self._reference_datetimes[0]

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
        This function should be implemented by the child. It should load the profiles the available
        profiles for demands and sources from the correct source and saves them in the _profiles
        attribute. It should also sets the _reference_datetime_index attribute to the correct
        index to be used in the DataStore when loading the profiles
        """
        raise NotImplementedError


class InfluxDBProfileReader(BaseProfileReader):

    def __init__(self, energy_system: esdl.EnergySystem, file_path: Optional[Path]):
        super().__init__(energy_system=energy_system, file_path=file_path)

    def _load_profiles_from_source(self, heat_network_components: Dict[str, Set[str]],
                                   esdl_asset_id_to_name_map: Dict[str, str],
                                   ensemble_size: int) -> None:
        profiles: Dict[str, pd.DataFrame] = dict()
        logger.info("Caching profiles...")
        error_neighbourhoods = list()
        for profile in [x for x in self._energy_system.eAllContents() if isinstance(x, esdl.InfluxDBProfile)]:
            profile_host = profile.host
            containing_asset_id = profile.eContainer().energyasset.id

            ssl_setting = False
            if "https" in profile_host:
                profile_host = profile_host[8:]
                ssl_setting = True
            elif "http" in profile_host:
                profile_host = profile_host[7:]
            if profile.port == 443:
                ssl_setting = True
            influx_host = "{}:{}".format(profile_host, profile.port)
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

            # TODO: Should raise Exceptions. Also, the start and end time of the profile should be
            #  the same for every profile read and this should be checked.
            # Error check start and end dates of profiles
            if time_series_data.end_datetime != profile.endDate:
                logger.error(
                    f"The user input profile end datetime: {profile.endDate} does not match the end"
                    f" datetime in the database: {time_series_data.end_datetime} for variable: "
                    f"{profile.field}"
                )
                sys.exit(1)
            if time_series_data.start_datetime != profile.startDate:
                logger.error(
                    f"The user input profile start datetime: {profile.startDate} does not match the"
                    f" start date in the database: {time_series_data.start_datetime} for variable: "
                    f"{profile.field}"
                )
                sys.exit(1)
            if time_series_data.start_datetime != time_series_data.profile_data_list[0][0]:
                logger.error(
                    f"The profile's variable value for the start datetime: "
                    f"{time_series_data.start_datetime} does not match the start datetime of the"
                    f" profile data: {time_series_data.profile_data_list[0][0]}"
                )
                sys.exit(1)
            if time_series_data.end_datetime != time_series_data.profile_data_list[-1][0]:
                logger.error(
                    f"The profile's variable value for the end datetime: "
                    f"{time_series_data.end_datetime} does not match the end datetime of the"
                    f" profile data: {time_series_data.profile_data_list[-1][0]}"
                )
                sys.exit(1)

            # Error check: ensure that the profile data has a time resolution of 3600s (1hour) as
            # expected
            for idp in range(len(time_series_data.profile_data_list) - 1):
                time_resolution = (
                        time_series_data.profile_data_list[idp + 1][0]
                        - time_series_data.profile_data_list[idp][0]
                )
                if time_resolution.seconds != 3600:
                    logger.error(
                        f"The time resolution of the profile:{profile.measurement}-{profile.field} is"
                        "not 3600s as expected"
                    )
                    sys.exit(1)

            data_points = {
                t[0].strftime("%Y-%m-%dT%H:%M:%SZ"): t[1] for t in
                time_series_data.profile_data_list
            }
            df = pd.DataFrame.from_dict(data_points, orient="index")
            df.index = pd.to_datetime(df.index, utc=True)

            # TODO add test case. Currently no test case for esdl parsing
            # Convert Power and Energy to standard unit of Watt and Joules
            for idf in range(len(df)):
                if (
                        profile.profileQuantityAndUnit.reference.physicalQuantity
                        == esdl.PhysicalQuantityEnum.POWER
                ):
                    df.iloc[idf] = convert_to_unit(
                        df.iloc[idf], profile.profileQuantityAndUnit, POWER_IN_W
                    )
                elif (
                        profile.profileQuantityAndUnit.reference.physicalQuantity
                        == esdl.PhysicalQuantityEnum.ENERGY
                ):
                    df.iloc[idf] = convert_to_unit(
                        df.iloc[idf], profile.profileQuantityAndUnit, ENERGY_IN_J
                    )
                else:
                    print(
                        f"Current the code only caters for: {esdl.PhysicalQuantityEnum.POWER} & "
                        f"{esdl.PhysicalQuantityEnum.ENERGY}, and it does not cater for "
                        f"{profile.profileQuantityAndUnit.reference.physicalQuantity}"
                    )
                    sys.exit(1)

            profiles[containing_asset_id] = df * profile.multiplier

            if len(error_neighbourhoods) > 0:
                raise RuntimeError(f"Encountered errors loading data for {error_neighbourhoods}")

        self._profiles = profiles


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
    __id_map: Dict[str, str]
    _sources: Set
    _demands: Set
    _electricity_sources: Set
    _electricity_demands: Set
    _gas_sources: Set
    _gas_demands: Set

    def __init__(self, id_map: Dict[str, str], heat_network_components: Dict[str, Set[str]]):
        # TODO: change naming source and demand to heat_source and heat_demand throughout code
        self.__id_map = id_map
        self._sources = set(heat_network_components.get("source", []))
        self._demands = set(heat_network_components.get("demand", []))
        self._electricity_sources = set(heat_network_components.get("electricity_source", []))
        self._electricity_demands = set(heat_network_components.get("electricity_demand", []))
        self._gas_sources = set(heat_network_components.get("gas_source", []))
        self._gas_demands = set(heat_network_components.get("gas_demand", []))

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
