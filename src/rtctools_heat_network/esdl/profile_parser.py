import logging
import sys
from typing import Dict, Optional

from rtctools.data.storage import DataStore

import esdl
from esdl.profiles.influxdbprofilemanager import ConnectionSettings
from esdl.profiles.influxdbprofilemanager import InfluxDBProfileManager
from esdl.units.conversion import ENERGY_IN_J, POWER_IN_W, convert_to_unit

import pandas as pd
import numpy as np

from pathlib import Path


logger = logging.getLogger()

influx_cred_map = {"wu-profiles.esdl-beta.hesi.energy:443": ("warmingup", "warmingup")}

class BaseProfileReader:
    _energy_system: esdl.EnergySystem
    _file_path: Optional[Path]
    _profiles: Dict[str, np.ndarray]
    _reference_datetime_index = pd.DatetimeIndex

    def __init__(self, energy_system: esdl.EnergySystem, file_path: Optional[Path]):
        self._profiles = dict()
        self._energy_system = energy_system
        self._file_path = file_path

    def read_profiles(self, io: DataStore, heat_network_components: Dict[str, str]) -> None:
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
        """
        self._load_profiles_from_source()
        # TODO: implement magic that actually sets the loaded profiles in the store...
        pass

    def _load_profiles_from_source(self) -> None:
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

    def _load_profiles_from_source(self) -> None:
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

