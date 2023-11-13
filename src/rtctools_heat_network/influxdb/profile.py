import logging
import sys
from datetime import timedelta as td

import esdl
from esdl.profiles.influxdbprofilemanager import ConnectionSettings
from esdl.profiles.influxdbprofilemanager import InfluxDBProfileManager
from esdl.units.conversion import ENERGY_IN_J, POWER_IN_W, convert_to_unit


import pandas as pd

logger = logging.getLogger()

data_set = {}
influx_cred_map = {"wu-profiles.esdl-beta.hesi.energy:443": ("warmingup", "warmingup")}
time_step = td(seconds=3600)
time_step_notation = "{}s".format(int(time_step.total_seconds()))


# def get_mw_multiplier(profile: esdl.GenericProfile):
#     ts = time_step.total_seconds()
#     factor = {
#         "ENERGY_IN_WH": 3.6 / (1e3 * ts),
#         "ENERGY_IN_KWH": 3.6 / ts,
#         "ENERGY_IN_MWH": (3.6 * 1e3) / ts,
#         "ENERGY_IN_GWH": (3.6 * 1e6) / ts,
#         "ENERGY_IN_TWH": (3.6 * 1e9) / ts,
#         "ENERGY_IN_PWH": (3.6 * 1e12) / ts,
#         "ENERGY_IN_J": 1 / (1e6 * ts),
#         "ENERGY_IN_KJ": 1 / (1e3 * ts),
#         "ENERGY_IN_MJ": 1 / ts,
#         "ENERGY_IN_GJ": 1e3 / ts,
#         "ENERGY_IN_TJ": 1e6 / ts,
#         "ENERGY_IN_PJ": 1e9 / ts,
#         "POWER_IN_W": 1 / 1e6,
#         "POWER_IN_KW": 1 / 1e3,
#         "POWER_IN_MW": 1,
#         "POWER_IN_GW": 1e3,
#         "POWER_IN_TW": 1e6,
#         "POWER_IN_PW": 1e9,
#     }

#     mult_suffix = {
#         "NONE": "",
#         "KILO": "K",
#         "MEGA": "M",
#         "GIGA": "G",
#         "TERRA": "T",
#         "PETA": "P",
#         "MILLI": "-",
#         "MICRO": "-",
#         "NANO": "-",
#         "PICO": "-",
#     }

#     if profile.profileType is not None and profile.profileType != esdl.ProfileTypeEnum.UNDEFINED:
#         if profile.profileType.name not in factor:
#             print("Unsupported profile type : {}".format(profile.profileType))
#             return 0
#         return factor[profile.profileType.name]

#     if isinstance(profile.profileQuantityAndUnit, esdl.QuantityAndUnitReference):
#         p = profile.profileQuantityAndUnit.reference
#     elif isinstance(profile.profileQuantityAndUnit, esdl.QuantityAndUnitType):
#         p = profile.profileQuantityAndUnit

#     if p is not None:
#         if p.unit == esdl.UnitEnum.JOULE:
#             phy_quantity = "ENERGY_IN_"
#             suffix = "J"
#         elif p.unit == esdl.UnitEnum.WATTHOUR:
#             phy_quantity = "ENERGY_IN_"
#             suffix = "WH"
#         elif p.unit == esdl.UnitEnum.WATT:
#             phy_quantity = "POWER_IN_"
#             suffix = "W"
#         else:
#             print("Unsupported unit : {}".format(p.unit))
#             return 0

#         prefix = mult_suffix[p.multiplier.name]
#         if prefix == "-":
#             print("Unsupported multiplier : {}".format(p.multiplier))
#             return 0

#         return factor["{}{}{}".format(phy_quantity, prefix, suffix)]


def parse_esdl_profiles(es, start_date=None, end_date=None):
    logger.info("Caching profiles...")
    error_neighbourhoods = list()
    for profile in [x for x in es.eAllContents() if isinstance(x, esdl.InfluxDBProfile)]:
        profile_host = profile.host
        containing_asset_id = profile.eContainer().energyasset.id
        # to_mw_multiplier = get_mw_multiplier(profile)

        ssl_setting = False
        if "https" in profile_host:
            profile_host = profile_host[8:]
            ssl_setting = True
        elif "http" in profile_host:
            profile_host = profile_host[7:]
        if profile.port == 443:
            ssl_setting = True
        profile_host = profile_host[8:]
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
        # Error check: ensure that the profile data has a time resolutuon of 3600s (1hour) as
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
            t[0].strftime("%Y-%m-%dT%H:%M:%SZ"): t[1] for t in time_series_data.profile_data_list
        }
        df = pd.DataFrame.from_dict(data_points, orient="index")
        df.index = pd.to_datetime(df.index, utc=True)
        # TODO: remove old code below once test case with profiles have been included
        # data_set[containing_asset_id] = df * profile.multiplier * to_mw_multiplier * 1.0e6

        # TODO add test case. Currently no test case for esdl parsing
        # Convert Power and Energy to standard unit of Watt and Joules
        # Note: at the time of this code below being implemented pyESDL catered for limited
        # power/energy units which is in the process of being expanded
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

        data_set[containing_asset_id] = df * profile.multiplier

        if len(error_neighbourhoods) > 0:
            raise RuntimeError(f"Encountered errors loading data for {error_neighbourhoods}")

    return data_set
