import logging
from datetime import timedelta as td

import esdl

from influxdb import InfluxDBClient

import pandas as pd

import requests.exceptions


logger = logging.getLogger()


data_set = {}
influx_cred_map = {"wu-profiles.esdl-beta.hesi.energy:443": ("warmingup", "warmingup")}
time_step = td(seconds=3600)
time_step_notation = "{}s".format(int(time_step.total_seconds()))


def get_mw_multiplier(profile: esdl.GenericProfile):
    ts = time_step.total_seconds()
    factor = {
        "ENERGY_IN_WH": 3.6 / (1e3 * ts),
        "ENERGY_IN_KWH": 3.6 / ts,
        "ENERGY_IN_MWH": (3.6 * 1e3) / ts,
        "ENERGY_IN_GWH": (3.6 * 1e6) / ts,
        "ENERGY_IN_TWH": (3.6 * 1e9) / ts,
        "ENERGY_IN_PWH": (3.6 * 1e12) / ts,
        "ENERGY_IN_J": 1 / (1e6 * ts),
        "ENERGY_IN_KJ": 1 / (1e3 * ts),
        "ENERGY_IN_MJ": 1 / ts,
        "ENERGY_IN_GJ": 1e3 / ts,
        "ENERGY_IN_TJ": 1e6 / ts,
        "ENERGY_IN_PJ": 1e9 / ts,
        "POWER_IN_W": 1 / 1e6,
        "POWER_IN_KW": 1 / 1e3,
        "POWER_IN_MW": 1,
        "POWER_IN_GW": 1e3,
        "POWER_IN_TW": 1e6,
        "POWER_IN_PW": 1e9,
    }

    mult_suffix = {
        "NONE": "",
        "KILO": "K",
        "MEGA": "M",
        "GIGA": "G",
        "TERRA": "T",
        "PETA": "P",
        "MILLI": "-",
        "MICRO": "-",
        "NANO": "-",
        "PICO": "-",
    }

    if profile.profileType is not None and profile.profileType != esdl.ProfileTypeEnum.UNDEFINED:
        if profile.profileType.name not in factor:
            print("Unsupported profile type : {}".format(profile.profileType))
            return 0
        return factor[profile.profileType.name]

    if isinstance(profile.profileQuantityAndUnit, esdl.QuantityAndUnitReference):
        p = profile.profileQuantityAndUnit.reference
    elif isinstance(profile.profileQuantityAndUnit, esdl.QuantityAndUnitType):
        p = profile.profileQuantityAndUnit

    if p is not None:
        if p.unit == esdl.UnitEnum.JOULE:
            phy_quantity = "ENERGY_IN_"
            suffix = "J"
        elif p.unit == esdl.UnitEnum.WATTHOUR:
            phy_quantity = "ENERGY_IN_"
            suffix = "WH"
        elif p.unit == esdl.UnitEnum.WATT:
            phy_quantity = "POWER_IN_"
            suffix = "W"
        else:
            print("Unsupported unit : {}".format(p.unit))
            return 0

        prefix = mult_suffix[p.multiplier.name]
        if prefix == "-":
            print("Unsupported multiplier : {}".format(p.multiplier))
            return 0

        return factor["{}{}{}".format(phy_quantity, prefix, suffix)]


def parse_esdl_profiles(es, start_date=None, end_date=None):
    logger.info("Caching profiles...")
    error_neighbourhoods = list()
    for profile in [x for x in es.eAllContents() if isinstance(x, esdl.InfluxDBProfile)]:
        # if profile.eContainer().carrier.id == self.carrier_id:
        profile_host = profile.host
        containing_asset_id = profile.eContainer().energyasset.id
        to_mw_multiplier = get_mw_multiplier(profile)
        # print('Processing profile of {}'.format(containing_asset_id))
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
        client = InfluxDBClient(
            host=profile_host,
            port=profile.port,
            username=username,
            password=password,
            ssl=ssl_setting,
            verify_ssl=ssl_setting,
            timeout=1,
        )
        client.switch_database(profile.database)
        # TODO: check these timebounds how they work.
        # if profile.startDate is not None:
        start_date = profile.startDate.isoformat()
        start_date = start_date if start_date.endswith("+00:00") else start_date + "+00:00"
        # else:
        #     start_date = start_date.isoformat()
        # if profile.endDate is not None:
        end_date_suffix = " AND time <= '{}'".format(profile.endDate.isoformat())
        end_date_suffix = (
            end_date_suffix
            if end_date_suffix.endswith("+00:00'")
            else end_date_suffix[:-1] + "+00:00'"
        )
        # else:
        #     end_date_suffix = ' AND time <= \'{}\''.format(end_date.isoformat())
        if profile.filters is not None and profile.filters != "":
            filter_suffix = " AND {}".format(profile.filters)
        else:
            filter_suffix = ""

        query = 'SELECT MEAN("{}") FROM "{}" WHERE time >= \'{}\'{}{} GROUP BY time({})'.format(
            profile.field,
            profile.measurement,
            start_date,
            end_date_suffix,
            filter_suffix,
            time_step_notation,
        )
        logger.debug("Executing query...\n{}".format(query))
        try:
            data = client.query(query=query)
        except requests.exceptions.ConnectTimeout:
            error_neighbourhoods.append(profile.field)
            logger.error(
                f"Timeout on query for {profile.field}, probably because either the host "
                f"{profile_host} can't be reached or the {profile.port} is wrong."
            )
        data_points = {t[0]: t[1] for t in data.raw["series"][0]["values"]}
        df = pd.DataFrame.from_dict(data_points, orient="index")
        df.index = pd.to_datetime(df.index, utc=True)
        data_set[containing_asset_id] = df * profile.multiplier * to_mw_multiplier * 1.0e6
        client.close()

        if len(error_neighbourhoods) > 0:
            raise RuntimeError(f"Encountered errors loading data for {error_neighbourhoods}")
    return data_set
