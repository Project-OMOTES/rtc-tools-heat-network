import json

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

WATT_TO_MEGA_WATT = 1.0e6
WATT_TO_KILO_WATT = 1.0e3

profile_year = pd.read_csv(
    "C:\\Users\\rojerj\\Documents\\Git\\RTC-tools\\mpc-development\\"
    "examples\\GROW\\to_mpc\\Warmteprofielen_1_year_v3.csv"
)
daily_avg_demands = np.array([0.0] * 365)
for column in profile_year:
    if column != "GMT+1":
        for i in range(365):
            temp = np.asarray(profile_year[column].iloc[(i * 24) : (i * 24) + 24].values)
            daily_avg_demands[i] += np.mean(temp) / WATT_TO_KILO_WATT * WATT_TO_MEGA_WATT
            # temporary hack for missing neighbourhoods
            if column == "HeatingDemand_Artiestenbuurt":
                temp = np.asarray(profile_year[column].iloc[(i * 24) : (i * 24) + 24].values)
                daily_avg_demands[i] += (
                    np.mean(temp) / WATT_TO_KILO_WATT * WATT_TO_MEGA_WATT / 8.76 * 3 * 7.0
                )

demands = [
    "HeatingDemand_Kleurenbuurt",
    "HeatingDemand_Artiestbuurt_2",
    "HeatingDemand_Artiestenbuurt_1",
    "HeatingDemand_Muziekbuurt_1",
    "HeatingDemand_Muziekbuurt_2",
    "HeatingDemand_Muziekbuurt_3",
    "HeatingDemand_Presidentenbuurt",
    "HeatingDemand_DeStrijp_2",
    "HeatingDemand_DeStrijp_3",
    "HeatingDemand_DeStrijp_1",
    "HeatingDemand_Ministerbuurt_1",
    "HeatingDemand_Ministerbuurt_2",
    "HeatingDemand_Stervoorde_1",
    "HeatingDemand_Stervoorde_2",
    "HeatingDemand_Stervoorde_3",
    "HeatingDemand_Stervoorde_4",
    "HeatingDemand_Stervoorde_5",
    "HeatingDemand_Hoekpolder",
    "HeatingDemand_HuisTeLande",
    "HeatingDemand_Stationskwartier",
    "HeatingDemand_TeWerve_1",
    "HeatingDemand_TeWerve_2",
    "HeatingDemand_TeWerve_3",
    "HeatingDemand_TeWerve_4",
    "HeatingDemand_Rembrandtkwartier_1",
    "HeatingDemand_fea2",
    "HeatingDemand_2bbd",
    "HeatingDemand_Rembrandtkwartier_2",
    "HeatingDemand_WelGelegen",
]

decentral_storage = [0]  # , 100, 200, 500]
ATES_options = [0]  # , 20, 30, 50]

LCOEs = []
CO2s = []
OPEXs = []
storages = []
sourcess = []
pipess = []
tank_size_average = []
TCOs = []
source_stategy = []

for ates_power in ATES_options:
    LCOEs.append([])
    CO2s.append([])
    OPEXs.append([])
    storages.append([])
    sourcess.append([])
    pipess.append([])
    tank_size_average.append([])
    TCOs.append([])
    source_stategy.append([])
    for tank_size in decentral_storage:
        # filename = (
        #     "C:\\Users\\rojerj\\Documents\\Git\\RTC-tools\\mpc-development\\examples"
        #     "\\GROW\\model\\results.json"
        # )
        filename = (
            f"C:\\Users\\rojerj\\Documents\\Git\\RTC-tools\\mpc-development\\examples"
            f"\\GROW\\{tank_size}new\\{ates_power}mw\\results.json"
        )
        filename_pars = (
            "C:\\Users\\rojerj\\Documents\\Git\\RTC-tools\\mpc-development\\examples"
            "\\GROW\\pntpc\\parameters.json"
        )

        with open(filename) as json_file:
            results = json.load(json_file)

        with open(filename_pars) as json_file:
            pars = json.load(json_file)

        tank_sizes = [results[f"HeatStorage{i}__max_size"] for i in range(1, 28)]
        tank_placement = [results[f"HeatStorage{i}__buffer_placed"] for i in range(1, 28)]
        total_tank = np.sum(tank_sizes) / (998 * 4200 * 30) * 0.69 + np.sum(tank_placement) * 0.3
        peak_sizes = [results[f"HeatProducer{i}__max_size"] for i in range(1, 28)]
        peak_placement = [results[f"HeatProducer{i}__source_placed"] for i in range(1, 28)]
        total_peak = np.sum(peak_sizes) / 1.0e6 * 100 + np.sum(peak_placement) * 300
        WL = results["WarmteLink__max_size"] / 10**6 * 300 + 300
        GEO = results["GeoSource__number_of_wells"] * 2.0e4 * 1.800
        storage = (total_tank + results["ATES__number_of_doublets"] * 2000) / 1.0e3
        sources = (GEO + WL + total_peak) / 1.0e3

        pipes = list()
        dn = list()
        lengths = list()
        count = 0
        for i in range(1, 118):
            check1 = False
            check2 = False
            try:
                pipes.append(results[f"Pipe{i}__hn_cost"] * pars[f"Pipe{i}.length"])
                dn.append(results[f"Pipe{i}__hn_diameter"] * pars[f"Pipe{i}.length"])
                lengths.append(pars[f"Pipe{i}.length"])
            except KeyError:
                check1 = True
            try:
                pipes.append(results[f"Pipe{i}_a__hn_cost"] * pars[f"Pipe{i}_a.length"])
                pipes.append(results[f"Pipe{i}_b__hn_cost"] * pars[f"Pipe{i}_b.length"])
                dn.append(results[f"Pipe{i}_a__hn_diameter"] * pars[f"Pipe{i}_a.length"])
                dn.append(results[f"Pipe{i}_b__hn_diameter"] * pars[f"Pipe{i}_b.length"])
                lengths.append(pars[f"Pipe{i}_a.length"])
                lengths.append(pars[f"Pipe{i}_b.length"])
            except KeyError:
                check2 = True
            if check1 and check2:
                count += 1

        pipes_total = np.sum(pipes) / 1.0e6 * 2.0
        dn_avg = np.sum(dn) / np.sum(lengths)

        OPEX_production = 0.0
        OPEX_fixed = 0.0
        total_prod = 0.0
        prod = []
        WL_prod = 0.0
        geo_prod = 0.0
        peak_prod = 0.0
        CO2 = 0.0
        elecco2 = 132.43
        for i in range(365):
            prod.append(0)
            OPEX_production += results[f"GeoSource__daily_avg_{i}"] * 6
            CO2 += results[f"GeoSource__daily_avg_{i}"] * elecco2 * 0.051 + 0.041
            # CO2 += results[f"GeoSource__daily_avg_{i}"] * 230 * 0.169
            geo_prod += results[f"GeoSource__daily_avg_{i}"]
            OPEX_production += results[f"WarmteLink__daily_avg_{i}"] * 32
            CO2 += results[f"WarmteLink__daily_avg_{i}"] * 0.1 * elecco2
            OPEX_production += results[f"ATES__charge_amount_{i}"] * 0.7
            OPEX_production += results[f"ATES__discharge_amount_{i}"] * 0.7
            CO2 += results[f"ATES__charge_amount_{i}"] * 0.009 * elecco2
            CO2 += results[f"ATES__discharge_amount_{i}"] * 0.009 * elecco2
            WL_prod += results[f"WarmteLink__daily_avg_{i}"]
            prod[-1] += results[f"GeoSource__daily_avg_{i}"]
            prod[-1] += results[f"WarmteLink__daily_avg_{i}"]
            prod[-1] += results[f"ATES__daily_avg_{i}"]
            for k in range(1, 28):
                OPEX_production += results[f"HeatProducer{k}__daily_avg_{i}"] * 200
                CO2 += results[f"HeatProducer{k}__daily_avg_{i}"] * 1.126 * elecco2
                prod[-1] += results[f"HeatProducer{k}__daily_avg_{i}"]
                peak_prod += results[f"HeatProducer{k}__daily_avg_{i}"]
        OPEX_fixed += results["ATES__number_of_doublets"] * 125000.0 / 1.0e6 * 25
        OPEX_fixed += (
            results["GeoSource__number_of_wells"] * ((51367.0 + 15300.0) * 20) / 1.0e6 * 25
        )

        WOS_costs = 0.0
        for demand in demands:
            max_heat = np.max(results[f"{demand}.Heat_demand"])
            WOS_costs += max_heat / 1.0e6 * 30.0e3 / 1.0e6

        distribution_pipes = 0.0

        house_connection_costs = 0.0

        OPEX_production = OPEX_production * 25.0 * 24 / 1.0e6
        CO2 = CO2 * 25 * 24
        WL_prod = WL_prod * 25 * 24
        total_prod = np.sum(prod)
        total_prod = total_prod * 25 * 24
        # geo_avg_full_load = geo_prod / (results["GeoSource__max_size"] / 10**6 * 365)
        geo_prod = geo_prod * 24 * 25
        peak_prod = peak_prod * 24 * 25
        produced_total = peak_prod + geo_prod + WL_prod

        TCO = OPEX_fixed + OPEX_production + sources + storage + pipes_total

        TCO_with_distribution = TCO + WOS_costs + distribution_pipes + house_connection_costs

        everything_hp = 180.0 * 660.0e3 * (1.0 + 25 * 0.04) + 70 * total_prod / 4.5

        LCOE_HP = everything_hp / total_prod

        LCOE = TCO / total_prod
        LCOEs[-1].append(LCOE)
        CO2s[-1].append(CO2 / (total_prod * 1.126 * elecco2))
        OPEXs[-1].append(OPEX_fixed + OPEX_production)
        storages[-1].append(storage)
        sourcess[-1].append(sources)
        pipess[-1].append(pipes_total)
        tank_size_average[-1].append(np.mean(tank_sizes) / (998 * 4200 * 30))
        TCOs[-1].append(TCO)
        source_stategy[-1].append(
            (geo_prod / produced_total, WL_prod / produced_total, peak_prod / produced_total)
        )

        MGW = 19300
        EGW = 7900
        secondary_grid_costs = (MGW * (2750 + 1435) + EGW * (4500 + 6750)) / 1.0e6


print(np.asmatrix(LCOEs))
print(np.asmatrix(CO2s))
# print(np.asmatrix(TCOs))

hourly_ates = np.asarray(results["ATES.Heat_ates"].copy())
hourly_sources = None
hourly_geo = None
hourly_wl = None
hourly_peak = None
hourly_geo = np.asarray(results["GeoSource.Heat_source"].copy())
hourly_wl = np.asarray(results["WarmteLink.Heat_source"].copy())


total_demand = None

for demand in demands:
    if total_demand is None:
        total_demand = np.asarray(results[f"{demand}.Heat_demand"].copy())
    else:
        total_demand += np.asarray(results[f"{demand}.Heat_demand"].copy())


for i in range(1, 28):
    source = f"HeatProducer{i}"
    if hourly_peak is None:
        hourly_peak = np.asarray(results[f"{source}.Heat_source"].copy())
    else:
        hourly_peak += np.asarray(results[f"{source}.Heat_source"].copy())
    if hourly_sources is None:
        hourly_sources = np.asarray(results[f"{source}.Heat_source"].copy())
    else:
        hourly_sources += np.asarray(results[f"{source}.Heat_source"].copy())
hourly_buffer = None
for i in range(1, 28):
    buffer = f"HeatStorage{i}"
    if hourly_buffer is None:
        hourly_buffer = np.asarray(results[f"{buffer}.Heat_buffer"].copy())
    else:
        hourly_buffer += np.asarray(results[f"{buffer}.Heat_buffer"].copy())


plt.figure()
plt.plot(hourly_geo / 1.0e6, label="Geo", color="green")
plt.plot(hourly_wl / 1.0e6, label="WarmteLinQ", color="orange")
plt.plot(hourly_peak / 1.0e6, label="peak", color="red")
plt.plot(-hourly_buffer / 1.0e6, label="buffer", color="cyan")
plt.plot(hourly_ates / 1.0e6, label="ates", color="blue")
plt.plot(total_demand / 1.0e6, label="total_demand")
plt.xlabel("Time [hr]")
plt.ylabel("Heat Demand [MW]")
plt.ylim((-50, 200))
plt.title("Peak Day")
plt.legend()
daily_wl = [results[f"WarmteLink__daily_avg_{i}"] for i in range(365)]
daily_ates = [results[f"ATES__daily_avg_{i}"] for i in range(365)]
daily_geo = [results[f"GeoSource__daily_avg_{i}"] for i in range(365)]
daily_total_peak = [
    sum([results[f"HeatProducer{j}__daily_avg_{i}"] for j in range(1, 27)]) for i in range(365)
]
total_daily = [
    x + y + z + q for x, y, z, q in zip(daily_ates, daily_wl, daily_geo, daily_total_peak)
]

plt.figure()
plt.plot(daily_wl, label="WarmteLink", color="orange")
plt.plot(daily_ates, label="ATES", color="blue")
plt.plot(daily_geo, label="Geo", color="green")
plt.plot(daily_total_peak, label="Peak", color="red")
plt.plot(daily_avg_demands / WATT_TO_MEGA_WATT, label="total_demand", color="black")
plt.ylim((-25, 75))
plt.legend()
plt.title("Seasonal")
plt.xlabel("Day [-]")
plt.ylabel("Power [MW]")


plt.show()

a = 1
