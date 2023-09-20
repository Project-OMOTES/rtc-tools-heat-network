import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Type, Union

import esdl
from esdl import TimeUnitEnum, UnitEnum

from rtctools_heat_network.pycml import Model as _Model

from .common import Asset
from .esdl_model_base import _RetryLaterException, _SkipAssetException

logger = logging.getLogger("rtctools_heat_network")

MODIFIERS = Dict[str, Union[str, int, float]]

HEAT_STORAGE_M3_WATER_PER_DEGREE_CELCIUS = 4200 * 988
WATTHOUR_TO_JOULE = 3600

MULTI_ENUM_NAME_TO_FACTOR = {
    esdl.MultiplierEnum.ATTO: 1e-18,
    esdl.MultiplierEnum.FEMTO: 1e-15,
    esdl.MultiplierEnum.PICO: 1e-12,
    esdl.MultiplierEnum.NANO: 1e-9,
    esdl.MultiplierEnum.MICRO: 1e-6,
    esdl.MultiplierEnum.MILLI: 1e-3,
    esdl.MultiplierEnum.CENTI: 1e-2,
    esdl.MultiplierEnum.DECI: 1e-1,
    esdl.MultiplierEnum.NONE: 1e0,
    esdl.MultiplierEnum.DEKA: 1e1,
    esdl.MultiplierEnum.HECTO: 1e2,
    esdl.MultiplierEnum.KILO: 1e3,
    esdl.MultiplierEnum.MEGA: 1e6,
    esdl.MultiplierEnum.GIGA: 1e9,
    esdl.MultiplierEnum.TERA: 1e12,
    esdl.MultiplierEnum.TERRA: 1e12,
    esdl.MultiplierEnum.PETA: 1e15,
    esdl.MultiplierEnum.EXA: 1e18,
}


class _AssetToComponentBase:
    # A map of pipe class name to edr asset in _edr_pipes.json
    STEEL_S1_PIPE_EDR_ASSETS = {
        "DN20": "Steel-S1-DN-20",
        "DN25": "Steel-S1-DN-25",
        "DN32": "Steel-S1-DN-32",
        "DN40": "Steel-S1-DN-40",
        "DN50": "Steel-S1-DN-50",
        "DN65": "Steel-S1-DN-65",
        "DN80": "Steel-S1-DN-80",
        "DN100": "Steel-S1-DN-100",
        "DN125": "Steel-S1-DN-125",
        "DN150": "Steel-S1-DN-150",
        "DN200": "Steel-S1-DN-200",
        "DN250": "Steel-S1-DN-250",
        "DN300": "Steel-S1-DN-300",
        "DN350": "Steel-S1-DN-350",
        "DN400": "Steel-S1-DN-400",
        "DN450": "Steel-S1-DN-450",
        "DN500": "Steel-S1-DN-500",
        "DN600": "Steel-S1-DN-600",
        "DN700": "Steel-S1-DN-700",
        "DN800": "Steel-S1-DN-800",
        "DN900": "Steel-S1-DN-900",
        "DN1000": "Steel-S1-DN-1000",
        "DN1100": "Steel-S1-DN-1100",
        "DN1200": "Steel-S1-DN-1200",
    }

    component_map = {
        "ATES": "ates",
        "ElectricityCable": "electricity_cable",
        "ElectricityDemand": "electricity_demand",
        "ElectricityProducer": "electricity_source",
        "Bus": "electricity_node",
        "GenericConsumer": "demand",
        "HeatExchange": "heat_exchanger",
        "HeatingDemand": "demand",
        "HeatPump": "heat_pump",
        "GasHeater": "source",
        "GasProducer": "gas_source",
        "GasDemand": "gas_demand",
        "GenericProducer": "source",
        "GeothermalSource": "source",
        "HeatProducer": "source",
        "ResidualHeatSource": "source",
        "GenericConversion": "heat_exchanger",
        "Joint": "node",
        "Pipe": "pipe",
        "Pump": "pump",
        "HeatStorage": "buffer",
        "Sensor": "skip",
        "Valve": "control_valve",
        "CheckValve": "check_valve",
    }

    def __init__(self):
        self._port_to_q_nominal = {}
        self._port_to_esdl_component_type = {}
        self._edr_pipes = json.load(
            open(os.path.join(Path(__file__).parent, "_edr_pipes.json"), "r")
        )

    def convert(self, asset: Asset) -> Tuple[Type[_Model], MODIFIERS]:
        """
        Converts an asset to a PyCML Heat component type and its modifiers.

        With more descriptive variable names the return type would be:
            Tuple[pycml_heat_component_type, Dict[component_attribute, new_attribute_value]]
        """
        ports = []
        if asset.in_ports is not None:
            ports.extend(asset.in_ports)
        if asset.out_ports is not None:
            ports.extend(asset.out_ports)
        assert len(ports) > 0

        for port in ports:
            self._port_to_esdl_component_type[port] = asset.asset_type

        dispatch_method_name = f"convert_{self.component_map[asset.asset_type]}"
        return getattr(self, dispatch_method_name)(asset)

    def _pipe_get_diameter_and_insulation(self, asset: Asset):
        # There are multiple ways to specify pipe properties like diameter and
        # material / insulation. We assume that DN `diameter` takes precedence
        # over `innerDiameter` and `material` (while logging warnings if both
        # are specified)
        full_name = f"{asset.asset_type} '{asset.name}'"
        if asset.attributes["innerDiameter"] and asset.attributes["diameter"].value > 0:
            logger.warning(
                f"{full_name}' has both 'innerDiameter' and 'diameter' specified. "
                f"Diameter of {asset.attributes['diameter'].name} will be used."
            )
        if asset.attributes["material"] and asset.attributes["diameter"].value > 0:
            logger.warning(
                f"{full_name}' has both 'material' and 'diameter' specified. "
                f"Insulation properties of {asset.attributes['diameter'].name} will be used."
            )
        if asset.attributes["material"] and (
            asset.attributes["diameter"].value == 0 and not asset.attributes["innerDiameter"]
        ):
            logger.warning(
                f"{full_name}' has only 'material' specified, but no information on diameter. "
                f"Diameter and insulation properties of DN200 will be used."
            )
        if asset.attributes["diameter"].value == 0 and not asset.attributes["innerDiameter"]:
            if asset.attributes["material"]:
                logger.warning(
                    f"{full_name}' has only 'material' specified, but no information on diameter. "
                    f"Diameter and insulation properties of DN200 will be used."
                )
            else:
                logger.warning(
                    f"{full_name}' has no DN size or innerDiameter specified. "
                    f"Diameter and insulation properties of DN200 will be used. "
                )

        edr_dn_size = None
        if asset.attributes["diameter"].value > 0:
            edr_dn_size = str(asset.attributes["diameter"].name)
        elif not asset.attributes["innerDiameter"]:
            edr_dn_size = "DN200"

        # NaN means the default values will be used
        insulation_thicknesses = math.nan
        conductivies_insulation = math.nan

        if edr_dn_size:
            # Get insulation and diameter properties from EDR asset with this size.
            edr_asset = self._edr_pipes[self.STEEL_S1_PIPE_EDR_ASSETS[edr_dn_size]]
            diameter = edr_asset["inner_diameter"]
            insulation_thicknesses = edr_asset["insulation_thicknesses"]
            conductivies_insulation = edr_asset["conductivies_insulation"]
        else:
            assert asset.attributes["innerDiameter"]
            diameter = asset.attributes["innerDiameter"]

            # Insulation properties
            material = asset.attributes["material"]

            if material is not None:
                if isinstance(material, esdl.esdl.MatterReference):
                    material = material.reference

                assert isinstance(material, esdl.esdl.CompoundMatter)
                components = material.component.items
                if components:
                    insulation_thicknesses = [x.layerWidth for x in components]
                    conductivies_insulation = [x.matter.thermalConductivity for x in components]

        return diameter, insulation_thicknesses, conductivies_insulation

    def _is_disconnectable_pipe(self, asset):
        # Source and buffer pipes are disconnectable by default
        assert asset.asset_type == "Pipe"
        if len(asset.in_ports) == 1 and len(asset.out_ports) == 1:
            connected_type_in = self._port_to_esdl_component_type.get(
                asset.in_ports[0].connectedTo[0], None
            )
            connected_type_out = self._port_to_esdl_component_type.get(
                asset.out_ports[0].connectedTo[0], None
            )
        else:
            raise RuntimeError("Pipe does not have 1 in port and 1 out port")
        # TODO: add other components which can be disabled and thus of which the pipes are allowed
        #  to be disabled: , "heat_exchanger", "heat_pump", "ates"
        types = {k for k, v in self.component_map.items() if v in {"source", "buffer"}}

        if types.intersection({connected_type_in, connected_type_out}):
            return True
        elif connected_type_in is None or connected_type_out is None:
            raise _RetryLaterException(
                f"Could not determine if {asset.asset_type} '{asset.name}' "
                f"is a source or buffer pipe"
            )
        else:
            return False

    def _set_q_nominal(self, asset, q_nominal):
        self._port_to_q_nominal[asset.in_ports[0]] = q_nominal
        self._port_to_q_nominal[asset.out_ports[0]] = q_nominal

    def _get_connected_q_nominal(self, asset):
        if len(asset.in_ports) == 1 and len(asset.out_ports) == 1:
            try:
                connected_port = asset.in_ports[0].connectedTo[0]
                q_nominal = self._port_to_q_nominal[connected_port]
            except KeyError:
                connected_port = asset.out_ports[0].connectedTo[0]
                q_nominal = self._port_to_q_nominal.get(connected_port, None)

            if q_nominal is not None:
                self._set_q_nominal(asset, q_nominal)
                return q_nominal
            else:
                raise _RetryLaterException(
                    f"Could not determine nominal discharge for {asset.asset_type} '{asset.name}'"
                )
        elif len(asset.in_ports) >= 2 and len(asset.out_ports) == 2:
            q_nominals = {}
            for p in asset.in_ports:
                if isinstance(p.carrier, esdl.HeatCommodity):
                    out_port = None
                    for p2 in asset.out_ports:
                        if p2.carrier.name.replace("_ret", "") == p.carrier.name.replace(
                            "_ret", ""
                        ):
                            out_port = p2
                    try:
                        connected_port = p.connectedTo[0]
                        q_nominal = self._port_to_q_nominal[connected_port]
                    except KeyError:
                        connected_port = out_port.connectedTo[0]
                        q_nominal = self._port_to_q_nominal.get(connected_port, None)
                    if q_nominal is not None:
                        self._port_to_q_nominal[p] = q_nominal
                        self._port_to_q_nominal[out_port] = q_nominal
                        if "_ret" in p.carrier.name:
                            q_nominals["Secondary"] = {"Q_nominal": q_nominal}
                        else:
                            q_nominals["Primary"] = {"Q_nominal": q_nominal}
                    else:
                        raise _RetryLaterException(
                            f"Could not determine nominal discharge for {asset.asset_type} "
                            f"{asset.name}"
                        )
            return q_nominals

    def _get_cost_figure_modifiers(self, asset: Asset) -> Dict:
        modifiers = {}

        if asset.attributes["costInformation"] is None:
            RuntimeWarning(f"{asset.name} has no cost information specified")
            return modifiers

        if asset.asset_type == "HeatStorage":
            modifiers["variable_operational_cost_coefficient"] = self.get_variable_opex_costs(asset)
            modifiers["fixed_operational_cost_coefficient"] = self.get_fixed_opex_costs(asset)
            modifiers["investment_cost_coefficient"] = self.get_investment_costs(
                asset, per_unit=UnitEnum.JOULE
            )
            modifiers["installation_cost"] = self.get_installation_costs(asset)
        elif asset.asset_type == "Pipe":
            modifiers["investment_cost_coefficient"] = self.get_investment_costs(
                asset, per_unit=UnitEnum.METRE
            )
            modifiers["installation_cost"] = self.get_installation_costs(asset)
        elif asset.asset_type == "HeatingDemand":
            modifiers["investment_cost_coefficient"] = self.get_investment_costs(
                asset, per_unit=UnitEnum.WATT
            )
            modifiers["installation_cost"] = self.get_installation_costs(asset)
        else:
            modifiers["variable_operational_cost_coefficient"] = self.get_variable_opex_costs(asset)
            modifiers["fixed_operational_cost_coefficient"] = self.get_fixed_opex_costs(asset)
            modifiers["investment_cost_coefficient"] = self.get_investment_costs(
                asset, per_unit=UnitEnum.WATT
            )
            modifiers["installation_cost"] = self.get_installation_costs(asset)

        return modifiers

    @staticmethod
    def _get_supply_return_temperatures(asset: Asset) -> Tuple[float, float]:
        assert len(asset.in_ports) == 1 and len(asset.out_ports) == 1
        carrier = asset.global_properties["carriers"][asset.in_ports[0].carrier.id]
        supply_temperature = carrier["supplyTemperature"]
        return_temperature = carrier["returnTemperature"]

        if carrier["__rtc_type"] == "supply":
            supply_temperature_id = carrier["id_number_mapping"]
            return_temperature_id = asset.global_properties["carriers"][
                asset.in_ports[0].carrier.id + "_ret"
            ]["id_number_mapping"]
        else:
            supply_temperature_id = asset.global_properties["carriers"][
                asset.in_ports[0].carrier.id[:-4]
            ]["id_number_mapping"]
            return_temperature_id = carrier["id_number_mapping"]

        assert supply_temperature > return_temperature
        # This is a bit dangerous, but the default (not-set) value is 0.0. We
        # however require it to be explicitly set.
        assert supply_temperature != 0.0
        assert return_temperature != 0.0

        return supply_temperature, return_temperature, supply_temperature_id, return_temperature_id

    def _supply_return_temperature_modifiers(self, asset: Asset) -> MODIFIERS:
        if len(asset.in_ports) == 1 and len(asset.out_ports) == 1:
            (
                supply_temperature,
                return_temperature,
                supply_temperature_id,
                return_temperature_id,
            ) = self._get_supply_return_temperatures(asset)
            return {
                "T_supply": supply_temperature,
                "T_return": return_temperature,
                "T_supply_id": supply_temperature_id,
                "T_return_id": return_temperature_id,
            }
        elif len(asset.in_ports) >= 2 and len(asset.out_ports) == 2:
            for p in asset.in_ports:
                if isinstance(p.carrier, esdl.HeatCommodity):
                    carrier = asset.global_properties["carriers"][p.carrier.id]
                    if "_ret" in p.carrier.name:
                        # This in the Secondary side carrier
                        sec_supply_temperature = carrier["supplyTemperature"]
                        sec_return_temperature_id = carrier["id_number_mapping"]
                        sec_return_temperature = carrier["returnTemperature"]
                        assert sec_supply_temperature > sec_return_temperature
                        assert sec_supply_temperature > 0.0
                    else:
                        # This in the Primary side carrier
                        prim_supply_temperature = carrier["supplyTemperature"]
                        prim_return_temperature = carrier["returnTemperature"]
                        prim_supply_temperature_id = carrier["id_number_mapping"]
                        assert prim_supply_temperature > prim_return_temperature
                        assert prim_supply_temperature > 0.0
            for p in asset.out_ports:
                if isinstance(p.carrier, esdl.HeatCommodity):
                    carrier = asset.global_properties["carriers"][p.carrier.id]
                    if "_ret" in p.carrier.name:
                        prim_return_temperature_id = carrier["id_number_mapping"]
                    else:
                        sec_supply_temperature_id = carrier["id_number_mapping"]
            if not prim_supply_temperature or not sec_supply_temperature:
                raise RuntimeError(
                    f"{asset.name} carriers are not specified correctly there should be a "
                    f"dedicated _ret carrier for each hydraulically coupled network"
                )
            temperatures = {
                "Primary": {
                    "T_supply": prim_supply_temperature,
                    "T_return": prim_return_temperature,
                    "T_supply_id": prim_supply_temperature_id,
                    "T_return_id": prim_return_temperature_id,
                },
                "Secondary": {
                    "T_supply": sec_supply_temperature,
                    "T_return": sec_return_temperature,
                    "T_supply_id": sec_supply_temperature_id,
                    "T_return_id": sec_return_temperature_id,
                },
            }
            return temperatures
        else:
            # unknown model type
            return {}

    @staticmethod
    def get_state(asset: Asset):
        if asset.attributes["state"].name == "DISABLED":
            value = 0.0
        elif asset.attributes["state"].name == "OPTIONAL":
            value = 2.0
        else:
            value = 1.0
        return value

    def get_variable_opex_costs(self, asset: Asset):
        """
        Returns the variable opex costs of an asset in Euros per Wh
        """
        cost_infos = dict()
        cost_infos["variableOperationalAndMaintenanceCosts"] = asset.attributes[
            "costInformation"
        ].variableOperationalAndMaintenanceCosts
        cost_infos["variableOperationalCosts"] = asset.attributes[
            "costInformation"
        ].variableOperationalCosts
        cost_infos["variableMaintenanceCosts"] = asset.attributes[
            "costInformation"
        ].variableMaintenanceCosts

        if all(cost_info is None for cost_info in cost_infos.values()):
            logger.warning(f"No variable OPEX cost information specified for asset {asset.name}")

        value = 0.0
        for cost_info in cost_infos.values():
            if cost_info is None:
                continue
            cost_value, unit, per_unit, per_time = self.get_cost_value_and_unit(cost_info)
            if unit != UnitEnum.EURO:
                logger.warning(f"Expected cost information {cost_info} to provide a cost in euros.")
                continue
            if per_time != TimeUnitEnum.NONE:
                logger.warning(
                    f"Specified OPEX for asset {asset.name} include a "
                    f"component per time, which we cannot handle."
                )
                continue
            if per_unit != UnitEnum.WATTHOUR:
                logger.warning(
                    f"Expected the specified OPEX for asset "
                    f"{asset.name} to be per Wh, but they are provided "
                    f"in {per_unit} instead."
                )
                continue
            value += cost_value

        return value

    def get_fixed_opex_costs(self, asset: Asset):
        """
        Returns the fixed opex costs of an asset in Euros per W
        """
        cost_infos = dict()
        cost_infos["fixedOperationalAndMaintenanceCosts"] = asset.attributes[
            "costInformation"
        ].fixedOperationalAndMaintenanceCosts
        cost_infos["fixedOperationalCosts"] = asset.attributes[
            "costInformation"
        ].fixedOperationalCosts
        cost_infos["fixedMaintenanceCosts"] = asset.attributes[
            "costInformation"
        ].fixedMaintenanceCosts

        if all(cost_info is None for cost_info in cost_infos.values()):
            logger.warning(f"No fixed OPEX cost information specified for asset {asset.name}")
            value = 0.0
        else:
            value = 0.0
            for cost_info in cost_infos.values():
                if cost_info is None:
                    continue
                cost_value, unit, per_unit, per_time = self.get_cost_value_and_unit(cost_info)
                if unit != UnitEnum.EURO:
                    RuntimeWarning(
                        f"Expected cost information {cost_info} to " f"provide a cost in euros."
                    )
                    continue
                if per_unit == UnitEnum.CUBIC_METRE:
                    # index is 0 because buffers only have one in out port
                    supply_temp = asset.global_properties["carriers"][asset.in_ports[0].carrier.id][
                        "supplyTemperature"
                    ]
                    return_temp = asset.global_properties["carriers"][asset.in_ports[0].carrier.id][
                        "returnTemperature"
                    ]
                    delta_temp = supply_temp - return_temp
                    m3_to_joule_factor = delta_temp * HEAT_STORAGE_M3_WATER_PER_DEGREE_CELCIUS
                    cost_value = cost_value / m3_to_joule_factor
                elif per_unit == UnitEnum.NONE:
                    if asset.asset_type == "HeatStorage":
                        size = asset.attributes["capacity"]
                        if size == 0.0:
                            # index is 0 because buffers only have one in out port
                            supply_temp = asset.global_properties["carriers"][
                                asset.in_ports[0].carrier.id
                            ]["supplyTemperature"]
                            return_temp = asset.global_properties["carriers"][
                                asset.in_ports[0].carrier.id
                            ]["returnTemperature"]
                            delta_temp = supply_temp - return_temp
                            m3_to_joule_factor = (
                                delta_temp * HEAT_STORAGE_M3_WATER_PER_DEGREE_CELCIUS
                            )
                            size = asset.attributes["volume"] * m3_to_joule_factor
                            if size == 0.0:
                                RuntimeWarning(f"{asset.name} has not capacity or volume set")
                                return 0.0
                    elif asset.asset_type == "ATES":
                        size = asset.attributes["maxChargeRate"]
                        if size == 0.0:
                            size = asset.attributes["capacity"] / (
                                365 * 24 * 3600 / 2
                            )  # only half a year it can load
                            if size == 0.0:
                                RuntimeWarning(
                                    f"{asset.name} has not capacity or maximum charge rate set"
                                )
                                return 0.0
                    else:
                        try:
                            size = asset.attributes["power"]
                            if size == 0.0:
                                continue
                        except KeyError:
                            return 0.0
                    cost_value = cost_value / size
                elif per_unit != UnitEnum.WATT:
                    RuntimeWarning(
                        f"Expected the specified OPEX for asset "
                        f"{asset.name} to be per W or m3, but they are provided "
                        f"in {per_unit} instead."
                    )
                    continue
                value += cost_value
        return value

    @staticmethod
    def get_cost_value_and_unit(cost_info: esdl.SingleValue):
        cost_value = cost_info.value
        unit_info = cost_info.profileQuantityAndUnit
        unit = unit_info.unit
        per_time_uni = unit_info.perTimeUnit
        per_unit = unit_info.perUnit
        multiplier = unit_info.multiplier
        per_multiplier = unit_info.perMultiplier

        cost_value *= MULTI_ENUM_NAME_TO_FACTOR[multiplier]
        cost_value /= MULTI_ENUM_NAME_TO_FACTOR[per_multiplier]

        return cost_value, unit, per_unit, per_time_uni

    def get_installation_costs(self, asset: Asset):
        cost_info = asset.attributes["costInformation"].installationCosts
        if cost_info is None:
            logger.warning(f"No installation cost info provided for asset " f"{asset.name}.")
            return 0.0
        cost_value, unit, per_unit, per_time = self.get_cost_value_and_unit(cost_info)
        if unit != UnitEnum.EURO:
            logger.warning(f"Expect cost information {cost_info} to " f"provide a cost in euros")
            return 0.0
        if not per_time == TimeUnitEnum.NONE:
            logger.warning(
                f"Specified installation costs of asset {asset.name}"
                f" include a component per time, which we "
                f"cannot handle."
            )
            return 0.0
        if not per_unit == UnitEnum.NONE:
            logger.warning(
                f"Specified installation costs of asset {asset.name}"
                f" include a component per unit {per_unit}, which we "
                f"cannot handle."
            )
            return 0.0
        return cost_value

    def get_investment_costs(self, asset: Asset, per_unit: UnitEnum = UnitEnum.WATT):
        """
        Returns the investment costs of an asset in Euros per W.
        """
        cost_info = asset.attributes["costInformation"].investmentCosts
        if cost_info is None:
            RuntimeWarning(f"No investment costs provided for asset " f"{asset.name}.")
            return 0.0
        (
            cost_value,
            unit_provided,
            per_unit_provided,
            per_time_provided,
        ) = self.get_cost_value_and_unit(cost_info)
        if unit_provided != UnitEnum.EURO:
            logger.warning(f"Expect cost information {cost_info} to " f"provide a cost in euros")
            return 0.0
        if not per_time_provided == TimeUnitEnum.NONE:
            logger.warning(
                f"Specified investment costs for asset {asset.name}"
                f" include a component per time, which we "
                f"cannot handle."
            )
            return 0.0
        if per_unit == UnitEnum.WATT:
            if not per_unit_provided == UnitEnum.WATT:
                logger.warning(
                    f"Expected the specified investment costs "
                    f"of asset {asset.name} to be per W, but they "
                    f"are provided in {per_unit_provided} "
                    f"instead."
                )
            return cost_value
        elif per_unit == UnitEnum.WATTHOUR:
            if not per_unit_provided == UnitEnum.WATTHOUR:
                logger.warning(
                    f"Expected the specified investment costs "
                    f"of asset {asset.name} to be per Wh, but they "
                    f"are provided in {per_unit_provided} "
                    f"instead."
                )
                return 0.0
            return cost_value
        elif per_unit == UnitEnum.METRE:
            if not per_unit_provided == UnitEnum.METRE:
                logger.warning(
                    f"Expected the specified investment costs "
                    f"of asset {asset.name} to be per meter, but they "
                    f"are provided in {per_unit_provided} "
                    f"instead."
                )
                return 0.0
            return cost_value
        elif per_unit == UnitEnum.JOULE:
            if per_unit_provided == UnitEnum.WATTHOUR:
                return cost_value / WATTHOUR_TO_JOULE
            elif per_unit_provided == UnitEnum.CUBIC_METRE:
                # index is 0 because buffers only have one in out port
                supply_temp = asset.global_properties["carriers"][asset.in_ports[0].carrier.id][
                    "supplyTemperature"
                ]
                return_temp = asset.global_properties["carriers"][asset.in_ports[0].carrier.id][
                    "returnTemperature"
                ]
                delta_temp = supply_temp - return_temp
                m3_to_joule_factor = delta_temp * HEAT_STORAGE_M3_WATER_PER_DEGREE_CELCIUS
                return cost_value / m3_to_joule_factor
            else:
                logger.warning(
                    f"Expected the specified investment costs "
                    f"of asset {asset.name} to be per Wh or m3, but "
                    f"they are provided in {per_unit_provided} "
                    f"instead."
                )
                return 0.0
        else:
            logger.warning(
                f"Cannot provide investment costs for asset " f"{asset.name} per {per_unit}"
            )
            return 0.0

    def convert_skip(self, asset: Asset):
        raise _SkipAssetException(asset)
