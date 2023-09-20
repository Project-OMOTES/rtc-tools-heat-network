import logging
import math
from typing import Dict, Tuple, Type

import esdl

from rtctools_heat_network.pycml.component_library.heat import (
    ATES,
    Buffer,
    CheckValve,
    ControlValve,
    Demand,
    ElectricityCable,
    ElectricityDemand,
    ElectricityNode,
    ElectricitySource,
    GasDemand,
    GasNode,
    GasPipe,
    GasSource,
    GeothermalSource,
    HeatExchanger,
    HeatPump,
    HeatPumpElec,
    Node,
    Pipe,
    Pump,
    Source,
)

from .asset_to_component_base import MODIFIERS, _AssetToComponentBase
from .common import Asset
from .esdl_model_base import _ESDLModelBase

logger = logging.getLogger("rtctools_heat_network")


class _ESDLInputException(Exception):
    pass


class AssetToHeatComponent(_AssetToComponentBase):
    def __init__(
        self,
        *args,
        v_nominal=1.0,
        v_max=5.0,
        rho=988.0,
        cp=4200.0,
        min_fraction_tank_volume=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.v_nominal = v_nominal
        self.v_max = v_max
        self.rho = rho
        self.cp = cp
        self.min_fraction_tank_volume = min_fraction_tank_volume

    @property
    def _rho_cp_modifiers(self):
        return dict(rho=self.rho, cp=self.cp)

    def convert_buffer(self, asset: Asset) -> Tuple[Type[Buffer], MODIFIERS]:
        assert asset.asset_type == "HeatStorage"

        temperature_modifiers = self._supply_return_temperature_modifiers(asset)

        supply_temperature = temperature_modifiers["T_supply"]
        return_temperature = temperature_modifiers["T_return"]

        # Assume that:
        # - the capacity is the relative heat that can be stored in the buffer;
        # - the tanks are always at least `min_fraction_tank_volume` full;
        # - same height as radius to compute dimensions.
        if asset.attributes["capacity"] and asset.attributes["volume"]:
            logger.warning(
                f"{asset.asset_type} '{asset.name}' has both capacity and volume specified. "
                f"Volume with value of {asset.attributes['volume']} m3 will be used."
            )

        capacity = 0.0
        if asset.attributes["volume"]:
            capacity = (
                asset.attributes["volume"]
                * self.rho
                * self.cp
                * (supply_temperature - return_temperature)
            )
        elif asset.attributes["capacity"]:
            capacity = asset.attributes["capacity"]
        else:
            logger.error(
                f"{asset.asset_type} '{asset.name}' has both not capacity and volume specified. "
                f"Please specify one of the two"
            )

        assert capacity > 0.0
        min_fraction_tank_volume = self.min_fraction_tank_volume
        if self.get_state(asset) == 0 or self.get_state(asset) == 2:
            min_fraction_tank_volume = 0.0
        # We assume that the height equals the radius of the buffer.
        r = (
            capacity
            * (1 + min_fraction_tank_volume)
            / (self.rho * self.cp * (supply_temperature - return_temperature) * math.pi)
        ) ** (1.0 / 3.0)

        min_heat = capacity * min_fraction_tank_volume
        max_heat = capacity * (1 + min_fraction_tank_volume)
        assert max_heat > 0.0
        # default is set to 10MW

        hfr_charge_max = (
            asset.attributes.get("maxChargeRate")
            if asset.attributes.get("maxChargeRate")
            else 10.0e6
        )
        hfr_discharge_max = (
            asset.attributes.get("maxDischargeRate")
            if asset.attributes.get("maxDischargeRate")
            else 10.0e6
        )

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            height=r,
            radius=r,
            heat_transfer_coeff=1.0,
            state=self.get_state(asset),
            min_fraction_tank_volume=min_fraction_tank_volume,
            Stored_heat=dict(min=min_heat, max=max_heat),
            Heat_buffer=dict(min=-hfr_discharge_max, max=hfr_charge_max),
            Heat_flow=dict(min=-hfr_discharge_max, max=hfr_charge_max, nominal=hfr_charge_max),
            init_Heat=min_heat,
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
            **self._get_cost_figure_modifiers(asset),
        )

        return Buffer, modifiers

    def convert_demand(self, asset: Asset) -> Tuple[Type[Demand], MODIFIERS]:
        assert asset.asset_type in {"GenericConsumer", "HeatingDemand"}

        max_demand = asset.attributes["power"] if asset.attributes["power"] else math.inf

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            Heat_demand=dict(max=max_demand),
            Heat_flow=dict(max=max_demand, nominal=max_demand / 2.0),
            state=self.get_state(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
            **self._get_cost_figure_modifiers(asset),
        )

        return Demand, modifiers

    def convert_node(self, asset: Asset) -> Tuple[Type[Node], MODIFIERS]:
        assert asset.asset_type == "Joint"

        sum_in = 0
        sum_out = 0

        node_carrier = None
        for x in asset.attributes["port"].items:
            if node_carrier is None:
                node_carrier = x.carrier.name
            else:
                if node_carrier != x.carrier.name:
                    raise _ESDLInputException(
                        f"{asset.name} has multiple carriers mixing which is not allowed. "
                        f"Only one carrier (carrier couple) allowed in hydraulicly "
                        f"coupled system"
                    )
            if isinstance(x, esdl.esdl.InPort):
                sum_in += len(x.connectedTo)
            if isinstance(x, esdl.esdl.OutPort):
                sum_out += len(x.connectedTo)

        modifiers = dict(
            n=sum_in + sum_out,
            state=self.get_state(asset),
        )

        if isinstance(asset.in_ports[0].carrier, esdl.esdl.GasCommodity) or isinstance(
            asset.out_ports[0].carrier, esdl.esdl.GasCommodity
        ):
            modifiers = dict(
                n=sum_in + sum_out,
            )
            return GasNode, modifiers

        return Node, modifiers

    def convert_pipe(self, asset: Asset) -> Tuple[Type[Pipe], MODIFIERS]:
        assert asset.asset_type == "Pipe"

        length = asset.attributes["length"]
        if length < 25.0:
            length = 25.0

        (
            diameter,
            insulation_thicknesses,
            conductivies_insulation,
        ) = self._pipe_get_diameter_and_insulation(asset)

        if isinstance(asset.in_ports[0].carrier, esdl.esdl.GasCommodity):
            modifiers = dict(length=length, diameter=diameter)

            return GasPipe, modifiers

        temperature_modifiers = self._supply_return_temperature_modifiers(asset)

        supply_temperature = temperature_modifiers["T_supply"]
        return_temperature = temperature_modifiers["T_return"]

        if "_ret" in asset.attributes["name"]:
            temperature = return_temperature
        else:
            temperature = supply_temperature

        # Compute the maximum heat flow based on an assumed maximum velocity
        area = math.pi * diameter**2 / 4.0
        q_max = area * self.v_max
        q_nominal = area * self.v_nominal

        self._set_q_nominal(asset, q_nominal)

        # TODO: This might be an underestimation. We need to add the total
        # heat losses in the system to get a proper upper bound. Maybe move
        # calculation of Heat bounds to the HeatMixin?
        delta_temperature = supply_temperature - return_temperature
        # hfr_max = self.rho * self.cp * q_max * delta_temperature * 2
        hfr_max = (
            self.rho * self.cp * q_max * delta_temperature
        )  # TODO: are there any physical implications of using this bound

        assert hfr_max > 0.0

        modifiers = dict(
            Q_nominal=q_nominal,
            length=length,
            diameter=diameter,
            temperature=temperature,
            disconnectable=self._is_disconnectable_pipe(asset),
            HeatIn=dict(
                Heat=dict(min=-hfr_max, max=hfr_max),
                Q=dict(min=-q_max, max=q_max),
            ),
            HeatOut=dict(
                Heat=dict(min=-hfr_max, max=hfr_max),
                Q=dict(min=-q_max, max=q_max),
            ),
            Heat_flow=dict(min=-hfr_max, max=hfr_max, nominal=hfr_max),
            insulation_thickness=insulation_thicknesses,
            conductivity_insulation=conductivies_insulation,
            state=self.get_state(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
            **self._get_cost_figure_modifiers(asset),
        )
        if "T_ground" in asset.attributes.keys():
            modifiers["T_ground"] = asset.attributes["T_ground"]

        return Pipe, modifiers

    def convert_pump(self, asset: Asset) -> Tuple[Type[Pump], MODIFIERS]:
        assert asset.asset_type == "Pump"

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            state=self.get_state(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Pump, modifiers

    def convert_heat_exchanger(self, asset: Asset) -> Tuple[Type[HeatExchanger], MODIFIERS]:
        assert asset.asset_type in {
            "GenericConversion",
            "HeatExchange",
        }

        params_t = self._supply_return_temperature_modifiers(asset)
        params_q = self._get_connected_q_nominal(asset)
        params = {}

        if params_t["Primary"]["T_supply"] < params_t["Secondary"]["T_supply"]:
            logger.error(
                f"{asset.name} has a primary side supply temperature, "
                f"{params_t['Primary']['T_supply']}, that is higher than the secondary supply , "
                f"{params_t['Secondary']['T_supply']}. This is not possible as the HEX can only "
                "transfer heat from primary to secondary."
            )
            assert params_t["Primary"]["T_supply"] >= params_t["Secondary"]["T_supply"]
        if params_t["Primary"]["T_return"] < params_t["Secondary"]["T_return"]:
            logger.error(
                f"{asset.name} has a primary side return temperature that is lower than the "
                f"secondary return temperature. This is not possible as the HEX can only transfer "
                f"heat from primary to secondary."
            )
            assert params_t["Primary"]["T_return"] >= params_t["Secondary"]["T_return"]

        if asset.asset_type == "GenericConversion":
            max_power = asset.attributes["power"] if asset.attributes["power"] else math.inf
        else:
            # TODO: Current max_power estimation is not very accurate, a more physics based
            # estimation should be implemented, maybe using other ESDL attributs.
            max_power = (
                asset.attributes["heatTransferCoefficient"]
                * (params_t["Primary"]["T_supply"] - params_t["Secondary"]["T_return"])
                / 2.0
            )

        prim_heat = dict(
            Heat_in=dict(min=-max_power, max=max_power, nominal=max_power / 2.0),
            Heat_out=dict(min=-max_power, max=max_power, nominal=max_power / 2.0),
            Q_nominal=max_power
            / (
                2
                * self.rho
                * self.cp
                * (params_t["Primary"]["T_supply"] - params_t["Primary"]["T_return"])
            ),
        )
        sec_heat = dict(
            Heat_in=dict(min=-max_power, max=max_power, nominal=max_power / 2.0),
            Heat_out=dict(min=-max_power, max=max_power, nominal=max_power / 2.0),
            Q_nominal=max_power
            / (
                2
                * self.cp
                * self.rho
                * (params_t["Secondary"]["T_supply"] - params_t["Secondary"]["T_return"])
            ),
        )
        params["Primary"] = {**params_t["Primary"], **params_q["Primary"], **prim_heat}
        params["Secondary"] = {**params_t["Secondary"], **params_q["Secondary"], **sec_heat}

        if not asset.attributes["efficiency"]:
            efficiency = 1.0
        else:
            efficiency = asset.attributes["efficiency"]

        modifiers = dict(
            efficiency=efficiency,
            nominal=max_power / 2.0,
            Primary_heat=dict(min=0.0, max=max_power, nominal=max_power / 2.0),
            Secondary_heat=dict(min=0.0, max=max_power, nominal=max_power / 2.0),
            Heat_flow=dict(min=0.0, max=max_power, nominal=max_power / 2.0),
            state=self.get_state(asset),
            **self._get_cost_figure_modifiers(asset),
            **params,
        )
        return HeatExchanger, modifiers

    def convert_heat_pump(self, asset: Asset) -> Tuple[Type[HeatPump], MODIFIERS]:
        assert asset.asset_type in {
            "HeatPump",
        }
        if not asset.attributes["power"]:
            raise _ESDLInputException(f"{asset.name} has no power specified")
        else:
            power_electrical = asset.attributes["power"]

        if not asset.attributes["COP"]:
            raise _ESDLInputException(
                f"{asset.name} has not COP specified, this is required for the model"
            )
        else:
            cop = asset.attributes["COP"]

        params_t = self._supply_return_temperature_modifiers(asset)
        params_q = self._get_connected_q_nominal(asset)
        params = {}
        params["Primary"] = {**params_t["Primary"], **params_q["Primary"]}
        params["Secondary"] = {**params_t["Secondary"], **params_q["Secondary"]}
        max_power = power_electrical * (1.0 + cop)  # TODO: dit kan zijn power_electrical*cop

        modifiers = dict(
            COP=cop,
            Power_elec=dict(min=0.0, max=power_electrical, nominal=power_electrical / 2.0),
            Primary_heat=dict(min=0.0, max=max_power, nominal=max_power / 2.0),
            Secondary_heat=dict(min=0.0, max=max_power, nominal=max_power / 2.0),
            Heat_flow=dict(min=0.0, max=max_power, nominal=1.0e6 / 2.0),
            state=self.get_state(asset),
            **self._get_cost_figure_modifiers(asset),
            **params,
        )
        if len(asset.in_ports) == 2:
            return HeatPump, modifiers
        elif len(asset.in_ports) == 3:
            return HeatPumpElec, modifiers

    def convert_source(self, asset: Asset) -> Tuple[Type[Source], MODIFIERS]:
        assert asset.asset_type in {
            "GasHeater",
            "GenericProducer",
            "HeatProducer",
            "GeothermalSource",
            "ResidualHeatSource",
        }

        max_supply = asset.attributes["power"]

        if not max_supply:
            logger.error(f"{asset.asset_type} '{asset.name}' has no max power specified. ")
        assert max_supply > 0.0

        # get price per unit of energy,
        # assume cost of 1. if nothing is given (effectively heat loss minimization)

        co2_coefficient = 1.0
        if hasattr(asset.attributes["KPIs"], "kpi"):
            co2_coefficient = asset.attributes["KPIs"].kpi.items[0].value

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            state=self.get_state(asset),
            co2_coeff=co2_coefficient,
            Heat_source=dict(min=0.0, max=max_supply, nominal=max_supply / 2.0),
            Heat_flow=dict(min=0.0, max=max_supply, nominal=max_supply / 2.0),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
            **self._get_cost_figure_modifiers(asset),
        )

        if asset.asset_type == "GeothermalSource":
            modifiers["nr_of_doublets"] = asset.attributes["aggregationCount"]
            modifiers["Heat_source"] = dict(
                min=0.0,
                max=max_supply * asset.attributes["aggregationCount"],
                nominal=max_supply / 2.0,
            )
            modifiers["Heat_flow"] = dict(
                min=0.0,
                max=max_supply * asset.attributes["aggregationCount"],
                nominal=max_supply / 2.0,
            )
            try:
                modifiers["single_doublet_power"] = asset.attributes["single_doublet_power"]
            except KeyError:
                modifiers["single_doublet_power"] = max_supply
            # Note that the ESDL target flow rate is in kg/s, but we want m3/s
            try:
                modifiers["target_flow_rate"] = asset.attributes["flowRate"] / self.rho
            except KeyError:
                logger.warning(
                    f"{asset.asset_type} '{asset.name}' has no desired flow rate specified. "
                    f"'{asset.name}' will not be actuated in a constant manner"
                )

            return GeothermalSource, modifiers
        else:
            return Source, modifiers

    def convert_ates(self, asset: Asset) -> Tuple[Type[ATES], MODIFIERS]:
        assert asset.asset_type in {
            "ATES",
        }

        hfr_charge_max = asset.attributes.get("maxChargeRate", math.inf)
        hfr_discharge_max = asset.attributes.get("maxDischargeRate", math.inf)

        try:
            single_doublet_power = asset.attributes["single_doublet_power"]
        except KeyError:
            single_doublet_power = hfr_discharge_max

        # We assume the efficiency is realized over a period of 100 days
        efficiency = asset.attributes["dischargeEfficiency"]
        if not efficiency:
            efficiency = 0.7

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            single_doublet_power=single_doublet_power,
            heat_loss_coeff=(1.0 - efficiency ** (1.0 / 100.0)) / (3600.0 * 24.0),
            state=self.get_state(asset),
            nr_of_doublets=asset.attributes["aggregationCount"],
            Heat_ates=dict(
                min=-hfr_charge_max * asset.attributes["aggregationCount"],
                max=hfr_discharge_max * asset.attributes["aggregationCount"],
                nominal=hfr_discharge_max / 2.0,
            ),
            Stored_heat=dict(
                min=0.0,
                max=hfr_charge_max * asset.attributes["aggregationCount"] * 180.0 * 24 * 3600.0,
                nominal=hfr_charge_max * asset.attributes["aggregationCount"] * 30.0 * 24 * 3600.0,
            ),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
            **self._get_cost_figure_modifiers(asset),
        )

        return ATES, modifiers

    def convert_control_valve(self, asset: Asset) -> Tuple[Type[ControlValve], MODIFIERS]:
        assert asset.asset_type == "Valve"

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            state=self.get_state(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return ControlValve, modifiers

    def convert_check_valve(self, asset: Asset) -> Tuple[Type[CheckValve], MODIFIERS]:
        assert asset.asset_type == "CheckValve"

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            state=self.get_state(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return CheckValve, modifiers

    def convert_electricity_demand(self, asset: Asset) -> Tuple[Type[ElectricityDemand], MODIFIERS]:
        assert asset.asset_type in {"ElectricityDemand"}

        max_demand = asset.attributes.get("power", math.inf)

        modifiers = dict(Electricity_demand=dict(max=max_demand, nominal=max_demand / 2.0))

        return ElectricityDemand, modifiers

    def convert_electricity_source(self, asset: Asset) -> Tuple[Type[ElectricitySource], MODIFIERS]:
        assert asset.asset_type in {"ElectricityProducer"}

        max_supply = asset.attributes.get(
            "power", math.inf
        )  # I think it would break with math.inf as input

        modifiers = dict(
            Electricity_source=dict(min=0.0, max=max_supply, nominal=max_supply / 2.0),
            ElectricityOut=dict(
                V=dict(min=0.0), I=dict(min=0.0), Power=dict(nominal=max_supply / 2.0)
            ),
        )

        return ElectricitySource, modifiers

    def convert_electricity_node(self, asset: Asset) -> Tuple[Type[ElectricityNode], MODIFIERS]:
        assert asset.asset_type in {"Bus"}

        sum_in = 0
        sum_out = 0

        node_carrier = None
        for x in asset.attributes["port"].items:
            if node_carrier is None:
                node_carrier = x.carrier.name
            else:
                if node_carrier != x.carrier.name:
                    raise _ESDLInputException(
                        f"{asset.name} has multiple carriers mixing which is not allowed. "
                    )
            if isinstance(x, esdl.esdl.InPort):
                sum_in += len(x.connectedTo)
            if isinstance(x, esdl.esdl.OutPort):
                sum_out += len(x.connectedTo)

        modifiers = dict(n=sum_in + sum_out)

        return ElectricityNode, modifiers

    def convert_electricity_cable(self, asset: Asset) -> Tuple[Type[ElectricityCable], MODIFIERS]:
        assert asset.asset_type in {"ElectricityCable"}

        modifiers = dict(
            length=asset.attributes["length"],
            ElectricityOut=dict(
                V=dict(min=0.0), I=dict(min=-142.0, max=142.0), Power=dict(nominal=1e2)
            ),
        )
        return ElectricityCable, modifiers

    def convert_gas_demand(self, asset: Asset):
        assert asset.asset_type in {"GasDemand"}

        modifiers = dict()

        return GasDemand, modifiers

    def convert_gas_source(self, asset: Asset):
        assert asset.asset_type in {"GasProducer"}

        modifiers = dict()

        return GasSource, modifiers


class ESDLHeatModel(_ESDLModelBase):
    def __init__(self, assets: Dict[str, Asset], converter_class=AssetToHeatComponent, **kwargs):
        super().__init__(None)

        converter = converter_class(**kwargs)

        self._esdl_convert(converter, assets, "Heat")
