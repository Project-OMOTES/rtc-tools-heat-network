import logging
import math
from typing import Dict, Tuple, Type

import esdl

from rtctools_heat_network.pycml.component_library.heat import (
    Buffer,
    CheckValve,
    ControlValve,
    Demand,
    GeothermalSource,
    Node,
    Pipe,
    Pump,
    Source,
)

from .asset_to_component_base import MODIFIERS, _AssetToComponentBase
from .common import Asset
from .esdl_model_base import _ESDLModelBase

logger = logging.getLogger("rtctools_heat_network")


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

        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)

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

        r = (
            capacity
            * (1 + min_fraction_tank_volume)
            / (self.rho * self.cp * (supply_temperature - return_temperature) * math.pi)
        ) ** (1.0 / 3.0)

        min_heat = capacity * min_fraction_tank_volume
        max_heat = capacity * (1 + min_fraction_tank_volume)
        assert max_heat > 0.0
        hfr_charge_max = asset.attributes.get("maxChargeRate", math.inf) or math.inf
        hfr_discharge_max = asset.attributes.get("maxDischargeRate", math.inf) or math.inf

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            height=r,
            radius=r,
            heat_transfer_coeff=1.0,
            min_fraction_tank_volume=min_fraction_tank_volume,
            Stored_heat=dict(min=min_heat, max=max_heat),
            Heat_buffer=dict(min=-hfr_discharge_max, max=hfr_charge_max),
            init_Heat=min_heat,
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Buffer, modifiers

    def convert_demand(self, asset: Asset) -> Tuple[Type[Demand], MODIFIERS]:
        assert asset.asset_type in {"GenericConsumer", "HeatingDemand"}

        max_demand = asset.attributes["power"] if asset.attributes["power"] else math.inf

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            Heat_demand=dict(max=max_demand),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Demand, modifiers

    def convert_node(self, asset: Asset) -> Tuple[Type[Node], MODIFIERS]:
        assert asset.asset_type == "Joint"

        sum_in = 0
        sum_out = 0

        for x in asset.attributes["port"].items:
            if type(x) == esdl.esdl.InPort:
                sum_in += len(x.connectedTo)
            if type(x) == esdl.esdl.OutPort:
                sum_out += len(x.connectedTo)

        modifiers = dict(
            n=sum_in + sum_out,
        )

        return Node, modifiers

    def convert_pipe(self, asset: Asset) -> Tuple[Type[Pipe], MODIFIERS]:
        assert asset.asset_type == "Pipe"

        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)

        if "_ret" in asset.attributes["name"]:
            temperature = return_temperature
        else:
            temperature = supply_temperature

        (
            diameter,
            insulation_thicknesses,
            conductivies_insulation,
        ) = self._pipe_get_diameter_and_insulation(asset)

        # Compute the maximum heat flow based on an assumed maximum velocity
        area = math.pi * diameter**2 / 4.0
        q_max = area * self.v_max
        q_nominal = area * self.v_nominal

        self._set_q_nominal(asset, q_nominal)

        # TODO: This might be an underestimation. We need to add the total
        # heat losses in the system to get a proper upper bound. Maybe move
        # calculation of Heat bounds to the HeatMixin?
        delta_temperature = supply_temperature - return_temperature
        hfr_max = self.rho * self.cp * q_max * delta_temperature * 2

        assert hfr_max > 0.0

        modifiers = dict(
            Q_nominal=q_nominal,
            length=asset.attributes["length"],
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
            insulation_thickness=insulation_thicknesses,
            conductivity_insulation=conductivies_insulation,
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Pipe, modifiers

    def convert_pump(self, asset: Asset) -> Tuple[Type[Pump], MODIFIERS]:
        assert asset.asset_type == "Pump"

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Pump, modifiers

    def convert_source(self, asset: Asset) -> Tuple[Type[Source], MODIFIERS]:
        assert asset.asset_type in {
            "GasHeater",
            "GenericProducer",
            "GeothermalSource",
            "ResidualHeatSource",
        }

        max_supply = asset.attributes["power"]
        if not max_supply:
            logger.error(f"{asset.asset_type} '{asset.name}' has no max power specified. ")
        assert max_supply > 0.0

        # get price per unit of energy,
        # assume cost of 1. if nothing is given (effectively heat loss minimization)
        price = 1.0
        if "costInformation" in asset.attributes.keys():
            if hasattr(asset.attributes["costInformation"], "variableOperationalCosts"):
                if hasattr(asset.attributes["costInformation"].variableOperationalCosts, "value"):
                    price = asset.attributes["costInformation"].variableOperationalCosts.value

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            price=price,
            Heat_source=dict(min=0.0, max=max_supply, nominal=max_supply / 2.0),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        if asset.asset_type == "GeothermalSource":
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

    def convert_control_valve(self, asset: Asset) -> Tuple[Type[ControlValve], MODIFIERS]:
        assert asset.asset_type == "Valve"

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return ControlValve, modifiers

    def convert_check_valve(self, asset: Asset) -> Tuple[Type[CheckValve], MODIFIERS]:
        assert asset.asset_type == "CheckValve"

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return CheckValve, modifiers


class ESDLHeatModel(_ESDLModelBase):
    def __init__(self, assets: Dict[str, Asset], converter_class=AssetToHeatComponent, **kwargs):
        super().__init__(None)

        converter = converter_class(**kwargs)

        self._esdl_convert(converter, assets, "Heat")
