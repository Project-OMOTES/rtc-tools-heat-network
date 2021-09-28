import logging
import math
from typing import Dict, Tuple, Type

from rtctools_heat_network.pycml import SymbolicParameter
from rtctools_heat_network.pycml.component_library.qth import (
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

from . import esdl
from .asset_to_component_base import MODIFIERS, _AssetToComponentBase
from .common import Asset
from .esdl_model_base import _ESDLModelBase

logger = logging.getLogger("rtctools_heat_network")


class AssetToQTHComponent(_AssetToComponentBase):
    def __init__(
        self,
        theta,
        *args,
        v_nominal=1.0,
        v_max=5.0,
        minimum_temperature=10.0,
        maximum_temperature=110.0,
        rho=988.0,
        cp=4200.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.theta = theta
        self.v_nominal = v_nominal
        self.v_max = v_max
        self.minimum_temperature = minimum_temperature
        self.maximum_temperature = maximum_temperature
        self.rho = rho
        self.cp = cp

    @property
    def _rho_cp_modifiers(self):
        return dict(rho=self.rho, cp=self.cp)

    def convert_buffer(self, asset: Asset) -> Tuple[Type[Buffer], MODIFIERS]:
        assert asset.asset_type == "HeatStorage"

        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)

        # Assume that:
        # - the capacity is the relative heat that can be stored in the buffer;
        # - the tanks are always at least 5% full;
        # - same height as radius to compute dimensions.
        min_fraction_tank_volume = 0.05
        capacity = asset.attributes["capacity"]
        r = (
            capacity
            * (1 + min_fraction_tank_volume)
            / (self.rho * self.cp * (supply_temperature - return_temperature) * math.pi)
        ) ** (1.0 / 3.0)

        modifiers = dict(
            Q_nominal=self._get_connected_q_nominal(asset),
            height=r,
            radius=r,
            heat_transfer_coeff=1.0,
            min_fraction_tank_volume=min_fraction_tank_volume,
            init_T_hot_tank=supply_temperature,
            init_T_cold_tank=return_temperature,
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Buffer, modifiers

    def convert_demand(self, asset: Asset) -> Tuple[Type[Demand], MODIFIERS]:
        assert asset.asset_type in {"GenericConsumer", "HeatingDemand"}

        # TODO: Why is the default zero for both, that's just weird. What if I
        # actually want a minimum of 0.0 (instead of treating it as a
        # NaN/None/not specified)
        # TODO: Are these even the correct values to use? It does not
        # say anything about whether this is the min/max on the feed or return line?
        # currently assuming they are the min/max of the _feed_ line.

        minimum_temperature = asset.attributes["minTemperature"]
        if minimum_temperature == 0.0:
            logger.warning(
                f"{asset.asset_type} '{asset.name}' has an unspecified minimum temperature. "
                f"Using default value of {self.minimum_temperature}."
            )
            minimum_temperature = self.minimum_temperature

        maximum_temperature = asset.attributes["maxTemperature"]
        if maximum_temperature == 0.0:
            logger.warning(
                f"{asset.asset_type} '{asset.name}' has an unspecified maximum temperature. "
                f"Using default value of {self.maximum_temperature}."
            )
            maximum_temperature = self.maximum_temperature

        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)

        heat_nominal = asset.attributes["power"] / 2.0

        modifiers = dict(
            theta=self.theta,
            Q_nominal=self._get_connected_q_nominal(asset),
            QTHIn=dict(T=dict(min=minimum_temperature, max=maximum_temperature)),
            Heat_demand=dict(min=0.0, max=asset.attributes["power"], nominal=heat_nominal),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Demand, modifiers

    def convert_node(self, asset: Asset) -> Tuple[Type[Node], MODIFIERS]:
        assert asset.asset_type == "Joint"  # TODO: temperature?

        sum_in = 0
        sum_out = 0

        for x in asset.attributes["port"].items:
            if type(x) == esdl.esdl.InPort:
                sum_in += len(x.connectedTo)
            if type(x) == esdl.esdl.OutPort:
                sum_out += len(x.connectedTo)

        # TODO: what do we want if no carrier is specified.
        carrier = asset.global_properties["carriers"][asset.in_port.carrier.id]
        if carrier["__rtc_type"] == "supply":
            temp = carrier["supplyTemperature"]
        elif carrier["__rtc_type"] == "return":
            temp = carrier["returnTemperature"]
        else:
            temp = 50.0
        modifiers = dict(
            n=sum_in + sum_out,
            temperature=temp,
        )

        return Node, modifiers

    def convert_pipe(self, asset: Asset) -> Tuple[Type[Pipe], MODIFIERS]:
        assert asset.asset_type == "Pipe"

        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)

        if "_ret" in asset.attributes["name"]:
            temperature = return_temperature
        else:
            temperature = supply_temperature

        diameter = asset.attributes["innerDiameter"]
        area = math.pi * asset.attributes["innerDiameter"] ** 2 / 4.0
        q_nominal = self.v_nominal * area
        q_max = self.v_max * area

        self._set_q_nominal(asset, q_nominal)

        # Insulation properties
        material = asset.attributes["material"]
        # NaN means the default values will be used
        insulation_thicknesses = math.nan
        conductivies_insulation = math.nan

        if material is not None:
            if isinstance(material, esdl.esdl.MatterReference):
                material = material.reference

            assert isinstance(material, esdl.esdl.CompoundMatter)
            components = material.component.items
            if components:
                insulation_thicknesses = [x.layerWidth for x in components]
                conductivies_insulation = [x.matter.thermalConductivity for x in components]

        # TODO: We can do better with the temperature bounds.
        # Maybe global ones (temperature_supply_max / min, and temperature_return_max / min?)
        modifiers = dict(
            length=asset.attributes["length"],
            diameter=diameter,
            temperature=temperature,
            disconnectable=self._is_buffer_pipe(asset),
            Q=dict(min=-q_max, max=q_max, nominal=q_nominal),
            QTHIn=dict(T=dict(min=self.minimum_temperature, max=self.maximum_temperature)),
            QTHOut=dict(T=dict(min=self.minimum_temperature, max=self.maximum_temperature)),
            insulation_thickness=insulation_thicknesses,
            conductivity_insulation=conductivies_insulation,
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        return Pipe, modifiers

    def convert_pump(self, asset: Asset) -> Tuple[Type[Pump], MODIFIERS]:
        assert asset.asset_type == "Pump"

        # TODO: Maximum pump head should come from ESDL component
        # specification. Alpha-2 CHESS release only checks for capacity, head
        # is always assumed to be realizable.
        modifiers = dict(
            Q=dict(
                min=0.0,
                max=asset.attributes["pumpCapacity"] / 3600.0,
                nominal=asset.attributes["pumpCapacity"] / 7200.0,
            ),
            dH=dict(min=0.0),
        )

        return Pump, modifiers

    def convert_source(self, asset: Asset) -> Tuple[Type[Source], MODIFIERS]:
        assert asset.asset_type in {
            "GasHeater",
            "GenericProducer",
            "GeothermalSource",
            "ResidualHeatSource",
        }

        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)

        # TODO: Why is the default zero for both, that's just weird. What if I
        # actually want a minimum of 0.0 (instead of treating it as a
        # NaN/None/not specified)
        # TODO: Are these even the correct values to use? It does not
        # say anything about whether this is the min/max on the feed or return line?
        # currently assuming they are the min/max of the _feed_ line.
        if asset.asset_type == "GenericProducer":
            minimum_temperature = supply_temperature
            maximum_temperature = supply_temperature
        else:
            minimum_temperature = asset.attributes["minTemperature"]
            if minimum_temperature == 0.0:
                logger.warning(
                    f"{asset.asset_type} '{asset.name}' has an unspecified minimum temperature. "
                    f"Using default value of {self.minimum_temperature}."
                )
                minimum_temperature = self.minimum_temperature

            maximum_temperature = asset.attributes["maxTemperature"]
            if maximum_temperature == 0.0:
                logger.warning(
                    f"{asset.asset_type} '{asset.name}' has an unspecified maximum temperature. "
                    f"Using default value of {self.maximum_temperature}."
                )
                maximum_temperature = self.maximum_temperature

        # get price per unit of energy,
        # assume cost of 1. if nothing is given (effectively heat loss minimization)
        price = 1.0
        if "costInformation" in asset.attributes.keys():
            if hasattr(asset.attributes["costInformation"], "variableOperationalCosts"):
                if hasattr(asset.attributes["costInformation"].variableOperationalCosts, "value"):
                    price = asset.attributes["costInformation"].variableOperationalCosts.value

        modifiers = dict(
            theta=self.theta,
            Q_nominal=self._get_connected_q_nominal(asset),
            QTHOut=dict(T=dict(min=minimum_temperature, max=maximum_temperature)),
            price=price,
            Heat_source=dict(
                min=0.0, max=asset.attributes["power"], nominal=asset.attributes["power"] / 2.0
            ),
            **self._supply_return_temperature_modifiers(asset),
            **self._rho_cp_modifiers,
        )

        if asset.asset_type == "GeothermalSource":
            # Note that the ESDL target flow rate is in kg/s, but we want m3/s
            modifiers["target_flow_rate"] = asset.attributes["flowRate"] / self.rho
            return GeothermalSource, modifiers
        else:
            return Source, modifiers

    def convert_control_valve(self, asset: Asset) -> Tuple[Type[ControlValve], MODIFIERS]:
        assert asset.asset_type == "Valve"

        return ControlValve, {}

    def convert_check_valve(self, asset: Asset) -> Tuple[Type[CheckValve], MODIFIERS]:
        assert asset.asset_type == "CheckValve"

        return CheckValve, {}


class ESDLQTHModel(_ESDLModelBase):
    _converter_class: _AssetToComponentBase = None

    def __init__(self, assets: Dict[str, Asset], converter_class=AssetToQTHComponent, **kwargs):
        super().__init__(None)

        self.add_variable(SymbolicParameter, "theta")

        converter = converter_class(theta=self.theta, **kwargs)

        self._esdl_convert(converter, assets, "QTH")
