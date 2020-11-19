from typing import Dict, Tuple, Type, Union

from rtctools_heat_network.pycml import Model as _Model

from .common import Asset
from .esdl_model_base import _RetryLaterException

MODIFIERS = Dict[str, Union[str, int, float]]


class _AssetToComponentBase:

    component_map = {
        "GenericConsumer": "demand",
        "HeatingDemand": "demand",
        "GasHeater": "source",
        "GenericProducer": "source",
        "GeothermalSource": "source",
        "ResidualHeatSource": "source",
        "Joint": "node",
        "Pipe": "pipe",
        "Pump": "pump",
        "HeatStorage": "buffer",
    }

    def __init__(self):
        self._port_to_q_nominal = {}

    def convert(self, asset: Asset) -> Tuple[Type[_Model], MODIFIERS]:
        """
        Converts an asset to a PyCML Heat component type and its modifiers.

        With more descriptive variable names the return type would be:
            Tuple[pycml_heat_component_type, Dict[component_attribute, new_attribute_value]]
        """

        dispatch_method_name = f"convert_{self.component_map[asset.asset_type]}"
        return getattr(self, dispatch_method_name)(asset)

    def _set_q_nominal(self, asset, q_nominal):
        self._port_to_q_nominal[asset.in_port] = q_nominal
        self._port_to_q_nominal[asset.out_port] = q_nominal

    def _get_connected_q_nominal(self, asset):
        try:
            connected_port = asset.in_port.connectedTo[0]
            q_nominal = self._port_to_q_nominal[connected_port]
        except KeyError:
            connected_port = asset.out_port.connectedTo[0]
            q_nominal = self._port_to_q_nominal.get(connected_port, None)

        if q_nominal is not None:
            self._set_q_nominal(asset, q_nominal)
            return q_nominal
        else:
            raise _RetryLaterException(
                f"Could not determine nominal discharge for {asset.asset_type} '{asset.name}'"
            )

    @staticmethod
    def _get_supply_return_temperatures(asset: Asset) -> Tuple[float, float]:
        carrier = asset.global_properties["carriers"][asset.in_port.carrier.id]
        supply_temperature = carrier["supplyTemperature"]
        return_temperature = carrier["returnTemperature"]

        assert supply_temperature > return_temperature
        # This is a bit dangerous, but the default (not-set) value is 0.0. We
        # however require it to be explicitly set.
        assert supply_temperature != 0.0
        assert return_temperature != 0.0

        return supply_temperature, return_temperature

    def _supply_return_temperature_modifiers(self, asset: Asset) -> MODIFIERS:
        supply_temperature, return_temperature = self._get_supply_return_temperatures(asset)
        return {"T_supply": supply_temperature, "T_return": return_temperature}
