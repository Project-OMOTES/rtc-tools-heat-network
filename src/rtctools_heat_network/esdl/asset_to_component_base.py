import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Type, Union

import esdl

from rtctools_heat_network.pycml import Model as _Model

from .common import Asset
from .esdl_model_base import _RetryLaterException, _SkipAssetException

logger = logging.getLogger("rtctools_heat_network")

MODIFIERS = Dict[str, Union[str, int, float]]


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

        for port in [asset.in_port, asset.out_port]:
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
        connected_type_in = self._port_to_esdl_component_type.get(
            asset.in_port.connectedTo[0], None
        )
        connected_type_out = self._port_to_esdl_component_type.get(
            asset.out_port.connectedTo[0], None
        )

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

    def convert_skip(self, asset: Asset):
        raise _SkipAssetException(asset)
