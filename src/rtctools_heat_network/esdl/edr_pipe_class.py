from dataclasses import dataclass

from rtctools_heat_network.esdl.asset_to_component_base import _AssetToComponentBase
from rtctools_heat_network.pipe_class import PipeClass


@dataclass(frozen=True)
class EDRPipeClass(PipeClass):
    xml_string: str

    @classmethod
    def from_edr_class(cls, name, edr_class_name, maximum_velocity):
        if not hasattr(EDRPipeClass, "._edr_pipes"):
            # TODO: Currently using private API of RTC-Tools Heat Network.
            # Make this functionality part of public API?
            EDRPipeClass._edr_pipes = _AssetToComponentBase()._edr_pipes

        edr_class = EDRPipeClass._edr_pipes[edr_class_name]
        diameter = edr_class["inner_diameter"]
        u_1 = edr_class["u_1"]
        u_2 = edr_class["u_2"]
        investment_costs = edr_class["investment_costs"]
        xml_string = edr_class["xml_string"]

        return EDRPipeClass(
            name, diameter, maximum_velocity, (u_1, u_2), investment_costs, xml_string
        )
