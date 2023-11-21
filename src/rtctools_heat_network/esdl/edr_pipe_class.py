from dataclasses import dataclass

from rtctools_heat_network.esdl.asset_to_component_base import _AssetToComponentBase
from rtctools_heat_network.pipe_class import PipeClass


@dataclass(frozen=True)
class EDRPipeClass(PipeClass):
    """
    Dataclass specifically to save the EDR pipe class information in. Note that we here utilize the
    edr information for:

    diameter: in meter
    u_1, u_2: insulative properties
    investment cost: in Eur/m
    """

    xml_string: str

    @classmethod
    def from_edr_class(cls, name: str, edr_class_name: str, maximum_velocity: float):
        """
        This function creates an EDR pipe object with a name and retrieving the information from
        the specified edr class.

        Parameters
        ----------
        name : The name to be used for the class
        edr_class_name : The name of the edr pipe we will use the attributes of
        maximum_velocity : The maximum velocity in m/s

        Returns
        -------
        The EDR pipe class
        """
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

        # TODO: utilize max velocity from the edr data as well?
        return EDRPipeClass(
            name, diameter, maximum_velocity, (u_1, u_2), investment_costs, xml_string
        )
