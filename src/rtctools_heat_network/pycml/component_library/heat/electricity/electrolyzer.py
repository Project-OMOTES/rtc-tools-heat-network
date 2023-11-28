# from rtctools_heat_network.pycml.component_library.heat._internal import BaseAsset
# from rtctools_heat_network.pycml.component_library.heat._internal.electricity_component import \
#     ElectricityComponent
# from rtctools_heat_network.pycml.component_library.heat.electricity.electricity_base import (
#     ElectricityPort,
# )
# from rtctools_heat_network.pycml.component_library.heat.gas.gas_base import (
#     GasPort,
# )


# # TODO: for now in the electricity folder, but maybe we can make a multicommodity folder,
# # where this is then placed.
# class Electrolyzer(ElectricityComponent, BaseAsset):
#     """
#     The electrolyser component consumes electricity and converts it into hydrogen. The efficiency
#     of the electrolyzer is modelled with an inequality constraint approach in the HeatMixin.
#     """

#     def __init__(self, name, **modifiers):
#         super().__init__(
#             name,
#             **self.merge_modifiers(
#                 dict(),
#                 modifiers,
#             ),
#         )

#         # TODO: potentially we can keep the component type as heat_pump and set subcomponent to
#         # heat_pump_elec, first need to check if there wouldn't be anything conflicting then.
#         self.component_type = "electrolyzer"
#         self.min_voltage = 1.0e4

#         self.add_variable(ElectricityPort, "ElectricityIn")
#         self.add_variable(GasPort, "GasOut")
