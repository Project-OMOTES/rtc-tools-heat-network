import math
from abc import abstractmethod
from enum import IntEnum
from math import isfinite
from typing import Dict

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.homotopy_mixin import HomotopyMixin

import rtctools_heat_network._darcy_weisbach as darcy_weisbach
from rtctools_heat_network.constants import GRAVITATIONAL_CONSTANT


class HeadLossOption(IntEnum):
    """
    Enumeration for the possible options to take head loss in pipes into
    account.
    """

    NO_HEADLOSS = 1
    CQ2 = 2
    LINEARIZED_DW = 3


class BaseComponentTypeMixin(CollocatedIntegratedOptimizationProblem):
    @abstractmethod
    def heat_network_components(self) -> Dict[str, str]:
        raise NotImplementedError


class ModelicaComponentTypeMixin(BaseComponentTypeMixin):
    def heat_network_components(self) -> Dict[str, str]:
        try:
            return self.__hn_component_types
        except AttributeError:
            string_parameters = self.string_parameters(0)

            # Find the components in model, detection by string
            # (name.component_type: type)
            component_types = sorted({v for k, v in string_parameters.items()})

            components = {}
            for c in component_types:
                components[c] = sorted({k[:-15] for k, v in string_parameters.items() if v == c})

            self.__hn_component_types = components

            return components


class QTHMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    """
    Adds handling of QTH heat network objects in your model to your
    optimization problem.

    Relevant parameters and variables are read from the model, and from this
    data a set of constraints and objectives are automatically generated, e.g.
    for the head loss and temperature losses in pipes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self, HomotopyMixin):
            # Note that we inherit ourselves, as there is a certain in which
            # inheritance is required.
            raise Exception("Class needs inherit from HomotopyMixin")

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------+-----------+-----------------------------+
        | Option                         | Type      | Default value               |
        +================================+===========+========================-----+
        | ``minimum_pressure_far_point`` | ``float`` | ``1.0`` bar                 |
        +--------------------------------+-----------+-----------------------------+
        | ``dtemp_demand``               | ``float`` | ``30`` °C                   |
        +--------------------------------+-----------+-----------------------------+
        | ``maximum_temperature_der``    | ``float`` | ``2.0`` °C/hour             |
        +--------------------------------+-----------+-----------------------------+
        | ``wall_roughness``             | ``float`` | ``0.002`` m                 |
        +--------------------------------+-----------+-----------------------------+
        | ``head_loss_option``           | ``enum``  | ``HeadLossOption.CQ2``      |
        +--------------------------------+-----------+-----------------------------+
        | ``estimated_velocity``         | ``float`` | ``1.0`` m/s (CQ2)           |
        +--------------------------------+-----------+-----------------------------+
        | ``maximum_velocity``           | ``float`` | ``2.0`` m/s (LINEARIZED_DW) |
        +--------------------------------+-----------+-----------------------------+
        | ``n_linearization_lines``      | ``int``   | ``10`` (LINEARIZED_DW)      |
        +--------------------------------+-----------+-----------------------------+

        The ``minimum_pressure_far_point`` gives the minimum pressure
        requirement at any demand node, which means that the pressure at the
        furthest point is also satisfied without inspecting the topology.

        The ``dtemp_demand`` parameter specifies what the (fixed) temperature
        drop is  over the demand nodes.

        The ``wall_roughness`` of the pipes plays a role in determining the
        resistance of the pipes.

        To model the head loss in pipes, the ``head_lost_option`` refers to
        one of the ways this can be done. See :class:`HeadLossOption` for more
        explanation on what each option entails. Note that all options model
        the head loss as an inequality, i.e. :math:`\Delta H \ge f(Q)`.

        When ``HeadLossOption.CQ2`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H \ge C
        \cdot Q^2`.

        When ``HeadLossOption.LINEARIZED_DW`` is used, the
        ``maximum_velocity`` needs to be set. The Darcy-Weisbach head loss
        relationship from :math:`v = 0` until :math:`v = maximum_velocity`
        will then be linearized using ``n_linearization`` lines.
        """

        options = {}

        options["minimum_pressure_far_point"] = 1.0
        options["dtemp_demand"] = 30
        options["maximum_temperature_der"] = 2.0
        options["wall_roughness"] = 2e-3
        options["head_loss_option"] = HeadLossOption.CQ2
        options["estimated_velocity"] = 1.0
        options["maximum_velocity"] = 2.0
        options["n_linearization_lines"] = 10

        return options

    def __pipe_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components()

        # Check if head_loss_option is correct
        options = self.heat_network_options()
        head_loss_option = options["head_loss_option"]

        if head_loss_option not in HeadLossOption.__members__.values():
            raise Exception(f"Head loss option '{head_loss_option}' does not exist")

        # Apply head loss constraints in pipes depending on the option set by
        # the user.
        if head_loss_option == HeadLossOption.NO_HEADLOSS:
            for pipe in components["pipe"]:
                constraints.append((self.state(f"{pipe}.dH"), 0.0, 0.0))

        elif head_loss_option == HeadLossOption.CQ2:
            estimated_velocity = options["estimated_velocity"]
            wall_roughness = options["wall_roughness"]

            for pipe in components["pipe"]:
                diameter = parameters[f"{pipe}.diameter"]
                length = parameters[f"{pipe}.length"]
                temperature = parameters[f"{pipe}.temperature"]

                ff = darcy_weisbach.friction_factor(
                    estimated_velocity, diameter, length, wall_roughness, temperature
                )

                # Compute c_v constant (where |dH| ~ c_v * v^2)
                c_v = length * ff / (2 * GRAVITATIONAL_CONSTANT) / diameter
                area = 0.25 * math.pi * diameter ** 2

                v = self.state(f"{pipe}.QTHOut.Q") / area
                constraints.append((-self.state(f"{pipe}.dH") - c_v * v ** 2, 0.0, np.inf))

        elif head_loss_option == HeadLossOption.LINEARIZED_DW:
            wall_roughness = options["wall_roughness"]
            v_max = options["maximum_velocity"]
            n_lines = options["n_linearization_lines"]

            for pipe in components["pipe"]:
                diameter = parameters[f"{pipe}.diameter"]
                area = math.pi * diameter ** 2 / 4
                length = parameters[f"{pipe}.length"]
                temperature = parameters[f"{pipe}.temperature"]

                a, b = darcy_weisbach.get_linear_pipe_dh_vs_q_fit(
                    diameter,
                    length,
                    wall_roughness,
                    temperature=temperature,
                    n_lines=n_lines,
                    v_max=v_max,
                )

                # Vectorize constraint for speed
                dh = ca.repmat(self.state(f"{pipe}.dH"), len(a))
                q = ca.repmat(self.state(f"{pipe}.Q"), len(a))
                constraints.append((-dh - (a * q + b), 0.0, np.inf))

        return constraints

    def __source_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components()

        for source in components["source"]:
            c = parameters[f"{source}.head_loss"]

            if c == 0.0:
                constraints.append(
                    (self.state(f"{source}.QTHIn.H") - self.state(f"{source}.QTHOut.H"), 0.0, 0.0)
                )
            else:
                constraints.append(
                    (
                        self.state(f"{source}.QTHIn.H")
                        - self.state(f"{source}.QTHOut.H")
                        - c * self.state(f"{source}.QTHOut.Q") ** 2,
                        0.0,
                        np.inf,
                    )
                )

        return constraints

    def __demand_head_loss_constraints(self, ensemble_member):
        constraints = []

        options = self.heat_network_options()
        components = self.heat_network_components()

        # Convert minimum pressure at far point from bar to meter (water) head
        min_head_loss = options["minimum_pressure_far_point"] * 10.2

        for d in components["demand"]:
            constraints.append(
                (self.state(d + ".QTHIn.H") - self.state(d + ".QTHOut.H"), min_head_loss, np.inf)
            )

        return constraints

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member).copy()

        options = self.heat_network_options()
        components = self.heat_network_components()

        # Head (loss) constraints
        constraints.extend(self.__pipe_head_loss_path_constraints(ensemble_member))
        constraints.extend(self.__source_head_loss_path_constraints(ensemble_member))
        constraints.extend(self.__demand_head_loss_constraints(ensemble_member))

        # Temperature gradient in pipes
        maximum_temperature_der = options["maximum_temperature_der"]

        if maximum_temperature_der is not None and isfinite(maximum_temperature_der):
            for pipe in components["pipe"]:
                # Note that maximum temperature change is expressed in °C per
                # hour. RTC-Tools uses seconds, so we have to scale the
                # derivative with 3600.
                dtemp_dt = self.der(f"{pipe}.QTHIn.T") * 3600.0
                constraints.append((dtemp_dt, -maximum_temperature_der, maximum_temperature_der))

        # Fix dT at demand nodes
        dtemp = options["dtemp_demand"]
        for d in components["demand"]:
            constraints.append(
                (self.state(d + ".QTHIn.T") - self.state(d + ".QTHOut.T"), dtemp, dtemp)
            )

        return constraints
