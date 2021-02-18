import logging
import math
from abc import abstractmethod
from enum import IntEnum
from typing import List, Optional, Tuple, Type, Union

import casadi as ca

import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin_base import Goal, _GoalProgrammingMixinBase
from rtctools.optimization.optimization_problem import BT

import rtctools_heat_network._darcy_weisbach as darcy_weisbach
from rtctools_heat_network.base_component_type_mixin import BaseComponentTypeMixin

from .constants import GRAVITATIONAL_CONSTANT


logger = logging.getLogger("rtctools_heat_network")


class HeadLossOption(IntEnum):
    """
    Enumeration for the possible options to take head loss in pipes into
    account.
    """

    NO_HEADLOSS = 1
    CQ2_INEQUALITY = 2
    LINEARIZED_DW = 3
    LINEAR = 4
    CQ2_EQUALITY = 5


class _MinimizeHeadLosses(Goal):
    order = 1

    priority = 2 ** 31 - 1

    def __init__(self, optimization_problem: "_HeadLossMixin", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimization_problem = optimization_problem
        self.function_nominal = len(optimization_problem.times())

    def function(self, optimization_problem: "_HeadLossMixin", ensemble_member):
        sum_ = 0.0

        parameters = optimization_problem.parameters(ensemble_member)
        options = optimization_problem.heat_network_options()

        pumps = optimization_problem.heat_network_components.get("pump", [])

        for p in pumps:
            sum_ += optimization_problem.state(f"{p}.dH")

        assert options["head_loss_option"] != HeadLossOption.NO_HEADLOSS

        for p in optimization_problem.heat_network_components["pipe"]:
            if not parameters[f"{p}.has_control_valve"] and not parameters[f"{p}.length"] == 0.0:
                sym_name = optimization_problem._hn_pipe_to_head_loss_map[p]
                sum_ += optimization_problem.state(sym_name)

        return sum_


class _HeadLossMixin(
    BaseComponentTypeMixin, _GoalProgrammingMixinBase, CollocatedIntegratedOptimizationProblem
):
    """
    Adds handling of discharge - head (loss) relationship to the model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__pipe_head_loss_var = {}
        self.__pipe_head_loss_bounds = {}
        self.__pipe_head_loss_nominals = {}
        self.__pipe_head_loss_zero_bounds = {}
        self._hn_pipe_to_head_loss_map = {}

        self.__priority = None

    def pre(self):

        super().pre()

        self.__initialize_nominals_and_bounds()

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------+-----------+-----------------------------------+
        | Option                         | Type      | Default value                     |
        +================================+===========+===================================+
        | ``minimum_pressure_far_point`` | ``float`` | ``1.0`` bar                       |
        +--------------------------------+-----------+-----------------------------------+
        | ``wall_roughness``             | ``float`` | ``0.002`` m                       |
        +--------------------------------+-----------+-----------------------------------+
        | ``head_loss_option``           | ``enum``  | ``HeadLossOption.CQ2_INEQUALITY`` |
        +--------------------------------+-----------+-----------------------------------+
        | ``estimated_velocity``         | ``float`` | ``1.0`` m/s (CQ2_* & LINEAR)      |
        +--------------------------------+-----------+-----------------------------------+
        | ``maximum_velocity``           | ``float`` | ``2.5`` m/s (LINEARIZED_DW)       |
        +--------------------------------+-----------+-----------------------------------+
        | ``n_linearization_lines``      | ``int``   | ``5`` (LINEARIZED_DW)             |
        +--------------------------------+-----------+-----------------------------------+
        | ``minimize_head_losses``       | ``bool``  | ``True``                          |
        +--------------------------------+-----------+-----------------------------------+

        The ``minimum_pressure_far_point`` gives the minimum pressure
        requirement at any demand node, which means that the pressure at the
        furthest point is also satisfied without inspecting the topology.

        The ``wall_roughness`` of the pipes plays a role in determining the
        resistance of the pipes.

        To model the head loss in pipes, the ``head_loss_option`` refers to
        one of the ways this can be done. See :class:`HeadLossOption` for more
        explanation on what each option entails. Note that some options model
        the head loss as an inequality, i.e. :math:`\Delta H \ge f(Q)`, whereas
        others model it as an equality.

        When ``HeadLossOption.CQ2_INEQUALITY`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H \ge C
        \cdot Q^2`.

        When ``HeadLossOption.LINEARIZED_DW`` is used, the
        ``maximum_velocity`` needs to be set. The Darcy-Weisbach head loss
        relationship from :math:`v = 0` until :math:`v = maximum_velocity`
        will then be linearized using ``n_linearization`` lines.

        When ``HeadLossOption.LINEAR`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H = C \cdot
        Q`. For pipes that contain a control valve, the formulation of
        ``HeadLossOption.CQ2_INEQUALITY`` is used.

        When ``HeadLossOption.CQ2_EQUALITY`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H = C \cdot
        Q^2`. Note that this formulation is non-convex. At `theta < 1` we
        therefore use the formulation ``HeadLossOption.LINEAR``. For pipes
        that contain a control valve, the formulation of
        ``HeadLossOption.CQ2_INEQUALITY`` is used.


        When ``minimize_head_losses`` is set to True (default), a last
        priority is inserted where the head losses in the system are
        minimized if the ``head_loss_option`` is not `NO_HEADLOSS`.
        This is related to the assumption that control valves are
        present in the system to steer water in the right direction the case
        of multiple routes. If such control valves are not present, enabling
        this option will give warnings in case the found solution is not
        feasible. In case the option is False, both the minimization and
        checks are skipped.
        """

        options = {}

        options["minimum_pressure_far_point"] = 1.0
        options["wall_roughness"] = 2e-3
        options["head_loss_option"] = HeadLossOption.CQ2_INEQUALITY
        options["estimated_velocity"] = 1.0
        options["maximum_velocity"] = 2.5
        options["n_linearization_lines"] = 5
        options["minimize_head_losses"] = True

        return options

    @abstractmethod
    def _hn_get_pipe_head_loss_option(
        self, pipe, heat_network_options, parameters, **kwargs
    ) -> HeadLossOption:
        """
        The global user head loss option is not necessarily the same as the
        head loss option for a specific pipe. For example, when a control
        valve is present, a .LINEAR global head loss option could mean a
        .CQ2_INEQUALITY formulation should be used instead.

        See also the explanation of `head_loss_option` (and its values) in
        :py:meth:`.heat_network_options`.
        """
        raise NotImplementedError

    @abstractmethod
    def _hn_pipe_head_loss_constraints(self, ensemble_member) -> List[Tuple[ca.MX, float, float]]:
        """
        This method should be implemented to relate the three variables:

        - discharge: e.g. `pipe.Q`
        - head difference: e.g. `pipe.dH`
        - head loss of the pipe (note: proxy symbol that is >= abs(actual head loss))

        The internal variable name/symbol for the head loss can be retried via
        `self._hn_pipe_to_head_loss_map[pipe]`. The method that should/can be called to
        relate these three variables is :py:meth:`._hn_pipe_head_loss`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _hn_prefix(self) -> str:
        raise NotImplementedError

    @property
    def _hn_minimization_goal_class(self) -> Type[Goal]:
        return _MinimizeHeadLosses

    def __initialize_nominals_and_bounds(self):
        self.__pipe_head_loss_nominals = AliasDict(self.alias_relation)

        options = self.heat_network_options()
        parameters = self.parameters(0)

        head_loss_option = options["head_loss_option"]
        if head_loss_option not in HeadLossOption.__members__.values():
            raise Exception(f"Head loss option '{head_loss_option}' does not exist")

        for p in self.heat_network_components["pipe"]:
            length = parameters[f"{p}.length"]
            if length < 0.0:
                raise ValueError("Pipe length has to be larger than or equal to zero")

            head_loss_option = self._hn_get_pipe_head_loss_option(p, options, parameters)

            if head_loss_option == HeadLossOption.NO_HEADLOSS or (
                length == 0.0 and not parameters[f"{p}.has_control_valve"]
            ):
                self.__pipe_head_loss_zero_bounds[f"{p}.dH"] = (0.0, 0.0)
            else:
                area = 0.25 * math.pi * parameters[f"{p}.diameter"] ** 2
                q_nominal = np.array([area * options["estimated_velocity"]])
                q_max = np.array([area * options["maximum_velocity"]])
                head_loss_nominal = self._hn_pipe_head_loss(p, options, parameters, q_nominal)[0]
                head_loss_max = self._hn_pipe_head_loss(p, options, parameters, q_max)[0]

                self.__pipe_head_loss_nominals[f"{p}.dH"] = head_loss_nominal

                # The .dH is by definition "Out - In". The .__head_loss is by
                # definition larger than or equal to the absolute value of dH.
                head_loss_var = f"{p}.__head_loss"

                self._hn_pipe_to_head_loss_map[p] = head_loss_var
                self.__pipe_head_loss_var[head_loss_var] = ca.MX.sym(head_loss_var)

                self.__pipe_head_loss_nominals[head_loss_var] = head_loss_nominal
                self.__pipe_head_loss_bounds[head_loss_var] = (0.0, head_loss_max)

    def _hn_pipe_head_loss(
        self,
        pipe: str,
        heat_network_options,
        parameters,
        discharge: Union[ca.MX, np.ndarray],
        head_loss: Optional[ca.MX] = None,
        dh: Optional[ca.MX] = None,
    ) -> Union[List[Tuple[ca.MX, BT, BT]], np.ndarray]:
        """
        This function has two purposes:
        - return the head loss constraint expression(s) or
        - compute the head loss numerically (always positive).

        Note that there are different head loss formulations (see
        :class:`HeadLossOption`). Some formulations require the passing of
        `head_loss` (a symbol that is always positive by definition), and
        others require the passing of `dh` (which is negative when the flow is
        positive).

        When `head_loss` or `dh` is its corresponding MX symbol/expression,
        the appropriate constraint expression is returned. When `head_loss`
        and `dh` are both None, the `discharge` is assumed numerical, and the
        numerical computation of the appropriate head loss formulation is
        returned. Note that this returned numerical value is always positive,
        regardless of the sign of the discharge.
        """

        if head_loss is None and dh is None:
            symbolic = False
        else:
            symbolic = True

        head_loss_option = self._hn_get_pipe_head_loss_option(
            pipe, heat_network_options, parameters
        )
        assert (
            head_loss_option != HeadLossOption.NO_HEADLOSS
        ), "This method should be skipped when NO_HEADLOSS is set."

        length = parameters[f"{pipe}.length"]

        if length == 0.0:
            if not symbolic:
                return np.zeros_like(discharge)
            else:
                # dH is set to zero in bounds
                return []

        maximum_velocity = heat_network_options["maximum_velocity"]
        estimated_velocity = heat_network_options["estimated_velocity"]
        wall_roughness = heat_network_options["wall_roughness"]

        diameter = parameters[f"{pipe}.diameter"]
        area = math.pi * diameter ** 2 / 4
        temperature = parameters[f"{pipe}.temperature"]
        has_control_valve = parameters[f"{pipe}.has_control_valve"]

        if head_loss_option == HeadLossOption.LINEAR:
            assert not has_control_valve

            ff = darcy_weisbach.friction_factor(
                estimated_velocity, diameter, length, wall_roughness, temperature
            )

            # Compute c_v constant (where |dH| ~ c_v * v^2)
            c_v = length * ff / (2 * GRAVITATIONAL_CONSTANT) / diameter

            linearization_velocity = estimated_velocity
            linearization_head_loss = c_v * linearization_velocity ** 2
            linearization_discharge = linearization_velocity * area

            expr = linearization_head_loss * discharge / linearization_discharge

            if symbolic:
                q_nominal = self.variable_nominal(f"{pipe}.Q")
                head_loss_nominal = self.variable_nominal(f"{pipe}.dH")
                constraint_nominal = (
                    head_loss_nominal
                    * linearization_head_loss
                    * q_nominal
                    / linearization_discharge
                ) ** 0.5
                return [((-1 * dh - expr) / constraint_nominal, 0.0, 0.0)]
            else:
                return expr * np.sign(discharge)

        elif head_loss_option in {
            HeadLossOption.CQ2_INEQUALITY,
            HeadLossOption.CQ2_EQUALITY,
        }:
            ff = darcy_weisbach.friction_factor(
                estimated_velocity, diameter, length, wall_roughness, temperature
            )

            # Compute c_v constant (where |dH| ~ c_v * v^2)
            c_v = length * ff / (2 * GRAVITATIONAL_CONSTANT) / diameter

            v = discharge / area
            expr = c_v * v ** 2

            if symbolic:
                if head_loss_option == HeadLossOption.CQ2_INEQUALITY:
                    ub = np.inf
                else:
                    ub = 0.0

                q_nominal = self.variable_nominal(f"{pipe}.Q")
                head_loss_nominal = self.variable_nominal(f"{pipe}.dH")
                constraint_nominal = (head_loss_nominal * c_v * (q_nominal / area) ** 2) ** 0.5
                return [((head_loss - expr) / constraint_nominal, 0.0, ub)]
            else:
                return expr

        elif head_loss_option == HeadLossOption.LINEARIZED_DW:
            n_lines = heat_network_options["n_linearization_lines"]

            a, b = darcy_weisbach.get_linear_pipe_dh_vs_q_fit(
                diameter,
                length,
                wall_roughness,
                temperature=temperature,
                n_lines=n_lines,
                v_max=maximum_velocity,
            )

            # The function above only gives result in the positive quadrant
            # (positive head loss, positive discharge). We also need a
            # positive head loss for _negative_ discharges.
            a = np.hstack([-a, a])
            b = np.hstack([b, b])

            # Vectorize constraint for speed
            if symbolic:
                q_nominal = self.variable_nominal(f"{pipe}.Q")
                head_loss_nominal = self.variable_nominal(f"{pipe}.dH")
                head_loss_vec = ca.repmat(head_loss, len(a))
                discharge_vec = ca.repmat(discharge, len(a))

                a_vec = np.repeat(a, discharge.size1())
                b_vec = np.repeat(b, discharge.size1())
                constraint_nominal = np.abs(head_loss_nominal * a_vec * q_nominal) ** 0.5

                return [
                    (
                        (head_loss_vec - (a_vec * discharge_vec + b_vec)) / constraint_nominal,
                        0.0,
                        np.inf,
                    )
                ]
            else:
                return np.amax(a * np.tile(discharge, (len(a), 1)).transpose() + b, axis=1)

    def __pipe_head_loss_path_constraints(self, _ensemble_member):
        constraints = []

        pref = self._hn_prefix

        # We set this constraint relating .dH to the upstream and downstream
        # heads here in the Mixin for scaling purposes (dH nominal is
        # calculated in pre()).
        for pipe in self.heat_network_components["pipe"]:
            dh = self.state(f"{pipe}.dH")
            h_down = self.state(f"{pipe}.{pref}Out.H")
            h_up = self.state(f"{pipe}.{pref}In.H")

            constraint_nominal = (
                self.variable_nominal(f"{pipe}.dH") * self.variable_nominal(f"{pipe}.{pref}In.H")
            ) ** 0.5
            constraints.append(((dh - (h_down - h_up)) / constraint_nominal, 0.0, 0.0))

        return constraints

    def __source_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components

        pref = self._hn_prefix

        for source in components["source"]:
            c = parameters[f"{source}.head_loss"]

            if c == 0.0:
                constraints.append(
                    (
                        self.state(f"{source}.{pref}In.H") - self.state(f"{source}.{pref}Out.H"),
                        0.0,
                        0.0,
                    )
                )
            else:
                constraints.append(
                    (
                        self.state(f"{source}.{pref}In.H")
                        - self.state(f"{source}.{pref}Out.H")
                        - c * self.state(f"{source}.{pref}In.Q") ** 2,
                        0.0,
                        np.inf,
                    )
                )

        return constraints

    def __demand_head_loss_path_constraints(self, _ensemble_member):
        constraints = []

        options = self.heat_network_options()
        components = self.heat_network_components

        pref = self._hn_prefix

        # Convert minimum pressure at far point from bar to meter (water) head
        min_head_loss = options["minimum_pressure_far_point"] * 10.2

        for d in components["demand"]:
            constraints.append(
                (
                    self.state(f"{d}.{pref}In.H") - self.state(f"{d}.{pref}Out.H"),
                    min_head_loss,
                    np.inf,
                )
            )

        return constraints

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        if self.heat_network_options()["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            constraints.extend(self._hn_pipe_head_loss_constraints(ensemble_member))

        return constraints

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member).copy()

        options = self.heat_network_options()

        # Add source/demand head loss constrains only if head loss is non-zero
        if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            constraints.extend(self.__pipe_head_loss_path_constraints(ensemble_member))
            constraints.extend(self.__source_head_loss_path_constraints(ensemble_member))
            constraints.extend(self.__demand_head_loss_path_constraints(ensemble_member))

        return constraints

    def priority_started(self, priority):
        super().priority_started(priority)
        self.__priority = priority

    def priority_completed(self, priority):
        super().priority_completed(priority)

        options = self.heat_network_options()

        pref = self._hn_prefix

        if (
            options["minimize_head_losses"]
            and options["head_loss_option"] != HeadLossOption.NO_HEADLOSS
            and priority == self._hn_minimization_goal_class.priority
        ):
            components = self.heat_network_components

            rtol = 1e-5
            atol = 1e-4

            for ensemble_member in range(self.ensemble_size):
                parameters = self.parameters(ensemble_member)
                results = self.extract_results(ensemble_member)

                for pipe in components["pipe"]:
                    if parameters[f"{pipe}.has_control_valve"]:
                        continue

                    # Just like with a control valve, if pipe is disconnected
                    # there is nothing to check.
                    q_full = results[f"{pipe}.Q"]
                    if parameters[f"{pipe}.disconnectable"]:
                        inds = q_full != 0.0
                    else:
                        inds = np.arange(len(q_full), dtype=int)

                    q = results[f"{pipe}.Q"][inds]
                    head_loss = results[self._hn_pipe_to_head_loss_map[pipe]][inds]
                    head_loss_target = self._hn_pipe_head_loss(pipe, options, parameters, q, None)

                    if not np.allclose(head_loss, head_loss_target, rtol=rtol, atol=atol):
                        logger.warning(
                            f"Pipe {pipe} has artificial head loss; "
                            f"at least one more control valve should be added to the network."
                        )

                for source in components["source"]:
                    c = parameters[f"{source}.head_loss"]
                    head_loss = results[f"{source}.{pref}In.H"] - results[f"{source}.{pref}Out.H"]
                    head_loss_target = c * results[f"{source}.{pref}In.Q"] ** 2

                    if not np.allclose(head_loss, head_loss_target, rtol=rtol, atol=atol):
                        logger.warning(f"Source {source} has artificial head loss.")

                min_head_loss_target = options["minimum_pressure_far_point"] * 10.2
                min_head_loss = None

                for demand in components["demand"]:
                    head_loss = results[f"{demand}.{pref}In.H"] - results[f"{demand}.{pref}Out.H"]
                    if min_head_loss is None:
                        min_head_loss = head_loss
                    else:
                        min_head_loss = np.minimum(min_head_loss, head_loss)

                if not np.allclose(min_head_loss, min_head_loss_target, rtol=rtol, atol=atol):
                    logger.warning("Minimum head at demands is higher than target minimum.")

    def path_goals(self):
        g = super().path_goals().copy()

        options = self.heat_network_options()
        if (
            options["minimize_head_losses"]
            and options["head_loss_option"] != HeadLossOption.NO_HEADLOSS
        ):
            g.append(self._hn_minimization_goal_class(self))

        return g

    @property
    def path_variables(self):
        variables = super().path_variables.copy()
        variables.extend(self.__pipe_head_loss_var.values())
        return variables

    def variable_nominal(self, variable):
        try:
            return self.__pipe_head_loss_nominals[variable]
        except KeyError:
            return super().variable_nominal(variable)

    def bounds(self):
        bounds = super().bounds().copy()

        bounds.update(self.__pipe_head_loss_bounds)
        bounds.update(self.__pipe_head_loss_zero_bounds)

        return bounds
