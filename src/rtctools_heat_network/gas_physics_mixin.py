import logging

import esdl

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)

from .base_component_type_mixin import BaseComponentTypeMixin
from .head_loss_class import HeadLossClass, HeadLossOption


logger = logging.getLogger("rtctools_heat_network")


class GasPhysicsMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    __allowed_head_loss_options = {
        HeadLossOption.NO_HEADLOSS,
        HeadLossOption.LINEAR,
        HeadLossOption.LINEARIZED_DW,
    }
    """
    This class is used to model the physics of a gas network with its assets. We model
    the different components with variety of linearization strategies.
    """

    def __init__(self, *args, **kwargs):
        """
        In this __init__ we prepare the dicts for the variables added by the HeatMixin class
        """

        self._head_loss_class = HeadLossClass("gas_network")

        self.__gas_pipe_head_bounds = {}

        self.__gas_pipe_head_loss_var = {}
        self.__gas_pipe_head_loss_bounds = {}
        self.__gas_pipe_head_loss_nominals = {}
        self.__gas_pipe_head_loss_zero_bounds = {}
        self._hn_gas_pipe__to_head_loss_map = {}

        # Boolean path-variable for the direction of the flow, inport to outport is positive flow.
        self.__flow_direct_var = {}
        self.__flow_direct_bounds = {}
        self._pipe_to_flow_direct_map = {}

        super().__init__(*args, **kwargs)

    def pre(self):
        """
        In this pre method we fill the dicts initiated in the __init__. This means that we create
        the Casadi variables and determine the bounds, nominals and create maps for easier
        retrieving of the variables.
        """
        super().pre()

        options = self.heat_network_options()
        parameters = self.parameters(0)

        bounds = self.bounds()

        for pipe_name in self.heat_network_components.get("gas_pipe", []):
            if isinstance(
                self.esdl_assets[self.esdl_asset_name_to_id_map[pipe_name]].in_ports[0].carrier
                , esdl.GasCommodity
            ):
                commodity_type = self.esdl_assets[self.esdl_asset_name_to_id_map[pipe_name]].in_ports[0].carrier
                (
                    self.__gas_pipe_head_bounds,
                    self.__gas_pipe_head_loss_zero_bounds,
                    self._hn_gas_pipe__to_head_loss_map,
                    self.__gas_pipe_head_loss_var,
                    self.__gas_pipe_head_loss_nominals,
                    self.__gas_pipe_head_loss_bounds,
                ) = self._head_loss_class.initialize_variables_nominals_and_bounds(
                        self, commodity_type, pipe_name
                )
                temp2 = 12.0

            temp = 0.0

        self.__maximum_total_head_loss = self.__get_maximum_total_head_loss()

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------------+-----------+-----------------------------+
        | Option                               | Type      | Default value               |
        +======================================+===========+=============================+
        | ``minimum_pressure_far_point``       | ``float`` | ``1.0`` bar                 |
        +--------------------------------------+-----------+-----------------------------+
        +--------------------------------------+-----------+-----------------------------+
        | ``minimum_velocity``                 | ``float`` | ``0.005`` m/s               |
        +--------------------------------------+-----------+-----------------------------+
        | ``head_loss_option`` (inherited)     | ``enum``  | ``HeadLossOption.LINEAR``   |
        +--------------------------------------+-----------+-----------------------------+
        | ``minimize_head_losses`` (inherited) | ``bool``  | ``False``                   |
        +--------------------------------------+-----------+-----------------------------+
        ....
        """

        options = self._head_loss_class.head_loss_network_options()

        options["minimum_velocity"] = 0.005
        options["minimum_pressure_far_point"] = 1.0
        options["head_loss_option"] = HeadLossOption.LINEAR
        options["minimize_head_losses"] = False

        return options

    @property
    def extra_variables(self):
        """
        In this function we add all the variables defined in the HeatMixin to the optimization
        problem. Note that these are only the normal variables not path variables.
        """
        variables = super().extra_variables.copy()

        return variables

    @property
    def path_variables(self):
        """
        In this function we add all the path variables defined in the HeatMixin to the
        optimization problem. Note that path_variables are variables that are created for each
        time-step.
        """
        variables = super().path_variables.copy()
        variables.extend(self.__gas_pipe_head_loss_var.values())

        return variables

    def variable_is_discrete(self, variable):
        """
        All variables that only can take integer values should be added to this function.
        """
        # some binary var still to be added for equality constraints
        # if (
        #     or variable in self.__demand_insulation_class_var
        # ):
        #     return True
        # else:
        #     return super().variable_is_discrete(variable)

        return super().variable_is_discrete(variable)

    def variable_nominal(self, variable):
        """
        In this function we add all the nominals for the variables defined/added in the HeatMixin.
        """

        if variable in self.__gas_pipe_head_loss_nominals:
            return self.__gas_pipe_head_loss_nominals[variable]
        else:
            return super().variable_nominal(variable)

    def bounds(self):
        """
        In this function we add the bounds to the problem for all the variables defined/added in
        the HeatMixin.
        """
        bounds = super().bounds()

        bounds.update(self.__gas_pipe_head_loss_bounds)
        bounds.update(self.__gas_pipe_head_loss_zero_bounds)

        for k, v in self.__gas_pipe_head_bounds.items():
            bounds[k] = self.merge_bounds(bounds[k], v)

        return bounds

    def path_goals(self):
        """
        Here we add the goals for minimizing the head loss and hydraulic power depending on the
        configuration. Please note that we only do hydraulic power for the MILP problem thus only
        for the linearized head_loss options.
        """
        g = super().path_goals().copy()

        options = self.heat_network_options()
        if (
            options["minimize_head_losses"]
            and options["head_loss_option"] != HeadLossOption.NO_HEADLOSS
        ):
            g.append(self._head_loss_class._hn_minimization_goal_class(self, "gas_network"))

        return g

    def __get_maximum_total_head_loss(self):
        """
        Get an upper bound on the maximum total head loss that can be used in
        big-M formulations of e.g. check valves and disconnectable pipes.

        There are multiple ways to calculate this upper bound, depending on
        what options are set. We compute all these upper bounds, and return
        the lowest one of them.
        """

        options = self.heat_network_options()
        components = self.heat_network_components

        if options["head_loss_option"] == HeadLossOption.NO_HEADLOSS:
            # Undefined, and all constraints using this methods value should
            # be skipped.
            return np.nan

        # Summing head loss in pipes
        max_sum_dh_pipes = 0.0

        for ensemble_member in range(self.ensemble_size):
            parameters = self.parameters(ensemble_member)

            head_loss = 0.0

            pipe_type = "pipe"
            if len(components.get("pipe", [])) == 0:
                pipe_type = "gas_pipe"

            for pipe in components.get(pipe_type, []):
                area = parameters[f"{pipe}.area"]
                max_discharge = options["maximum_velocity"] * area
                head_loss += self._head_loss_class._hn_pipe_head_loss(
                    pipe, self, options, parameters, max_discharge
                )

            head_loss += options["minimum_pressure_far_point"] * 10.2

            max_sum_dh_pipes = max(max_sum_dh_pipes, head_loss)

        # Maximum pressure difference allowed with user options
        # NOTE: Does not yet take elevation differences into acccount
        max_dh_network_options = (
            options["pipe_maximum_pressure"] - options["pipe_minimum_pressure"]
        ) * 10.2

        return min(max_sum_dh_pipes, max_dh_network_options)

    def __gas_node_heat_mixing_path_constraints(self, ensemble_member):
        """
        This function adds constraints for each gas network node/joint to have as much
        flow going in as out. Effectively, it is setting the sum of flow to zero.
        """
        constraints = []

        for node, connected_pipes in self.heat_network_topology.gas_nodes.items():
            q_sum = 0.0
            q_nominals = []

            for i_conn, (_pipe, orientation) in connected_pipes.items():
                gas_conn = f"{node}.GasConn[{i_conn + 1}].Q"
                q_sum += orientation * self.state(gas_conn)
                q_nominals.append(self.variable_nominal(gas_conn))

            q_nominal = np.median(q_nominals)
            constraints.append((q_sum / q_nominal, 0.0, 0.0))

            q_sum = 0.0
            q_nominals = []

            for i_conn, (_pipe, orientation) in connected_pipes.items():
                gas_conn = f"{node}.GasConn[{i_conn + 1}].Q_shadow"
                q_sum += orientation * self.state(gas_conn)
                q_nominals.append(self.variable_nominal(gas_conn))

            q_nominal = np.median(q_nominals)
            constraints.append((q_sum / q_nominal, 0.0, 0.0))

        return constraints

    def __state_vector_scaled(self, variable, ensemble_member):
        """
        This functions returns the casadi symbols scaled with their nominal for the entire time
        horizon.
        """
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    def _hn_gas_pipe__head_loss_constraints(self, ensemble_member):
        """
        This function adds the head loss constraints for pipes. There are two options namely with
        and without pipe class optimization. In both cases we assume that disconnected pipes, pipes
        without flow have no head loss.

        Under pipe-class optimization the head loss constraints per pipe class are added and
        applied with the big_m method (is_topo_disconnected) to only activate the correct
        constraints.

        Under constant pipe class constraints for only one diameter are added.
        """
        constraints = []

        options = self.heat_network_options()
        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components
        # Set the head loss according to the direction in the pipes. Note that
        # the `.__head_loss` symbol is always positive by definition, but that
        # `.dH` is not (positive when flow is negative, and vice versa).
        # If the pipe is disconnected, we leave the .__head_loss symbol free
        # (and it has no physical meaning). We also do not set any discharge
        # relationship in this case (but dH is still equal to Out - In of
        # course).

        for pipe in components.get("gas_pipe", []):
            if parameters[f"{pipe}.length"] == 0.0:
                # If the pipe does not have a control valve, the head loss is
                # forced to zero via bounds. If the pipe _does_ have a control
                # valve, then there still is no relationship between the
                # discharge and the head loss/dH.
                continue

            head_loss_sym = self._hn_gas_pipe__to_head_loss_map[pipe]

            dh = self.__state_vector_scaled(f"{pipe}.dH", ensemble_member)
            head_loss = self.__state_vector_scaled(head_loss_sym, ensemble_member)
            discharge = self.__state_vector_scaled(f"{pipe}.Q", ensemble_member)

            # We need to make sure the dH is decoupled from the discharge when
            # the pipe is disconnected. Simply put, this means making the
            # below constraints trivial.
            is_disconnected_var = self._pipe_disconnect_map.get(pipe)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.__state_vector_scaled(is_disconnected_var, ensemble_member)

            max_discharge = None
            max_head_loss = -np.inf

            if pipe in self._pipe_topo_pipe_class_map:
                # Multiple diameter options for this pipe
                pipe_classes = self._pipe_topo_pipe_class_map[pipe]
                max_discharge = max(c.maximum_discharge for c in pipe_classes)

                for pc, pc_var_name in pipe_classes.items():
                    if pc.inner_diameter == 0.0:
                        continue

                    head_loss_max_discharge = self._head_loss_class._hn_pipe_head_loss(
                        pipe, self, options, parameters, max_discharge, pipe_class=pc
                    )

                    big_m = max(1.1 * self.__maximum_total_head_loss, 2 * head_loss_max_discharge)

                    is_topo_disconnected = 1 - self.extra_variable(pc_var_name, ensemble_member)
                    is_topo_disconnected = ca.repmat(is_topo_disconnected, dh.size1())

                    # Note that we add the two booleans `is_disconnected` and
                    # `is_topo_disconnected`. This is allowed because of the way the
                    # resulting expression is used in the Big-M formulation. We only care
                    # that the expression (i.e. a single boolean or the sum of the two
                    # booleans) is either 0 when the pipe is connected, or >= 1 when it
                    # is disconnected.
                    constraints.extend(
                        self._head_loss_class._hn_pipe_head_loss(
                            pipe,
                            self,
                            options,
                            parameters,
                            discharge,
                            head_loss,
                            dh,
                            is_disconnected + is_topo_disconnected,
                            big_m,
                            pc,
                        )
                    )

                    # Contrary to the Big-M calculation above, the relation
                    # between dH and the head loss symbol requires the
                    # maximum head loss that can be realized effectively. So
                    # we pass the current pipe class's maximum discharge.
                    max_head_loss = max(
                        max_head_loss,
                        self._head_loss_class._hn_pipe_head_loss(
                            pipe, self, options, parameters, pc.maximum_discharge, pipe_class=pc
                        ),
                    )
            else:
                # Only a single diameter for this pipe. Note that we rely on
                # the diameter parameter being overridden automatically if a
                # single pipe class is set by the user.
                area = parameters[f"{pipe}.area"]
                max_discharge = options["maximum_velocity"] * area

                is_topo_disconnected = int(parameters[f"{pipe}.diameter"] == 0.0)

                constraints.extend(
                    self._head_loss_class._hn_pipe_head_loss(
                        pipe,
                        self,
                        options,
                        parameters,
                        discharge,
                        head_loss,
                        dh,
                        is_disconnected + is_topo_disconnected,
                        1.1 * self.__maximum_total_head_loss,
                    )
                )

                max_head_loss = self._head_loss_class._hn_pipe_head_loss(
                    pipe, self, options, parameters, max_discharge
                )

            # Relate the head loss symbol to the pipe's dH symbol.

            # FIXME: Ugly hack. Cold pipes should be modelled completely with
            # their own integers as well.
            # kvr still to add
            # flow_dir = self.__state_vector_scaled(
            #     self._pipe_to_flow_direct_map[pipe], ensemble_member
            # )
            # flow_dir = 1.0

            # Note that the Big-M should _at least_ cover the maximum
            # distance between `head_loss` and `dh`. If `head_loss` can be at
            # most 1.0 (= `max_head_loss`), that means our Big-M should be at
            # least double (i.e. >= 2.0). And because we do not want Big-Ms to
            # be overly tight, we include an additional factor of 2.
            # big_m = 2 * 2 * max_head_loss

            # constraints.append(
            #     (
            #         (-dh - head_loss + (1 - flow_dir) * big_m) / big_m,
            #         0.0,
            #         np.inf,
            #     )
            # )
            # constraints.append(((dh - head_loss + flow_dir * big_m) / big_m, 0.0, np.inf))

        return constraints

    def path_constraints(self, ensemble_member):
        """
        Here we add all the path constraints to the optimization problem. Please note that the
        path constraints are the constraints that are applied to each time-step in the problem.
        """

        constraints = super().path_constraints(ensemble_member)

        options = self.heat_network_options()

        constraints.extend(self.__gas_node_heat_mixing_path_constraints(ensemble_member))

        # Add source/demand head loss constrains only if head loss is non-zero
        if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            constraints.extend(
                self._head_loss_class._pipe_head_loss_path_constraints(self, ensemble_member)
            )

        return constraints

    def constraints(self, ensemble_member):
        """
        This function adds the normal constraints to the problem. Unlike the path constraints these
        are not applied to every time-step in the problem. Meaning that these constraints either
        consider global variables that are independent of time-step or that the relevant time-steps
        are indexed within the constraint formulation.
        """
        constraints = super().constraints(ensemble_member)

        options = self.heat_network_options()

        if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            constraints.extend(self._hn_gas_pipe__head_loss_constraints(ensemble_member))

        return constraints

    def goal_programming_options(self):
        """
        Here we set the goal programming configuration. We use soft constraints for consecutive
        goals.
        """
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        return options

    def solver_options(self):
        """
        Here we define the solver options. By default we use the open-source solver cbc and casadi
        solver qpsol.
        """
        options = super().solver_options()
        options["casadi_solver"] = "qpsol"
        options["solver"] = "highs"
        return options

    def compiler_options(self):
        """
        In this function we set the compiler configuration.
        """
        options = super().compiler_options()
        options["resolve_parameter_values"] = True
        return options

    def priority_completed(self, priority):
        """
        This function is called after a priority of goals is completed. This function is used to
        specify operations between consecutive goals. Here we set some parameter attributes after
        the optimization is completed.
        """
        options = self.heat_network_options()

        if (
            options["minimize_head_losses"]
            and options["head_loss_option"] != HeadLossOption.NO_HEADLOSS
            and priority == self._head_loss_class._hn_minimization_goal_class.priority
        ):
            components = self.heat_network_components

            rtol = 1e-5
            atol = 1e-4

            for ensemble_member in range(self.ensemble_size):
                parameters = self.parameters(ensemble_member)
                results = self.extract_results(ensemble_member)

                for pipe in components.get("gas_pipe", []):
                    # if parameters[f"{pipe}.has_control_valve"]: # not used at all
                    #     continue

                    # Just like with a control valve, if pipe is disconnected
                    # there is nothing to check.
                    q_full = results[f"{pipe}.Q"]
                    # if parameters[f"{pipe}.disconnectable"]: # not used yet
                    #     inds = q_full != 0.0
                    # else:
                    #     inds = np.arange(len(q_full), dtype=int)
                    inds = np.arange(len(q_full), dtype=int)

                    if parameters[f"{pipe}.diameter"] == 0.0:
                        # Pipe is disconnected. Head loss is free, so nothing to check.
                        continue

                    q = results[f"{pipe}.Q"][inds]
                    head_loss_target = self._head_loss_class._hn_pipe_head_loss(
                        pipe, self, options, parameters, q, None
                    )
                    if options["head_loss_option"] == HeadLossOption.LINEAR:
                        head_loss = np.abs(results[f"{pipe}.dH"][inds])
                    else:
                        head_loss = results[self._hn_gas_pipe__to_head_loss_map[pipe]][inds]

                    if not np.allclose(head_loss, head_loss_target, rtol=rtol, atol=atol):
                        logger.warning(
                            f"Pipe {pipe} has artificial head loss; "
                            f"at least one more control valve should be added to the network."
                        )

        super().priority_completed(priority)
