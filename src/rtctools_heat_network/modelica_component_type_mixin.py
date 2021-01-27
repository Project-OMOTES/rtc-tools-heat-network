from typing import Dict

from pymoca.backends.casadi.alias_relation import AliasRelation

from .base_component_type_mixin import BaseComponentTypeMixin
from .heat_network_common import NodeConnectionDirection
from .topology import Topology


class ModelicaComponentTypeMixin(BaseComponentTypeMixin):
    def pre(self):
        components = self.heat_network_components
        nodes = components.get("node", [])
        pipes = components["pipe"]
        buffers = components.get("buffer", [])

        # Figure out which pipes are connected to which nodes, which pipes
        # are connected in series, and which pipes are connected to which buffers.

        pipes_set = set(pipes)
        parameters = [self.parameters(e) for e in range(self.ensemble_size)]
        node_connections = {}

        # Figure out if we are dealing with a Heat model, or a QTH model
        try:
            _ = self.variable(f"{pipes[0]}.HeatIn.Heat")
            heat_network_model_type = "Heat"
        except KeyError:
            heat_network_model_type = "QTH"

        for n in nodes:
            n_connections = [ens_params[f"{n}.n"] for ens_params in parameters]

            if len(set(n_connections)) > 1:
                raise Exception(
                    "Nodes cannot have differing number of connections per ensemble member"
                )

            n_connections = n_connections[0]

            # Note that we do this based on temperature, because discharge may
            # be an alias of yet some other further away connected pipe.
            node_connections[n] = connected_pipes = {}

            for i in range(n_connections):
                cur_port = f"{n}.{heat_network_model_type}Conn[{i + 1}]"
                prop = "T" if heat_network_model_type == "QTH" else "Heat"
                aliases = [
                    x
                    for x in self.alias_relation.aliases(f"{cur_port}.{prop}")
                    if not x.startswith(n) and x.endswith(f".{prop}")
                ]

                if len(aliases) > 1:
                    raise Exception(f"More than one connection to {cur_port}")
                elif len(aliases) == 0:
                    raise Exception(f"Found no connection to {cur_port}")

                in_suffix = ".QTHIn.T" if heat_network_model_type == "QTH" else ".HeatIn.Heat"
                out_suffix = ".QTHOut.T" if heat_network_model_type == "QTH" else ".HeatOut.Heat"

                if aliases[0].endswith(out_suffix):
                    pipe_w_orientation = (
                        aliases[0][: -len(out_suffix)],
                        NodeConnectionDirection.IN,
                    )
                else:
                    assert aliases[0].endswith(in_suffix)
                    pipe_w_orientation = (
                        aliases[0][: -len(in_suffix)],
                        NodeConnectionDirection.OUT,
                    )

                assert pipe_w_orientation[0] in pipes_set

                connected_pipes[i] = pipe_w_orientation

        # Note that a pipe series can include both hot and cold pipes for
        # QTH models. It is only about figuring out which pipes are
        # related direction-wise.
        # For Heat models, only hot pipes are allowed to be part of pipe
        # series, as the cold part is zero heat by construction.
        if heat_network_model_type == "QTH":
            alias_relation = self.alias_relation
        elif heat_network_model_type == "Heat":
            # There is no proper AliasRelation yet (because there is heat loss in pipes).
            # So we build one, as that is the easiest way to figure out which pipes are
            # connected to each other in series. We do this by making a temporary/shadow
            # discharge (".Q") variable per pipe, as that way we can share the processing
            # logic for determining pipe series with that of QTH models.
            alias_relation = AliasRelation()

            # Look for aliases only in the hot pipes. All cold pipes are zero by convention anyway.
            hot_pipes = self.hot_pipes.copy()

            pipes_map = {f"{pipe}.HeatIn.Heat": pipe for pipe in hot_pipes}
            pipes_map.update({f"{pipe}.HeatOut.Heat": pipe for pipe in hot_pipes})

            for p in hot_pipes:
                for port in ["In", "Out"]:
                    heat_port = f"{p}.Heat{port}.Heat"
                    connected = self.alias_relation.aliases(heat_port).intersection(
                        pipes_map.keys()
                    )
                    connected.remove(heat_port)

                    if connected:
                        other_pipe_port = next(iter(connected))
                        if other_pipe_port.endswith(f".Heat{port}.Heat"):
                            sign_prefix = "-"
                        else:
                            sign_prefix = ""
                        other_pipe = pipes_map[other_pipe_port]
                        if f"{other_pipe}.Q" not in alias_relation.canonical_variables:
                            alias_relation.add(f"{p}.Q", f"{sign_prefix}{other_pipe}.Q")

        canonical_pipe_qs = {p: alias_relation.canonical_signed(f"{p}.Q") for p in pipes}
        # Move sign from canonical to alias
        canonical_pipe_qs = {(p, d): c for p, (c, d) in canonical_pipe_qs.items()}
        # Reverse the dictionary from `Dict[alias, canonical]` to `Dict[canonical, Set[alias]]`
        pipe_sets = {}
        for a, c in canonical_pipe_qs.items():
            pipe_sets.setdefault(c, []).append(a)

        pipe_series_with_orientation = list(pipe_sets.values())

        # Check that all pipes in the series have the same orientation
        pipe_series = []
        for ps in pipe_series_with_orientation:
            if not len({orientation for _, orientation in ps}) == 1:
                raise Exception(f"Pipes in series {ps} do not all have the same orientation")
            pipe_series.append([name for name, _ in ps])

        buffer_connections = {}

        for b in buffers:
            buffer_connections[b] = []

            for k in ["In", "Out"]:

                b_conn = f"{b}.{heat_network_model_type}{k}"
                prop = "T" if heat_network_model_type == "QTH" else "Heat"
                aliases = [
                    x
                    for x in self.alias_relation.aliases(f"{b_conn}.{prop}")
                    if not x.startswith(b) and x.endswith(f".{prop}")
                ]

                if len(aliases) > 1:
                    raise Exception(f"More than one connection to {b_conn}")
                elif len(aliases) == 0:
                    raise Exception(f"Found no connection to {b_conn}")

                in_suffix = ".QTHIn.T" if heat_network_model_type == "QTH" else ".HeatIn.Heat"
                out_suffix = ".QTHOut.T" if heat_network_model_type == "QTH" else ".HeatOut.Heat"

                if aliases[0].endswith(out_suffix):
                    pipe_w_orientation = (
                        aliases[0][: -len(out_suffix)],
                        NodeConnectionDirection.IN,
                    )
                else:
                    assert aliases[0].endswith(in_suffix)
                    pipe_w_orientation = (
                        aliases[0][: -len(in_suffix)],
                        NodeConnectionDirection.OUT,
                    )

                assert pipe_w_orientation[0] in pipes_set

                if k == "In":
                    assert self.is_hot_pipe(pipe_w_orientation[0])
                else:
                    assert self.is_cold_pipe(pipe_w_orientation[0])

                buffer_connections[b].append(pipe_w_orientation)

            buffer_connections[b] = tuple(buffer_connections[b])

        self.__topology = Topology(node_connections, pipe_series, buffer_connections)

        super().pre()

    @property
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

    @property
    def heat_network_topology(self) -> Topology:
        return self.__topology
