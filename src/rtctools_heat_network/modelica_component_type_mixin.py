from typing import Dict

from .base_component_type_mixin import BaseComponentTypeMixin
from .heat_network_common import NodeConnectionDirection
from .topology import Topology


class ModelicaComponentTypeMixin(BaseComponentTypeMixin):
    def pre(self):
        components = self.heat_network_components
        nodes = components["node"]
        pipes = components["pipe"]

        # Figure out which pipes are connected to which nodes, and which pipes
        # are connected in series.
        pipes_set = set(pipes)
        parameters = [self.parameters(e) for e in range(self.ensemble_size)]
        node_connections = {}

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
                cur_port = f"{n}.QTHConn[{i + 1}]"
                aliases = [
                    x for x in self.alias_relation.aliases(f"{cur_port}.T") if not x.startswith(n)
                ]

                if len(aliases) > 1:
                    raise Exception(f"More than one connection to {cur_port}")
                elif len(aliases) == 0:
                    raise Exception(f"Found no connection to {cur_port}")

                if aliases[0].endswith(".QTHOut.T"):
                    pipe_w_orientation = (aliases[0][:-9], NodeConnectionDirection.IN)
                else:
                    assert aliases[0].endswith(".QTHIn.T")
                    pipe_w_orientation = (aliases[0][:-8], NodeConnectionDirection.OUT)

                assert pipe_w_orientation[0] in pipes_set

                connected_pipes[i] = pipe_w_orientation

        # Note that a pipe series can include both hot and cold pipes. It is
        # only about figuring out which pipes are related direction-wise.
        canonical_pipe_qs = {p: self.alias_relation.canonical_signed(f"{p}.Q") for p in pipes}
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

        self.__topology = Topology(node_connections, pipe_series)

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
