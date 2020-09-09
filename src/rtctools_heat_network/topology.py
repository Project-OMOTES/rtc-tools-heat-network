from typing import Dict, List, Tuple

from .heat_network_common import NodeConnectionDirection


class Topology:
    def __init__(self, nodes=None, pipe_series=None):
        if nodes is not None:
            self._nodes = nodes
        if pipe_series is not None:
            self._pipe_series = pipe_series

    @property
    def nodes(self) -> Dict[str, Dict[int, Tuple[str, NodeConnectionDirection]]]:
        """
        Maps a node name to a dictionary of its connections. Written out using
        descriptive variable names the return type would be:
            Dict[node_name, Dict[connection_index, Tuple[connected_pipe_name, pipe_orientation]]]
        """
        try:
            return self._nodes
        except AttributeError:
            raise NotImplementedError

    @property
    def pipe_series(self) -> List[List[str]]:
        """
        Return a list of a pipe series (which itself is a list of pipe names
        per serie). Note that all pipes in a series should have the same
        orientation.
        """
        try:
            return self._pipe_series
        except AttributeError:
            raise NotImplementedError
