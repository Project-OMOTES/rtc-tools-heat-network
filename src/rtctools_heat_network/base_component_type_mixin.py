from abc import abstractmethod
from typing import Dict, List

from .topology import Topology


class BaseComponentTypeMixin:
    """
    The standard naming convention is that pipes have "_hot" and "_cold" suffixes.
    Such convention can be overridden using the `is_hot_pipe` and `is_cold_pipe` methods.
    Moreover, one has to set the mapping between hot and cold pipes via `hot_to_cold_pipe`
    and `cold_to_hot_pipe`.
    """

    @property
    @abstractmethod
    def heat_network_components(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def heat_network_topology(self) -> Topology:
        raise NotImplementedError

    def is_hot_pipe(self, pipe: str) -> bool:
        return pipe.endswith("_hot")

    def is_cold_pipe(self, pipe: str) -> bool:
        return pipe.endswith("_cold")

    def hot_to_cold_pipe(self, pipe: str):
        assert self.is_hot_pipe(pipe)

        return f"{pipe[:-4]}_cold"

    def cold_to_hot_pipe(self, pipe: str):
        assert self.is_cold_pipe(pipe)

        return f"{pipe[:-5]}_hot"

    @property
    def hot_pipes(self) -> List[str]:
        return [p for p in self.heat_network_components.get("pipe", []) if self.is_hot_pipe(p)]

    @property
    def cold_pipes(self) -> List[str]:
        return [p for p in self.heat_network_components.get("pipe", []) if self.is_cold_pipe(p)]
