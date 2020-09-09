from abc import abstractmethod
from typing import Dict

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)

from .topology import Topology


class BaseComponentTypeMixin(CollocatedIntegratedOptimizationProblem):
    @property
    @abstractmethod
    def heat_network_components(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def heat_network_topology(self) -> Topology:
        raise NotImplementedError
