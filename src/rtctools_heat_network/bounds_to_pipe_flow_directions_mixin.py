from typing import Dict

import numpy as np

from rtctools.optimization.timeseries import Timeseries

from .base_component_type_mixin import BaseComponentTypeMixin
from .heat_network_common import PipeFlowDirection


class BoundsToPipeFlowDirectionsMixin(BaseComponentTypeMixin):
    def pre(self):
        super().pre()

        bounds = self.bounds()
        components = self.heat_network_components
        pipes = components["pipe"]

        # Determine implied pipe directions from model bounds (that are
        # already available at this time)
        self.__implied_directions = [{} for e in range(self.ensemble_size)]

        for e in range(self.ensemble_size):
            for p in pipes:
                lb, ub = bounds[f"{p}.Q"]

                if not isinstance(lb, float) or not isinstance(ub, float):
                    raise ValueError(
                        "`BoundsToPipeFlowDirectionsMixin` only works for scalar bounds"
                    )

                if lb == ub and lb == 0.0:
                    # Pipe is disabled
                    self.__implied_directions[e][p] = PipeFlowDirection.DISABLED
                elif lb >= 0.0:
                    self.__implied_directions[e][p] = PipeFlowDirection.POSITIVE
                elif ub <= 0.0:
                    self.__implied_directions[e][p] = PipeFlowDirection.NEGATIVE

    def constant_inputs(self, ensemble_member):
        inputs = super().constant_inputs(ensemble_member)
        for p, d in self.__implied_directions[ensemble_member].items():
            k = self.heat_network_flow_directions[p]
            inputs[k] = Timeseries([-np.inf, np.inf], [d, d])
        return inputs

    @property
    def heat_network_flow_directions(self) -> Dict[str, str]:
        pipes = self.heat_network_components["pipe"]
        return {p: f"{p}__implied_direction" for p in pipes}
