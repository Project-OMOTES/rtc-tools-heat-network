from numpy import nan

from .source import Source


class GeothermalSource(Source):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_subtype = "geothermal"

        self.target_flow_rate = nan
        self.single_doublet_power = nan
        self.nr_of_doublets = 1.0
