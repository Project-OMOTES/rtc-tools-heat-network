from ._internal import HeatComponent
from ._non_storage_component import _NonStorageComponent


class HeatFourPort(HeatComponent):
    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(),
                modifiers,
            ),
        )

        self.add_variable(_NonStorageComponent, "Primary", **modifiers["Primary"])
        self.add_variable(_NonStorageComponent, "Secondary", **modifiers["Secondary"])
