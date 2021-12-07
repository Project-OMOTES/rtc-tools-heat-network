from dataclasses import dataclass
from typing import Optional

import esdl


@dataclass
class Asset:
    asset_type: str
    id: str
    name: str
    in_port: Optional[esdl.InPort]
    out_port: Optional[esdl.OutPort]
    attributes: dict
    global_properties: dict
