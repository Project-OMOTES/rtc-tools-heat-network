from dataclasses import dataclass

from . import esdl


@dataclass
class Asset:
    asset_type: str
    id: str
    name: str
    in_port: esdl.InPort
    out_port: esdl.InPort
    attributes: dict
    global_properties: dict
