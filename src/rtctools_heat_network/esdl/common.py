from dataclasses import dataclass


@dataclass
class Asset:
    asset_type: str
    id: str
    name: str
    in_ports: None
    out_ports: None
    attributes: dict
    global_properties: dict
