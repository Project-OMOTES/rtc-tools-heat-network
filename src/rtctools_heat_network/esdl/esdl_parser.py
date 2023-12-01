import base64
from pathlib import Path
from typing import Dict, Optional

import esdl.esdl_handler

from pyecore.resources import ResourceSet

from .common import Asset


class _ESDLInputException(Exception):
    pass


class BaseESDLParser:
    _global_properties: Dict[str, Dict] = {"carriers": dict()}
    _assets: Dict[str, Asset] = dict()
    _energy_system: Optional[esdl.EnergySystem] = None
    _esdl_string: Optional[str] = None
    _esdl_path: Optional[Path] = None

    def __init__(self):
        self._read_esdl()

    def _load_esdl_model(self) -> None:
        """
        This function should be implemented by the child. The function doesn't return anything but
        should set the _esdl_model property.
        """
        raise NotImplementedError

    def _read_esdl(self) -> None:
        self._load_esdl_model()
        id_to_idnumber_map = {}

        for x in self._energy_system.energySystemInformation.carriers.carrier.items:
            if isinstance(x, esdl.esdl.HeatCommodity):
                if x.supplyTemperature != 0.0 and x.returnTemperature == 0.0:
                    type_ = "supply"
                elif x.returnTemperature != 0.0 and x.supplyTemperature == 0.0:
                    type_ = "return"
                else:
                    type_ = "none"
                if x.id not in id_to_idnumber_map:
                    number_list = [int(s) for s in x.id if s.isdigit()]
                    number = ""
                    for nr in number_list:
                        number = number + str(nr)
                    if type_ == "return":
                        number = number + "000"
                    id_to_idnumber_map[x.id] = int(number)

                self._global_properties["carriers"][x.id] = dict(
                    name=x.name.replace("_ret", ""),
                    id=x.id,
                    id_number_mapping=id_to_idnumber_map[x.id],
                    supplyTemperature=x.supplyTemperature,
                    returnTemperature=x.returnTemperature,
                    __rtc_type=type_,
                )

        # For now, we only support networks with two carries; one hot, one cold.
        # When this no longer holds, carriers either have to specify both the
        # supply and return temperature (instead of one being 0.0), or we have to
        # pair them up.
        if (len(self._global_properties["carriers"]) % 2) != 0:
            _ESDLInputException(
                "Odd number of carriers specified, please use model with dedicated supply and return "
                "carriers. Every hydraulically coupled system should have one carrier for the supply "
                "side and one for the return side"
            )

        for c in self._global_properties["carriers"].values():
            supply_temperature = next(
                x["supplyTemperature"]
                for x in self._global_properties["carriers"].values()
                if x["supplyTemperature"] != 0.0 and x["name"] == c["name"]
            )
            return_temperature = next(
                x["returnTemperature"]
                for x in self._global_properties["carriers"].values()
                if x["returnTemperature"] != 0.0 and x["name"] == c["name"]
            )
            c["supplyTemperature"] = supply_temperature
            c["returnTemperature"] = return_temperature

        for x in self._energy_system.energySystemInformation.carriers.carrier.items:
            if isinstance(x, esdl.esdl.ElectricityCommodity):
                self._global_properties["carriers"][x.id] = dict(
                    name=x.name,
                    voltage=x.voltage,
                )

        assets = {}

        # Component ids are unique, but we require component names to be unique as well.
        component_names = set()

        # loop through assets
        for el in self._energy_system.eAllContents():
            if isinstance(el, esdl.Asset):
                if hasattr(el, "name") and el.name:
                    el_name = el.name
                else:
                    el_name = el.id

                if "." in el_name:
                    # Dots indicate hierarchy, so would be very confusing
                    raise ValueError(f"Dots in component names not supported: '{el_name}'")

                if el_name in component_names:
                    raise Exception(f"Asset names have to be unique: '{el_name}' already exists")
                else:
                    component_names.add(el_name)

                # For some reason `esdl_element.assetType` is `None`, so use the class name
                asset_type = el.__class__.__name__

                # Every asset should at least have a port to be connected to another asset
                assert len(el.port) >= 1

                in_ports = None
                out_ports = None
                for port in el.port:
                    if isinstance(port, esdl.InPort):
                        if in_ports is None:
                            in_ports = [port]
                        else:
                            in_ports.append(port)
                    elif isinstance(port, esdl.OutPort):
                        if out_ports is None:
                            out_ports = [port]
                        else:
                            out_ports.append(port)
                    else:
                        _ESDLInputException(f"The port for {el_name} is neither an IN or OUT port")

                # Note that e.g. el.__dict__['length'] does not work to get the length of a pipe.
                # We therefore built this dict ourselves using 'dir' and 'getattr'
                attributes = {k: getattr(el, k) for k in dir(el)}
                assets[el.id] = Asset(
                    asset_type,
                    el.id,
                    el_name,
                    in_ports,
                    out_ports,
                    attributes,
                    self._global_properties,
                )

    def get_assets(self) -> Dict[str, Asset]:
        return self._assets

    def get_carrier_properties(self) -> Dict:
        return self._global_properties["carriers"]

    def get_esdl_model(self) -> esdl.EnergySystem:
        return self._energy_system


class ESDLStringParser(BaseESDLParser):

    def __init__(self, **kwargs):
        try:
            esdl_string = kwargs.get("esdl_string")
        except KeyError:
            raise _ESDLInputException(f"Expected an ESDL string when parsing the system from a "
                                      f"string, but none provided")
        if isinstance(esdl_string, bytes):
            self._esdl_string = base64.b64decode(esdl_string).decode('utf-8')
        else:
            self._esdl_string = esdl_string
        super().__init__()

    def _load_esdl_model(self) -> None:
        esh = esdl.esdl_handler.EnergySystemHandler()
        self._energy_system = esh.load_from_string(self._esdl_string)


class ESDLFileParser(BaseESDLParser):

    def __init__(self, **kwargs):
        try:
            esdl_path = kwargs.get("esdl_path")
        except KeyError:
            raise _ESDLInputException(f"Expected an ESDL path when parsing the system from a "
                                      f"file but none provided")
        self._esdl_path = esdl_path
        super().__init__()

    def _load_esdl_model(self) -> None:
        # correct profile attribute
        esdl.ProfileElement.from_.name = "from"
        setattr(esdl.ProfileElement, "from", esdl.ProfileElement.from_)

        # using esdl as resourceset
        rset_existing = ResourceSet()

        # read esdl energy system
        resource_existing = rset_existing.get_resource(str(self._esdl_path))
        created_energy_system = resource_existing.contents[0]

        self._energy_system = created_energy_system


