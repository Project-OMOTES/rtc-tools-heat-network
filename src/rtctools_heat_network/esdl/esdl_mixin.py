import logging
import xml.etree.ElementTree as ET  # noqa: N817
from datetime import timedelta
from pathlib import Path
from typing import Dict, Union

from pyecore.resources import ResourceSet

import rtctools.data.pi as pi
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.io_mixin import IOMixin

import rtctools_heat_network.esdl.esdl as esdl
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.modelica_component_type_mixin import ModelicaComponentTypeMixin
from rtctools_heat_network.pycml.pycml_mixin import PyCMLMixin
from rtctools_heat_network.qth_mixin import QTHMixin

from .common import Asset
from .esdl_heat_model import ESDLHeatModel
from .esdl_qth_model import ESDLQTHModel

logger = logging.getLogger("rtctools_heat_network")


ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}


class ESDLMixin(
    ModelicaComponentTypeMixin, IOMixin, PyCMLMixin, CollocatedIntegratedOptimizationProblem
):

    esdl_run_info_path: Path = None

    esdl_pi_validate_timeseries = False

    esdl_pi_input_data_config = None
    esdl_pi_output_data_config = None

    def __init__(self, *args, **kwargs):

        if not self.esdl_run_info_path:
            self.esdl_run_info_path = Path(kwargs["input_folder"]) / "RunInfo.xml"

        if not self.esdl_pi_input_data_config:
            self.esdl_pi_input_data_config = _ESDLInputDataConfig

        if not self.esdl_pi_output_data_config:
            self.esdl_pi_output_data_config = _ESDLOutputDataConfig

        self.__run_info = _RunInfoReader(self.esdl_run_info_path)
        assets = self.__run_info.assets

        # Although we work with the names, the FEWS import data uses the component IDs
        self.__timeseries_id_map = {a.id: a.name for a in assets.values()}

        if isinstance(self, HeatMixin):
            self.__model = ESDLHeatModel(assets, **self.esdl_heat_model_options())
        else:
            assert isinstance(self, QTHMixin)
            self.__model = ESDLQTHModel(assets, **self.esdl_qth_model_options())

        root_logger = logging.getLogger("")

        if self.__run_info.output_diagnostic_file:
            # Add stream handler if it does not already exist.
            if not logger.hasHandlers() and not any(
                (isinstance(h, logging.StreamHandler) for h in logger.handlers)
            ):
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            # Add pi.DiagHandler. Only add if it doesn't already exist.
            if not any((isinstance(h, pi.DiagHandler) for h in root_logger.handlers)):
                basename = self.__run_info.output_diagnostic_file.stem
                folder = self.__run_info.output_diagnostic_file.parent

                handler = pi.DiagHandler(folder, basename)
                root_logger.addHandler(handler)

        self.__input_timeseries_file = self.__run_info.input_timeseries_file
        self.__output_timeseries_file = self.__run_info.output_timeseries_file

        super().__init__(*args, **kwargs)

    @property
    def esdl_asset_id_to_name_map(self):
        return self.__timeseries_id_map.copy()

    def esdl_heat_model_options(self) -> Dict:
        heat_network_options = self.heat_network_options()
        v_nominal = heat_network_options["estimated_velocity"]
        v_max = heat_network_options["maximum_velocity"]
        return dict(v_nominal=v_nominal, v_max=v_max)

    def esdl_qth_model_options(self) -> Dict:
        heat_network_options = self.heat_network_options()
        v_nominal = heat_network_options["estimated_velocity"]
        v_max = heat_network_options["maximum_velocity"]
        return dict(v_nominal=v_nominal, v_max=v_max)

    def is_hot_pipe(self, pipe: str) -> bool:
        return not self.is_cold_pipe(pipe)

    def is_cold_pipe(self, pipe: str) -> bool:
        return pipe.endswith("_ret")

    def hot_to_cold_pipe(self, pipe: str):
        return f"{pipe}_ret"

    def cold_to_hot_pipe(self, pipe: str):
        return pipe[:-4]

    def pycml_model(self):
        return self.__model

    def read(self):
        super().read()

        if self.__input_timeseries_file is None:
            return

        input_timeseries_file = Path(self.__input_timeseries_file)
        assert input_timeseries_file.is_absolute()
        assert input_timeseries_file.suffix == ".xml"

        timeseries_import_basename = input_timeseries_file.stem
        input_folder = input_timeseries_file.parent

        try:
            self.__timeseries_import = pi.Timeseries(
                self.esdl_pi_input_data_config(self.__timeseries_id_map),
                input_folder,
                timeseries_import_basename,
                binary=False,
                pi_validate_times=self.esdl_pi_validate_timeseries,
            )
        except IOError:
            raise Exception(
                "ESDLMixin: {}.xml not found in {}.".format(
                    timeseries_import_basename, input_folder
                )
            )

        # Convert timeseries timestamps to seconds since t0 for internal use
        timeseries_import_times = self.__timeseries_import.times

        # Offer input timeseries to IOMixin
        self.io.reference_datetime = self.__timeseries_import.forecast_datetime

        for ensemble_member in range(self.__timeseries_import.ensemble_size):
            for variable, values in self.__timeseries_import.items(ensemble_member):
                self.io.set_timeseries(variable, timeseries_import_times, values, ensemble_member)

    def write(self):
        super().write()

        if self.__output_timeseries_file is None:
            return

        output_timeseries_file = Path(self.__output_timeseries_file)
        assert output_timeseries_file.is_absolute()
        assert output_timeseries_file.suffix == ".xml"

        timeseries_export_basename = output_timeseries_file.stem
        output_folder = output_timeseries_file.parent

        try:
            self.__timeseries_export = pi.Timeseries(
                self.esdl_pi_output_data_config(self.__timeseries_id_map),
                output_folder,
                timeseries_export_basename,
                binary=False,
                pi_validate_times=self.esdl_pi_validate_timeseries,
            )
        except IOError:
            raise Exception(
                "ESDLMixin: {}.xml not found in {}.".format(
                    timeseries_export_basename, output_folder
                )
            )

        # Get time stamps
        times = self.times()
        if len(set(times[1:] - times[:-1])) == 1:
            dt = timedelta(seconds=times[1] - times[0])
        else:
            dt = None

        output_keys = [k for k, v in self.__timeseries_export.items()]

        # Start of write output
        # Write the time range for the export file.
        self.__timeseries_export.times = [
            self.__timeseries_import.times[self.__timeseries_import.forecast_index]
            + timedelta(seconds=s)
            for s in times
        ]

        # Write other time settings
        self.__timeseries_export.forecast_datetime = self.__timeseries_import.forecast_datetime
        self.__timeseries_export.dt = dt
        self.__timeseries_export.timezone = self.__timeseries_import.timezone

        # Write the ensemble properties for the export file.
        if self.ensemble_size > 1:
            self.__timeseries_export.contains_ensemble = True
        self.__timeseries_export.ensemble_size = self.ensemble_size
        self.__timeseries_export.contains_ensemble = self.ensemble_size > 1

        # Start looping over the ensembles for extraction of the output values.
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            # For all variables that are output variables the values are
            # extracted from the results.
            for variable in output_keys:
                try:
                    values = results[variable]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times, self.times(variable), values, self.interpolation_method(variable)
                        )
                except KeyError:
                    try:
                        ts = self.get_timeseries(variable, ensemble_member)
                        if len(ts.times) != len(times):
                            values = self.interpolate(times, ts.times, ts.values)
                        else:
                            values = ts.values
                    except KeyError:
                        logger.warning(
                            "ESDLMixin: Output requested for non-existent variable {}. "
                            "Will not be in output file.".format(variable)
                        )
                        continue

                self.__timeseries_export.set(variable, values, ensemble_member=ensemble_member)

        # Write output file to disk
        self.__timeseries_export.write()

    def pre(self):
        super().pre()

        # Make sure that all demands have a Timeseries set
        for demand in self.heat_network_components["demand"]:
            try:
                _ = self.__timeseries_import.get(f"{demand}.target_heat_demand")
            except KeyError:
                raise KeyError(
                    f"Could not find a Timeseries for '{demand}.Heat_demand' in the XML file"
                )


class _ESDLInputDataConfig:
    def __init__(self, id_map):
        self.__id_map = id_map

    def variable(self, pi_header):
        location_id = pi_header.find("pi:locationId", ns).text

        try:
            component_name = self.__id_map[location_id]
        except KeyError:
            parameter_id = pi_header.find("pi:parameterId", ns).text
            qualifiers = pi_header.findall("pi:qualifierId", ns)
            qualifier_ids = ":".join(q.text for q in qualifiers)
            return f"{location_id}:{parameter_id}:{qualifier_ids}"

        suffix = ".target_heat_demand"

        # Note that the qualifier id (if any specified) refers to the profile
        # element of the respective ESDL asset->in_port. For now we just
        # assume that only heat demand timeseries are set in the XML file.
        return f"{component_name}{suffix}"

    def pi_variable_ids(self, variable):
        raise NotImplementedError

    def parameter(self, parameter_id, location_id=None, model_id=None):
        raise NotImplementedError

    def pi_parameter_ids(self, parameter):
        raise NotImplementedError


class _ESDLOutputDataConfig:
    def __init__(self, id_map):
        self.__id_map = id_map

    def variable(self, pi_header):
        location_id = pi_header.find("pi:locationId", ns).text
        parameter_id = pi_header.find("pi:parameterId", ns).text

        component_name = self.__id_map[location_id]

        return f"{component_name}.{parameter_id}"

    def pi_variable_ids(self, variable):
        raise NotImplementedError

    def parameter(self, parameter_id, location_id=None, model_id=None):
        raise NotImplementedError

    def pi_parameter_ids(self, parameter):
        raise NotImplementedError


class _RunInfoReader:
    def __init__(self, filepath: Union[str, Path]):
        filepath = Path(filepath).resolve()

        root = ET.parse(filepath).getroot()

        # If the workDir is not absolute, we take it relative to the folder in
        # which the RunInfo file is.
        work_dir = Path(root.findtext("pi:workDir", namespaces=ns))
        if not work_dir.is_absolute():
            work_dir = filepath.parent / work_dir

        self._esdl_path = Path(root.find("pi:properties", ns)[0].attrib["value"])
        if not self._esdl_path.is_absolute():
            self._esdl_path = work_dir / self._esdl_path

        assets = _esdl_to_assets(self._esdl_path)

        parameters_file = root.findtext("pi:inputParameterFile", namespaces=ns)
        if parameters_file is not None:
            parameters_file = Path(parameters_file)
            if not parameters_file.is_absolute():
                parameters_file = work_dir / parameters_file
                assets = self._overwrite_parameters(parameters_file, assets)

        self._assets = assets

        try:
            self.input_timeseries_file = Path(root.findall("pi:inputTimeSeriesFile", ns)[0].text)
            if not self.input_timeseries_file.is_absolute():
                self.input_timeseries_file = work_dir / self.input_timeseries_file
        except IndexError:
            self.input_timeseries_file = None

        try:
            self.output_timeseries_file = Path(root.findall("pi:outputTimeSeriesFile", ns)[0].text)
            if not self.output_timeseries_file.is_absolute():
                self.output_timeseries_file = work_dir / self.output_timeseries_file
        except IndexError:
            self.output_timeseries_file = None

        try:
            self.output_diagnostic_file = Path(root.findall("pi:outputDiagnosticFile", ns)[0].text)
            if not self.output_diagnostic_file.is_absolute():
                self.output_diagnostic_file = work_dir / self.output_diagnostic_file
        except IndexError:
            self.output_diagnostic_file = None

    @property
    def assets(self):
        return self._assets

    @staticmethod
    def _overwrite_parameters(parameters_file, assets):
        paramroot = ET.parse(parameters_file).getroot()
        groups = paramroot.findall("pi:group", ns)

        for parameter in groups:
            id_ = parameter.attrib["id"]
            param_name = parameter[0].attrib["id"]
            param_value = parameter[0][0].text

            asset = assets[id_]
            type_ = type(asset.attributes[param_name])
            asset.attributes[param_name] = type_(param_value)

        return assets


def _esdl_to_assets(esdl_path: Union[Path, str]):
    # correct profile attribute
    esdl.ProfileElement.from_.name = "from"
    setattr(esdl.ProfileElement, "from", esdl.ProfileElement.from_)

    # using esdl as resourceset
    rset_existing = ResourceSet()

    # read esdl energy system
    resource_existing = rset_existing.get_resource(str(esdl_path))
    created_energy_system = resource_existing.contents[0]

    esdl_model = created_energy_system

    # global properties
    global_properties = {}
    global_properties["carriers"] = {}

    for x in esdl_model.energySystemInformation.carriers.carrier.items:
        if isinstance(x, esdl.esdl.HeatCommodity):
            if x.supplyTemperature != 0.0 and x.returnTemperature == 0.0:
                type = "supply"
            elif x.returnTemperature != 0.0 and x.supplyTemperature == 0.0:
                type = "return"
            else:
                type = "none"
            global_properties["carriers"][x.id] = dict(
                supplyTemperature=x.supplyTemperature,
                returnTemperature=x.returnTemperature,
                __rtc_type=type,
            )

    # For now, we only support networks with two carries; one hot, one cold.
    # When this no longer holds, carriers either have to specify both the
    # supply and return temperature (instead of one being 0.0), or we have to
    # pair them up.
    assert len(global_properties["carriers"]) == 2
    supply_temperature = next(
        x["supplyTemperature"]
        for x in global_properties["carriers"].values()
        if x["supplyTemperature"] != 0.0
    )
    return_temperature = next(
        x["returnTemperature"]
        for x in global_properties["carriers"].values()
        if x["returnTemperature"] != 0.0
    )

    for c in global_properties["carriers"].values():
        c["supplyTemperature"] = supply_temperature
        c["returnTemperature"] = return_temperature

    assets = {}

    # Component ids are unique, but we require component names to be unique as well.
    component_names = set()

    # loop through assets
    for el in esdl_model.eAllContents():
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

            assert 1 <= len(el.port) <= 2

            if len(el.port) == 1 and isinstance(el.port[0], esdl.InPort):
                in_port, out_port = el.port[0], None
            elif len(el.port) == 1 and isinstance(el.port[0], esdl.OutPort):
                out_port, in_port = el.port[0], None
            elif isinstance(el.port[0], esdl.InPort) and isinstance(el.port[1], esdl.OutPort):
                in_port, out_port = el.port
            elif isinstance(el.port[1], esdl.InPort) and isinstance(el.port[0], esdl.OutPort):
                out_port, in_port = el.port
            else:
                raise Exception(f"Unexpected combination of In- and OutPorts for '{el_name}'")

            # Note that e.g. el.__dict__['length'] does not work to get the length of a pipe.
            # We therefore built this dict ourselves using 'dir' and 'getattr'
            attributes = {k: getattr(el, k) for k in dir(el)}
            assets[el.id] = Asset(
                asset_type, el.id, el_name, in_port, out_port, attributes, global_properties
            )

    return assets
