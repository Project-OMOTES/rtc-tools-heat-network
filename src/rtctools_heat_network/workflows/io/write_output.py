import logging
import os
import uuid
from pathlib import Path

import esdl

import numpy as np

import pandas as pd

from rtctools_heat_network.esdl.edr_pipe_class import EDRPipeClass
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.workflows.utils.helpers import _sort_numbered

logger = logging.getLogger("rtctools_heat_network")


class ScenarioOutput(HeatMixin):
    __optimized_energy_system_handler = None

    def get_optimized_esh(self):
        return self.__optimized_energy_system_handler

    def _write_html_output(self, template_name="mpc_buffer_sizing_output"):
        from jinja2 import Environment, FileSystemLoader

        assert self.ensemble_size == 1

        results = self.extract_results()
        parameters = self.parameters(0)

        # Format the priority results
        priority_results = [
            dict(
                number=number,
                success=success,
                pretty_time=f"{int(seconds // 60):02d}:{seconds % 60:06.3f}",
                objective_value=objective_value,
                return_status=stats["return_status"],
                secondary_return_status=stats.get("secondary_return_status", ""),
            )
            for (
                number,
                seconds,
                success,
                objective_value,
                stats,
            ) in self._priorities_output
        ]

        # Format the buffer results
        results_buffers_placed = {}
        results_buffers_size = {}
        results_sources_placed = {}
        results_sources_size = {}
        results_max_charging_rate = {}
        results_max_discharging_rate = {}

        for buffer in _sort_numbered(self.heat_network_components.get("buffer", [])):
            if buffer in self._minimize_size_buffers:
                max_size_var = self._max_buffer_heat_map[buffer]
                results_buffers_size[buffer] = float(results[max_size_var][0]) / (
                    parameters[f"{buffer}.cp"]
                    * parameters[f"{buffer}.rho"]
                    * (parameters[f"{buffer}.T_supply"] - parameters[f"{buffer}.T_return"])
                )
            else:
                results_buffers_size[buffer] = "-"

            if buffer in self._optional_buffers:
                buffer_placement_var = self._buffer_placement_map[buffer]
                results_buffers_placed[buffer] = np.round(results[buffer_placement_var][0]) == 1.0
            else:
                results_buffers_placed[buffer] = "-"

            (_, hot_orient), _ = self.heat_network_topology.buffers[buffer]
            q = hot_orient * results[f"{buffer}.HeatIn.Q"]
            inds_charging = q > 0
            inds_discharging = q < 0

            results_max_charging_rate[buffer] = max(q[inds_charging]) if any(inds_charging) else 0.0
            results_max_discharging_rate[buffer] = (
                max(-1 * q[inds_discharging]) if any(inds_discharging) else 0.0
            )

        buffer_results = [
            dict(
                name=buffer,
                tune_size=buffer in self._minimize_size_buffers,
                tune_placement=buffer in self._optional_buffers,
                maximum_size=self._override_buffer_size[buffer],
                result_placed=results_buffers_placed[buffer],
                result_size=results_buffers_size[buffer],
                max_charging_rate=results_max_charging_rate[buffer],
                max_discharging_rate=results_max_discharging_rate[buffer],
            )
            for buffer in self.heat_network_components.get("buffer", [])
        ]

        for source in _sort_numbered(self.heat_network_components["source"]):
            if source in self._minimize_size_sources:
                max_size_var = self._max_source_heat_map[source]
                results_sources_size[source] = float(results[max_size_var][0]) / 10.0**3
            else:
                results_sources_size[source] = "-"

            if source in self._optional_sources:
                source_placement_var = self._source_placement_map[source]
                results_sources_placed[source] = np.round(results[source_placement_var][0]) == 1.0
            else:
                results_sources_placed[source] = "-"

        source_results = [
            dict(
                name=source,
                tune_size=source in self._minimize_size_sources,
                tune_placement=source in self._optional_sources,
                maximum_size=self._override_max_power[source],
                result_placed=results_sources_placed[source],
                result_size=results_sources_size[source],
            )
            for source in self.heat_network_components["source"]
        ]

        # Format the pipe results
        # Note that we do not distinguish between routing and sizing
        # internally, but for the sake of the output we do.
        pipe_results = []

        for p in _sort_numbered(self.hot_pipes):
            pipe_classes = self.pipe_classes(p)
            tune_routing = len([pc for pc in pipe_classes if pc.inner_diameter == 0.0]) == 1
            inner_diameter = parameters[f"{p}.diameter"]
            asset = next(a for a in self.esdl_assets.values() if a.name == p)
            esdl_diameter = asset.attributes["innerDiameter"]

            if len(pipe_classes) <= 1:
                tune_size = False
                min_dn_size = "-"
                max_dn_size = "-"
                result_placed = "-"
                result_dn_size = "-"
            elif len(pipe_classes) == 2 and tune_routing:
                tune_size = False
                min_dn_size = "-"
                max_dn_size = "-"
                result_placed = "Yes" if inner_diameter > 0 else "No"
                result_dn_size = "-"
            else:
                sorted_pipe_classes = sorted(
                    [pc for pc in pipe_classes if pc.inner_diameter > 0],
                    key=lambda pc: pc.inner_diameter,
                )

                tune_size = True
                min_dn_size = sorted_pipe_classes[0].name
                max_dn_size = sorted_pipe_classes[-1].name
                result_placed = "Yes" if inner_diameter > 0 else "No"
                result_pipe_class = self.get_optimized_pipe_class(p)
                result_dn_size = (
                    result_pipe_class.name if result_pipe_class is not None else inner_diameter
                )

            pipe_results.append(
                dict(
                    name=p,
                    tune_size=tune_size,
                    tune_routing=tune_routing,
                    esdl_diameter=esdl_diameter,
                    min_dn_size=min_dn_size,
                    max_dn_size=max_dn_size,
                    result_placed=result_placed,
                    result_dn_size=result_dn_size,
                )
            )

        input_csv_tables = {
            os.path.basename(x): pd.read_csv(x).to_dict("records")
            for x in self._csv_input_parameter_files
        }

        # Actually write out the html file based on the template
        templates_dir = Path(__file__).parent / "templates"

        env = Environment(loader=FileSystemLoader(templates_dir))
        template = env.get_template(template_name + ".html")

        os.makedirs(self._html_output_dir, exist_ok=True)

        filename = self._html_output_dir / (template_name + ".html")

        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(
                template.render(
                    buffer_results=buffer_results,
                    source_results=source_results,
                    pipe_results=pipe_results,
                    priority_results=priority_results,
                    input_csv_tables=input_csv_tables,
                )
            )

    def _write_updated_esdl(self, db_profiles=False, optimizer_sim=True):
        from esdl.esdl_handler import EnergySystemHandler
        from rtctools_heat_network.esdl.esdl_mixin import _RunInfoReader

        results = self.extract_results()
        parameters = self.parameters(0)

        esh = EnergySystemHandler()
        if self.esdl_string is None:
            run_info = _RunInfoReader(self.esdl_run_info_path)
            energy_system = esh.load_file(str(run_info.esdl_file))
        else:
            energy_system = esh.load_from_string(self.esdl_string)

        energy_system.id = str(uuid.uuid4())
        energy_system.name = energy_system.name + "_SmartControlOptimized"

        def _name_to_asset(name):
            return next(
                (x for x in energy_system.eAllContents() if hasattr(x, "name") and x.name == name)
            )

        # ------------------------------------------------------------------------------------------
        # KPIs
        # General cost breakdowns
        # ------------------------------------------------------------------------------------------
        kpis_top_level = esdl.KPIs(id=str(uuid.uuid4()))
        energy_wh = {}
        capex_breakdown = {}
        opex_breakdown = {}
        tot_install_cost_euro = 0.0
        tot_invest_cost_euro = 0.0
        tot_variable_opex_cost_euro = 0.0
        tot_fixed_opex_cost_euro = 0.0

        for _key, asset in self.esdl_assets.items():
            is_asset_opex_included = False
            if (
                asset.asset_type == "HeatProducer"
                or asset.asset_type == "GenericProducer"
                or asset.asset_type == "ResidualHeatSource"
                or asset.asset_type == "GeothermalSource"
                or asset.asset_type == "ResidualHeatSource"
                or asset.asset_type == "GasHeater"
            ):
                is_asset_opex_included = True

            asset_placement_var = self._asset_aggregation_count_var_map[asset.name]
            placed = np.round(results[asset_placement_var][0]) >= 1.0
            if placed:
                try:
                    tot_install_cost_euro += results[f"{asset.name}__installation_cost"][0]
                    tot_invest_cost_euro += results[f"{asset.name}__investment_cost"][0]
                    capex_breakdown[asset.asset_type] += (
                        tot_install_cost_euro + tot_invest_cost_euro
                    )

                    if is_asset_opex_included:
                        tot_variable_opex_cost_euro += results[
                            f"{asset.name}__variable_operational_cost"
                        ][0]
                        tot_fixed_opex_cost_euro += results[
                            f"{asset.name}__fixed_operational_cost"
                        ][0]
                        opex_breakdown[asset.asset_type] += (
                            tot_variable_opex_cost_euro + tot_fixed_opex_cost_euro
                        )

                except KeyError:
                    try:
                        tot_install_cost_euro = results[f"{asset.name}__installation_cost"][0]
                        tot_invest_cost_euro = results[f"{asset.name}__investment_cost"][0]
                        capex_breakdown[asset.asset_type] = (
                            tot_install_cost_euro + tot_invest_cost_euro
                        )

                        if is_asset_opex_included:
                            tot_variable_opex_cost_euro = results[
                                f"{asset.name}__variable_operational_cost"
                            ][0]
                            tot_fixed_opex_cost_euro = results[
                                f"{asset.name}__fixed_operational_cost"
                            ][0]
                            opex_breakdown[asset.asset_type] = (
                                tot_variable_opex_cost_euro + tot_fixed_opex_cost_euro
                            )
                    except KeyError:
                        # Do not add any costs. Items like joint
                        pass

                if is_asset_opex_included:
                    energy_wh[asset.name] = np.sum(
                        results[f"{asset.name}.Heat_source"][1:]
                        * (self.times()[1:] - self.times()[0:-1])
                        / 3600
                    )

        kpis_top_level.kpi.append(
            esdl.DistributionKPI(
                name="High level cost breakdown [EUR]",
                distribution=esdl.StringLabelDistribution(
                    stringItem=[
                        esdl.StringItem(
                            label="CAPEX",
                            value=tot_install_cost_euro + tot_invest_cost_euro,
                        ),
                        esdl.StringItem(
                            label="OPEX",
                            value=tot_variable_opex_cost_euro + tot_fixed_opex_cost_euro,
                        ),
                    ]
                ),
                quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                    physicalQuantity=esdl.PhysicalQuantityEnum.COST, unit=esdl.UnitEnum.EURO
                ),
            )
        )

        kpis_top_level.kpi.append(
            esdl.DistributionKPI(
                name="Overall cost breakdown [EUR]",
                distribution=esdl.StringLabelDistribution(
                    stringItem=[
                        esdl.StringItem(label="Installation", value=tot_install_cost_euro),
                        esdl.StringItem(label="Investment", value=tot_invest_cost_euro),
                        esdl.StringItem(label="Variable OPEX", value=tot_variable_opex_cost_euro),
                        esdl.StringItem(label="Fixed OPEX", value=tot_fixed_opex_cost_euro),
                    ]
                ),
                quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                    physicalQuantity=esdl.PhysicalQuantityEnum.COST, unit=esdl.UnitEnum.EURO
                ),
            )
        )

        kpis_top_level.kpi.append(
            esdl.DistributionKPI(
                name="CAPEX breakdown [EUR]",
                distribution=esdl.StringLabelDistribution(
                    stringItem=[
                        esdl.StringItem(label=key, value=value)
                        for key, value in capex_breakdown.items()
                    ]
                ),
                quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                    physicalQuantity=esdl.PhysicalQuantityEnum.COST, unit=esdl.UnitEnum.EURO
                ),
            )
        )

        kpis_top_level.kpi.append(
            esdl.DistributionKPI(
                name="OPEX breakdown [EUR]",
                distribution=esdl.StringLabelDistribution(
                    stringItem=[
                        esdl.StringItem(label=key, value=value)
                        for key, value in opex_breakdown.items()
                    ]
                ),
                quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                    physicalQuantity=esdl.PhysicalQuantityEnum.COST, unit=esdl.UnitEnum.EURO
                ),
            )
        )

        kpis_top_level.kpi.append(
            esdl.DistributionKPI(
                name="Energy production [Wh]",
                distribution=esdl.StringLabelDistribution(
                    stringItem=[
                        esdl.StringItem(label=key, value=value) for key, value in energy_wh.items()
                    ]
                ),
            )
        )
        energy_system.instance[0].area.KPIs = kpis_top_level
        # ------------------------------------------------------------------------------------------
        # Cost breakdowns per polygon areas (can consist of several assets of differents types)
        # Notes:
        # - OPEX KPIs are taken into account for energy sources only.
        # - We assume that all energy produced outside of the the subarea comes in via a heat
        #   exchanger that is part of the subarea.
        for subarea in energy_system.instance[0].area.area:
            area_investment_cost = 0.0
            area_installation_cost = 0.0
            area_variable_opex_cost = 0.0
            area_fixed_opex_cost = 0.0

            is_at_least_asset_opex_included = False
            kpis = esdl.KPIs(id=str(uuid.uuid4()))
            # Here we make a breakdown of the produced energy in the subarea. Where we assume that
            # all energy produced outside of the the subarea comes in via a heat exchanger that is
            # part of the subarea.
            energy_breakdown = {}
            for asset in subarea.asset:
                asset_name = asset.name
                asset_type = self.get_asset_from_asset_name(asset_name).asset_type

                asset_placement_var = self._asset_aggregation_count_var_map[asset.name]
                placed = np.round(results[asset_placement_var][0]) >= 1.0

                if placed:
                    is_asset_opex_included = False
                    if (
                        asset_type == "HeatProducer"
                        or asset_type == "GenericProducer"
                        or asset_type == "ResidualHeatSource"
                        or asset_type == "GeothermalSource"
                        or asset_type == "ResidualHeatSource"
                        or asset_type == "GasHeater"
                        or asset_type == "HeatStorage"
                    ):
                        is_asset_opex_included = True
                        is_at_least_asset_opex_included = True

                    if asset_type == "Joint":
                        continue
                    try:
                        energy_breakdown[asset_type] += np.sum(results[f"{asset_name}.Heat_source"])
                    except KeyError:
                        try:
                            energy_breakdown[asset_type] = np.sum(
                                results[f"{asset_name}.Heat_source"]
                            )
                        except KeyError:
                            try:
                                energy_breakdown[asset_type] += np.sum(
                                    results[f"{asset_name}.Secondary_heat"]
                                )
                            except KeyError:
                                try:
                                    energy_breakdown[asset_type] = np.sum(
                                        results[f"{asset_name}.Secondary_heat"]
                                    )
                                except KeyError:
                                    pass

                    # Create KPIs by using applicable costs for the specific asset
                    area_investment_cost += results[self._asset_investment_cost_map[asset_name]][0]
                    area_installation_cost += results[
                        self._asset_installation_cost_map[asset_name]
                    ][0]
                    if is_asset_opex_included:
                        area_variable_opex_cost += results[
                            self._asset_variable_operational_cost_map[asset_name]
                        ][0]
                        area_fixed_opex_cost += results[
                            self._asset_fixed_operational_cost_map[asset_name]
                        ][0]

            # Here we add KPIs to the polygon area which allows to visualize them by hoovering over
            # it with the mouse
            kpis.kpi.append(
                esdl.DoubleKPI(
                    value=area_investment_cost / 1.0e6,
                    name="Investment",
                    quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                        physicalQuantity=esdl.PhysicalQuantityEnum.COST,
                        unit=esdl.UnitEnum.EURO,
                        multiplier=esdl.MultiplierEnum.MEGA,
                    ),
                )
            )
            kpis.kpi.append(
                esdl.DoubleKPI(
                    value=area_installation_cost / 1.0e6,
                    name="Installation",
                    quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                        physicalQuantity=esdl.PhysicalQuantityEnum.COST,
                        unit=esdl.UnitEnum.EURO,
                        multiplier=esdl.MultiplierEnum.MEGA,
                    ),
                )
            )

            if is_at_least_asset_opex_included:
                kpis.kpi.append(
                    esdl.DoubleKPI(
                        value=area_variable_opex_cost / 1.0e6,
                        name="Variable OPEX",
                        quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                            physicalQuantity=esdl.PhysicalQuantityEnum.COST,
                            unit=esdl.UnitEnum.EURO,
                            multiplier=esdl.MultiplierEnum.MEGA,
                        ),
                    )
                )
                kpis.kpi.append(
                    esdl.DoubleKPI(
                        value=area_fixed_opex_cost / 1.0e6,
                        name="Fixed OPEX",
                        quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                            physicalQuantity=esdl.PhysicalQuantityEnum.COST,
                            unit=esdl.UnitEnum.EURO,
                            multiplier=esdl.MultiplierEnum.MEGA,
                        ),
                    )
                )

            # Top level KPIs: Cost breakdown in a polygon area (for all assest grouped together)
            kpi_name = f"{subarea.name}: Asset cost breakdown [EUR]"
            if is_asset_opex_included:
                kpis_top_level.kpi.append(
                    esdl.DistributionKPI(
                        name=kpi_name,
                        distribution=esdl.StringLabelDistribution(
                            stringItem=[
                                esdl.StringItem(label="Installation", value=area_installation_cost),
                                esdl.StringItem(label="Investment", value=area_investment_cost),
                                esdl.StringItem(
                                    label="Variable OPEX", value=area_variable_opex_cost
                                ),
                                esdl.StringItem(label="Fixed OPEX", value=area_fixed_opex_cost),
                            ]
                        ),
                        quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                            physicalQuantity=esdl.PhysicalQuantityEnum.COST, unit=esdl.UnitEnum.EURO
                        ),
                    )
                )
            else:
                kpis_top_level.kpi.append(
                    esdl.DistributionKPI(
                        name=kpi_name,
                        distribution=esdl.StringLabelDistribution(
                            stringItem=[
                                esdl.StringItem(label="Installation", value=area_installation_cost),
                                esdl.StringItem(label="Investment", value=area_investment_cost),
                            ]
                        ),
                        quantityAndUnit=esdl.esdl.QuantityAndUnitType(
                            physicalQuantity=esdl.PhysicalQuantityEnum.COST, unit=esdl.UnitEnum.EURO
                        ),
                    )
                )

            # Here we add a distribution KPI to the subarea to which gives a piechart
            # !!!!!!!!!!!!!!! This will only work if the source is in the area?
            # TODO: Still to be resolved since piecharts are still a work in progress in mapeditor
            # kpis.kpi.append(
            #     esdl.DistributionKPI(
            #         name="Energy breakdown ?",
            #         distribution=esdl.StringLabelDistribution(
            #             stringItem=[
            #                 esdl.StringItem(label=key, value=value) for key,
            #                 value in energy_breakdown.items()
            #             ]
            #         )
            #     )
            # )
            subarea.KPIs = kpis
        # end KPIs
        # ------------------------------------------------------------------------------------------
        # Placement
        for _, attributes in self.esdl_assets.items():
            name = attributes.name
            if name in [
                *self.heat_network_components.get("source", []),
                *self.heat_network_components.get("ates", []),
                *self.heat_network_components.get("buffer", []),
            ]:
                asset = _name_to_asset(name)
                asset_placement_var = self._asset_aggregation_count_var_map[name]
                placed = np.round(results[asset_placement_var][0]) >= 1.0
                max_size = results[self._asset_max_size_map[name]][0]

                if asset in self.heat_network_components.get("buffer", []):
                    asset.capacity = max_size
                    asset.volume = max_size / (
                        parameters[f"{name}.cp"]
                        * parameters[f"{name}.rho"]
                        * parameters[f"{name}.dT"]
                    )
                else:
                    asset.power = max_size
                if not placed:
                    asset.delete(recursive=True)
                else:
                    asset.state = esdl.AssetStateEnum.ENABLED

        if db_profiles:
            for name in [
                *self.heat_network_components.get("source", []),
                *self.heat_network_components.get("demand", []),
                *self.heat_network_components.get("pipe", []),
                *self.heat_network_components.get("buffer", []),
                *self.heat_network_components.get("ates", []),
                *self.heat_network_components.get("heat_exchanger", []),
                *self.heat_network_components.get("heat_pump", []),
            ]:
                asset = _name_to_asset(name)
                try:
                    # For all components dealing with one hydraulic system
                    for variable in ["Heat_flow", "HeatIn.Q", "HeatIn.H"]:
                        profile = esdl.InfluxDBProfile(
                            database="GROW results",
                            measurement=name,
                            field=variable,
                            port=443,
                            host="localhost",
                        )
                        asset.port[1].profile.append(profile)
                except Exception:
                    # For all components dealing with two hydraulic system
                    for variable in [
                        "Heat_flow",
                        "Primary.HeatIn.Q",
                        "Primary.HeatIn.H",
                        "Secondary.HeatIn.Q",
                        "Secondary.HeatIn.H",
                    ]:
                        profile = esdl.InfluxDBProfile(
                            database="GROW results",
                            measurement=name,
                            field=variable,
                            port=443,
                            host="localhost",
                        )
                        asset.port[1].profile.append(profile)

        # Pipes:
        edr_pipe_properties_to_copy = ["innerDiameter", "outerDiameter", "diameter", "material"]

        esh_edr = EnergySystemHandler()

        for pipe in self.hot_pipes:
            pipe_classes = self.pipe_classes(pipe)
            if not pipe_classes:
                # Nothing to change in the model
                continue

            pipe_class = self.get_optimized_pipe_class(pipe)
            cold_pipe = self.hot_to_cold_pipe(pipe)

            if parameters[f"{pipe}.diameter"] != 0.0 or any(np.abs(results[f"{pipe}.Q"]) > 1.0e-9):
                # if not isinstance(pipe_class, EDRPipeClass):
                #     assert pipe_class.name == f"{pipe}_orig"
                #     continue
                # print(results[f"{pipe}.Q"])
                # print(pipe + " has pipeclass: " + pipe_class.name )
                # print(pipe + f" has diameter: " + pipe_class.name)

                assert isinstance(pipe_class, EDRPipeClass)

                asset_edr = esh_edr.load_from_string(pipe_class.xml_string)

                for p in [pipe, cold_pipe]:
                    asset = _name_to_asset(p)
                    asset.state = esdl.AssetStateEnum.ENABLED
                    for prop in edr_pipe_properties_to_copy:
                        setattr(asset, prop, getattr(asset_edr, prop))
            else:
                for p in [pipe, cold_pipe]:
                    asset = _name_to_asset(p)
                    asset.delete(recursive=True)

        self.__optimized_energy_system_handler = esh
        self.optimized_esdl_string = esh.to_string()

        if self.esdl_string is None:
            if optimizer_sim:
                filename = run_info.esdl_file.with_name(
                    f"{run_info.esdl_file.stem}_SmartControlOptimized.esdl"
                )
            else:
                filename = run_info.esdl_file.with_name(
                    f"{run_info.esdl_file.stem}_Simulation.esdl"
                )
            esh.save(str(filename))
