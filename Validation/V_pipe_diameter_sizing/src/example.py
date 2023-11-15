from pathlib import Path
from unittest import TestCase
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from esdl import esdl
import yaml
import pywanda
from rtctools.util import run_optimization_problem
from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.esdl.esdl_mixin import _esdl_to_assets
from esdl.esdl_handler import EnergySystemHandler

from rtctools_heat_network.workflows import EndScenarioSizing
from rtctools_heat_network.pipe_class import PipeClass
import examples.pipe_diameter_sizing.src.example  # noqa: E402, I100
from examples.pipe_diameter_sizing.src.example import (
            PipeDiameterSizingProblem,
        )


def pipe_classes():
    return [
        PipeClass("DN40", 0.0431, 1.5, (0.179091, 0.005049), 1.0),
        PipeClass("DN50", 0.0545, 1.7, (0.201377, 0.006086), 1.0),
        PipeClass("DN65", 0.0703, 1.9, (0.227114, 0.007300), 1.0),
        PipeClass("DN80", 0.0825, 2.2, (0.238244, 0.007611), 1.0),
        PipeClass("DN100", 0.1071, 2.4, (0.247804, 0.007386), 1.0),
        PipeClass("DN125", 0.1325, 2.6, (0.287779, 0.009431), 1.0),
        PipeClass("DN150", 0.1603, 2.8, (0.328592, 0.011567), 1.0),
        PipeClass("DN200", 0.2101, 3.0, (0.346285, 0.011215), 1.0),
        PipeClass("DN250", 0.263, 3.0, (0.334606, 0.009037), 1.0),
        PipeClass("DN300", 0.3127, 3.0, (0.384640, 0.011141), 1.0),
        PipeClass("DN350", 0.3444, 3.0, (0.368061, 0.009447), 1.0),
        PipeClass("DN400", 0.3938, 3.0, (0.381603, 0.009349), 1.0),
        PipeClass("DN450", 0.4444, 3.0, (0.380070, 0.008506), 1.0),
        PipeClass("DN500", 0.4954, 3.0, (0.369282, 0.007349), 1.0),
        PipeClass("DN600", 0.5954, 3.0, (0.431023, 0.009155), 1.0),
    ]

def esdl_model_setup(esdl_file, esdl_path, input_file):
    with open(input_file, 'r') as file:
         val_input = yaml.safe_load(file)
    esh = EnergySystemHandler()
    es = esh.load_file(esdl_file)
    esdl_assets = esh.get_all_instances_of_type(esdl.Asset)
    esdl_carriers = esh.get_all_instances_of_type(esdl.HeatCommodity)
    esdl_carriers[0].supplyTemperature =  float(val_input['Producer']['Constant temperature'])
    esdl_carriers[1].returnTemperature = float(val_input['Consumer']['Return temperature'])
    for asset in esdl_assets:
        if isinstance(asset, esdl.Pipe):
            asset.length = float(val_input['Pipe']['Length'])
            asset.roughness = float(val_input['Pipe']['Wall roughness'])
            asset.state = 'OPTIONAL'
        if isinstance(asset, esdl.HeatingDemand):
            asset.power = -1*float(val_input['Consumer']['Initial heat supply'])
        if isinstance(asset, esdl.ResidualHeatSource):
            asset.power = float(val_input['Producer']['Power'])


    esh.save(esdl_path + 'newesdl.esdl')

    # TODO add option to adjust costs


def wanda_model_setup(wanda_file,wanda_bin,input_file):
    with open(input_file, 'r') as file:
        val_input = yaml.safe_load(file)
    wanda_model = pywanda.WandaModel(wanda_file, wanda_bin)
    wanda_model.switch_to_unit_SI()

    for component in wanda_model.get_all_components():
        if component.get_name() == 'P_supply':
            for key, value in val_input['Pipe'].items():
                prop = component.get_property(key)
                if type(value) == (int or float):
                    prop.set_scalar(value)
                # elif value == table:
                #     prop.set_table(value)
        if component.get_name() == 'P_return':
            for key, value in val_input['Pipe'].items():
                prop = component.get_property(key)
                if type(value) == (int or float):
                    prop.set_scalar(value)
                # elif value == table:
                #     prop.set_table(value)
        if component.get_name() == 'Heating_demand':
            for key, value in val_input['Consumer'].items():
                prop = component.get_property(key)
                if type(value) == (int or float):
                    prop.set_scalar(value)
                # elif value == table:
                #     prop.set_table(value)
        if component.get_name() == 'Source_supply':
            for key, value in val_input['Producer'].items():
                prop = component.get_property(key)
                if type(value) == (int or float):
                    prop.set_scalar(value)
                # elif value == table:
                #     prop.set_table(value)
        if component.get_name() == 'Valve':
            for key, value in val_input['Producer'].items():
                if key == 'Return temperature':
                    prop = component.get_property('Initial downstream temperature')
                    prop.set_scalar(value)
    wanda_model.save_model_input()
    wanda_model.close()

def wanda_model_run_stst(wanda_file,wanda_bin,diameters):
    Q_loss = []
    dh = []
    print('here')

    for diameter in diameters:
        wanda_model = pywanda.WandaModel(wanda_file, wanda_bin)
        wanda_model.reload_input()
        for component in wanda_model.get_all_components():
            if component.get_name() == 'P_supply' or component.get_name() == 'P_return':
                prop = component.get_property('Inner diameter')
                prop.set_scalar(diameter)
        print(' start run_wanda_model for diameter ', diameter)
        wanda_model.save_model_input()
        wanda_model.run_steady()
        print('finished run_wanda_model for diameter ', diameter)
        # print('run_wanda_model for diameter', diameter)
        wanda_model.reload_output()
        Q_loss.append(wanda_model.get_component('pipe_supply').get_property('Total heat flux'))
        H1 = wanda_model.get_component('pipe_supply').get_property('Head1')
        H2 = wanda_model.get_component('pipe_supply').get_property('Head2')
        dh.append(H1-H2)
        wanda_model.close()
    return Q_loss, dh

class ValidationPipeDiameterSizing(ESDLMixin):
    # def __init__(self):


    def pre(self):
        super().pre()


    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["peak_day_index"] = self.__indx_max_peak
        parameters["time_step_days"] = self.__day_steps
        parameters["number_of_years"] = self._number_of_years
        return parameters


    @property
    def esdl_assets(self):
        assets = super().esdl_assets

        # Store parameter values from ESDL
        for asset in assets.values():
            if isinstance(asset.esdl_asset, esdl.Pipe):
                asset.attributes
            if isinstance(asset.esdl_asset, esdl.HeatingDemand):
                asset.attributes
            if isinstance(asset.esdl_asset, esdl.ResidualHeatSource):
                asset.attributes

        return assets

    def post(self):
        super().post()

        results = self.extract_results()
        parameters = self.parameters(0)
        data_milp = {}  # Data storage

        # Pressure drop [Pa]
        data_milp = {"Pipe1_supply_dPress": results["Pipe1.dH"] * parameters["Pipe1.rho"] * 9.81}
        data_milp.update(
            {"Pipe1_return_dPress": results["Pipe1_ret.dH"] * parameters["Pipe1_ret.rho"] * 9.81}
        )

        # Volumetric flow [m3/s]
        data_milp.update({"Pipe1_supply_Q": results["Pipe1.HeatOut.Q"]})
        data_milp.update({"Pipe1_return_Q": results["Pipe1_ret.HeatOut.Q"]})

        # Mass flow [kg/s]
        data_milp.update(
            {
                "Pipe1_supply_mass_flow": results["HeatingDemand_1.Heat_demand"]
                / parameters["HeatingDemand_1.cp"]
                / parameters["HeatingDemand_1.dT"]
            }
        )
        data_milp.update(
            {
                "Pipe1_return_mass_flow": results["HeatingDemand_1.Heat_demand"]
                / parameters["HeatingDemand_1.cp"]
                / parameters["HeatingDemand_1.dT"]
            }
        )

        # Flow velocity [m/s]
        data_milp.update(
            {
                "Pipe1_supply_flow_vel": data_milp["Pipe1_supply_mass_flow"]
                / parameters["Pipe1.rho"]
                / parameters["Pipe1.area"]
            }
        )
        data_milp.update(
            {
                "Pipe1_return_flow_vel": data_milp["Pipe1_return_mass_flow"]
                / parameters["Pipe1_ret.rho"]
                / parameters["Pipe1_ret.area"]
            }
        )

        # Pipe deltaT
        data_milp.update({"Pipe1_supply_dT": parameters["Pipe1.dT"]})
        data_milp.update({"Pipe1_return_dT": parameters["Pipe1_ret.dT"]})

        # Heat source, demand and loss [W]
        data_milp.update({"Heat_source": results["ResidualHeatSource_1.Heat_source"]})
        data_milp.update({"Heat_demand": results["HeatingDemand_1.Heat_demand"]})
        data_milp.update(
            {
                "Heat_loss": results["ResidualHeatSource_1.Heat_source"]
                - results["HeatingDemand_1.Heat_demand"]
            }
        )

        # Hydraulic power via linearized method in MILP [W]
        data_milp.update({"Pipe1_supply_Hydraulic_power": results["Pipe1.Hydraulic_power"]})
        data_milp.update({"Pipe1_return_Hydraulic_power": results["Pipe1_ret.Hydraulic_power"]})



if __name__ == "__main__":

    wanda_model = r'C:\Users\star_kj\NWN-rtc-tools-heat-network\Validation\V_pipe_diameter_sizing\model\validation_pipe_diameter_sizing_46.wdi'
    esdl_model = r'C:\Users\star_kj\NWN-rtc-tools-heat-network\Validation\V_pipe_diameter_sizing\model\test_simple.esdl'
    esdl_string = 'test_simple.esdl'
    esdl_path =  r'C:\Users\star_kj\NWN-rtc-tools-heat-network\Validation\V_pipe_diameter_sizing\model\\'
    wanda_bin = r'c:\Program Files (x86)\Deltares\Wanda 4.6\Bin\\'
    esdl2wanda_config = r'.\esdl2wanda_cfg.yml'
    val_input_file = r'C:\Users\star_kj\NWN-rtc-tools-heat-network\Validation\src\input.yml'
    base_folder = r'C:\Users\star_kj\NWN-rtc-tools-heat-network\Validation\V_pipe_diameter_sizing\\'

    # wanda_model_setup(wanda_model, wanda_bin, val_input_file)
    esdl_model_setup(esdl_model, esdl_path, val_input_file)

    # sol = run_optimization_problem(ValidationPipeDiameterSizing)
    # run optimization routine
    #PipeDiameterSizingProblem
    sol = run_optimization_problem(PipeDiameterSizingProblem)
    print(sol.hot_pipes)

    # sol = run_optimization_problem(EndScenarioSizing, base_folder=base_folder)

    parameters = sol.parameters(0)
    diameters = {p: parameters[f"{p}.diameter"] for p in sol.hot_pipes}
    print(diameters)
    results = sol.extract_results()
    exit(0)
    # EndScenarioSizingHIGHS
    # sol = run_optimization_problem(
    #     EndScenarioSizingHIGHS
    # )


    # run a series of steady state for the wanda model with different diameters
    diameters = pipe_classes()

    # parameters = sol.parameters(0)
    # diameter_grow = {p: parameters[f"{p}.diameter"] for p in sol.hot_pipes}
    # print(diameter_grow)
    # diameters = {p: parameters[f"{p}.diameter"] for p in sol.hot_pipes}
    # results = sol.extract_results()

    # run wanda model
    # for D in list:
    #     open wanda model
    #     set new diameters
    #     save wanda model
    #     run wanda model
    #     obtain         (results)

    data_wanda = {}
    inner_diameters = [0.1,0.2,0.3,0.4]
    Q_loss, dh = wanda_model_run_stst(wanda_model, wanda_bin, inner_diameters)


    data_wanda = pd.DataFrame({"Diameter": ['DN100', 'DN200', 'DN300', 'DN400'],
                               "Inner diameter": inner_diameters,
                       "Q_loss": Q_loss,
                        "dh": dh}
                              )
    print(data_wanda)

    #plot results
    markers = ["o", "v", "d", "s", "^", "<", ">"]
    fig,ax = plt.subplot(1,1,figsize=(5,6))
    plt.xlabel('Inner diameter [mm]')
    plt.ylabel('Energy [W/m]')
    x = np.range(len(data_wanda['Diameter']))
    Q_loss = data_wanda['Q_loss']
    E_pump = data_wanda['dh']
    tot_energy = Q_loss + E_pump
    plt.plot(x, Q_loss, label='Q_loss')

    index_grow = data_wanda.loc[data_wanda['Diamter'] == diameter_grow].index[0]
    plt.plot([index_grow, index_grow ],[-1E6,1E6],'k--',label='result grow workflow')
    plt.legend()

    my_xticks = data_wanda[ 'Diameter']
    plt.xticks(x, my_xticks)
    ax.set_xlim([0,(len(x)+1)])
    plt.savefig(output_folder + '\\figures\\physics_validation.png',bbox_inches='tight')
    plt.close()



