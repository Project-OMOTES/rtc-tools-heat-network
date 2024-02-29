<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" esdlVersion="v2401" name="sourcesink with return network with return network with return network" version="3" id="5d539f68-f98e-466b-9ff5-b908a211e0ab_with_return_network_with_return_network_with_return_network" description="">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="11f4eafa-7fbc-4d82-b346-e893326d2c30">
    <carriers xsi:type="esdl:Carriers" id="eafbd8f4-1fde-4bb5-8dce-fdb74a1a1097">
      <carrier xsi:type="esdl:HeatCommodity" id="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="heat" supplyTemperature="70.0"/>
      <carrier xsi:type="esdl:HeatCommodity" id="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="heat_ret" returnTemperature="40.0"/>
    </carriers>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" id="90e7e098-038e-4462-89fe-a8852c501753" name="Untitled instance">
    <area xsi:type="esdl:Area" name="Untitled area" id="4fd1adc2-5371-4ab7-806a-b40e49d127e9">
      <asset xsi:type="esdl:HeatProducer" name="source" id="a479e4e6-6f75-460d-aeb2-d0e3e02314e0" power="10000000.0">
        <port xsi:type="esdl:OutPort" id="b0b1a87c-7b5a-4edb-a732-274d1bf69647" connectedTo="b639ef7a-58ad-49e5-aff6-65dec0b482db" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <port xsi:type="esdl:InPort" id="622d7e19-e360-46af-bfbf-eb35ec14548b" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="In" connectedTo="72f25aaf-c12b-4d92-b7ff-e8345445f09c"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.08646829489945" lon="4.386527538299561"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="demand" id="f6d5923d-ba9a-409d-80a0-26f73b2a574b" power="10000000.0">
        <port xsi:type="esdl:InPort" id="b8849fb5-fe97-48d9-91a8-9abcbf365738" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="fae038d9-ec38-4006-a7e3-dcd27539eea0"/>
        <port xsi:type="esdl:OutPort" id="eb68d4fe-b361-4e64-9f54-a1e05e5712ee" connectedTo="47457db7-1f8a-4a86-b941-40b3ac0502e3" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.086586960901776" lon="4.398479461669923"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe1" length="284.44" related="Pipe1_ret" outerDiameter="0.5" id="Pipe1" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="7de138f5-f934-4717-87c0-b31cc277f64c">
          <investmentCosts xsi:type="esdl:SingleValue" id="7ffb929c-9640-436c-907a-556e790a6c7d" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="b639ef7a-58ad-49e5-aff6-65dec0b482db" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="b0b1a87c-7b5a-4edb-a732-274d1bf69647"/>
        <port xsi:type="esdl:OutPort" id="b56c1473-5182-4b0a-97ce-551392c2b0a5" connectedTo="8c3d86b1-95da-41ef-938c-1027e451461e" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.08646829489945" lon="4.386527538299561"/>
          <point xsi:type="esdl:Point" lat="52.086494665149445" lon="4.390690326690675"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="47a862f8-97be-4f97-aa8c-38c940278d33" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ef7" id="7ef7856b-e168-4701-8b83-90dbf7627701">
        <port xsi:type="esdl:InPort" id="8c3d86b1-95da-41ef-938c-1027e451461e" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="b56c1473-5182-4b0a-97ce-551392c2b0a5"/>
        <port xsi:type="esdl:OutPort" id="051cf195-18ad-439b-887c-575e7326815c" connectedTo="2b533bb9-57a0-4db7-9cc3-c216e6518e92 b2b74f24-4cc5-4d39-897c-d6211b77c53b" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <geometry xsi:type="esdl:Point" lat="52.086494665149445" lon="4.390690326690675"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe2" length="500.00" related="Pipe2_ret" outerDiameter="0.5" id="Pipe2" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="7de138f5-f934-4717-87c0-b31cc277f64c">
          <investmentCosts xsi:type="esdl:SingleValue" id="7ffb929c-9640-436c-907a-556e790a6c7d" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="2b533bb9-57a0-4db7-9cc3-c216e6518e92" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="051cf195-18ad-439b-887c-575e7326815c"/>
        <port xsi:type="esdl:OutPort" id="64b9bc08-b7b9-4370-b21b-4e5b043c8b61" connectedTo="b9119eab-62b1-4c0d-8820-af4e1b2794da" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.086494665149445" lon="4.390690326690675"/>
          <point xsi:type="esdl:Point" lat="52.08648807258842" lon="4.390711784362794"/>
          <point xsi:type="esdl:Point" lat="52.08824825180137" lon="4.392396211624146"/>
          <point xsi:type="esdl:Point" lat="52.08648148002641" lon="4.394370317459107"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="47a862f8-97be-4f97-aa8c-38c940278d33" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe3" length="281.01" related="Pipe3_ret" outerDiameter="0.5" id="Pipe3" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="7de138f5-f934-4717-87c0-b31cc277f64c">
          <investmentCosts xsi:type="esdl:SingleValue" id="7ffb929c-9640-436c-907a-556e790a6c7d" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="5b77e7e6-dc9c-4f0e-9473-9bfe423b00a8" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="aa278a36-39b3-4fd3-b2ae-5fd652735895"/>
        <port xsi:type="esdl:OutPort" id="fae038d9-ec38-4006-a7e3-dcd27539eea0" connectedTo="b8849fb5-fe97-48d9-91a8-9abcbf365738" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.08648148002641" lon="4.394370317459107"/>
          <point xsi:type="esdl:Point" lat="52.08648148002641" lon="4.394381046295167"/>
          <point xsi:type="esdl:Point" lat="52.086586960901776" lon="4.398479461669923"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="47a862f8-97be-4f97-aa8c-38c940278d33" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_44da" id="44da7e93-ed3f-4738-afaf-e152b9ffbf80">
        <port xsi:type="esdl:InPort" id="b9119eab-62b1-4c0d-8820-af4e1b2794da" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="64b9bc08-b7b9-4370-b21b-4e5b043c8b61 e8d1d887-b952-49e9-b236-f4e208f82294"/>
        <port xsi:type="esdl:OutPort" id="aa278a36-39b3-4fd3-b2ae-5fd652735895" connectedTo="5b77e7e6-dc9c-4f0e-9473-9bfe423b00a8" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <geometry xsi:type="esdl:Point" lat="52.08648148002641" lon="4.394370317459107"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe4" length="100000.0" related="Pipe4_ret" outerDiameter="0.5" id="Pipe4" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="e7cd8135-cc6c-43a7-96b2-7a7ac4cc3de8">
          <investmentCosts xsi:type="esdl:SingleValue" id="7ffb929c-9640-436c-907a-556e790a6c7d" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="b2b74f24-4cc5-4d39-897c-d6211b77c53b" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="In" connectedTo="051cf195-18ad-439b-887c-575e7326815c"/>
        <port xsi:type="esdl:OutPort" id="e8d1d887-b952-49e9-b236-f4e208f82294" connectedTo="b9119eab-62b1-4c0d-8820-af4e1b2794da" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.086494665149445" lon="4.390690326690675"/>
          <point xsi:type="esdl:Point" lat="52.08352791428881" lon="4.392278194427491"/>
          <point xsi:type="esdl:Point" lat="52.08648148002641" lon="4.394370317459107"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="47a862f8-97be-4f97-aa8c-38c940278d33" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ef7_ret" id="c7f7f905-2475-4cc1-b045-c10527f79cdf">
        <port xsi:type="esdl:OutPort" id="5cb42b39-932b-4429-a629-11b9a6aff9c8" connectedTo="fb7a7f10-408f-425d-8b52-f8846162d0a1" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="ret_port"/>
        <port xsi:type="esdl:InPort" id="e4b825f9-bb86-4a60-afca-e73479e488f1" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="ret_port" connectedTo="5bf80172-a781-4949-8873-b6424a555e87 071264ec-7ecb-419c-8206-729eb3767145"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.08658466523944" lon="4.390176379264867"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_44da_ret" id="4659d15d-dfba-4148-89e2-93fb4f7fd972">
        <port xsi:type="esdl:InPort" id="ff22fb00-4105-4553-ae79-9bf19216d951" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="ret_port" connectedTo="852b8309-1d05-49fd-b75e-513765e71613"/>
        <port xsi:type="esdl:OutPort" id="65f30879-5a89-4034-97bb-c52842fda0ef" connectedTo="6e8476b4-7eff-4b03-8695-708dac6e7de3 4c542131-05d5-4d44-9dba-7515b0f98eb2" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="ret_port"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.086571480116405" lon="4.393856343516998"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe1_ret" length="284.44" related="Pipe1" outerDiameter="0.5" id="Pipe1_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="fb7a7f10-408f-425d-8b52-f8846162d0a1" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="In_ret" connectedTo="5cb42b39-932b-4429-a629-11b9a6aff9c8"/>
        <port xsi:type="esdl:OutPort" id="72f25aaf-c12b-4d92-b7ff-e8345445f09c" connectedTo="622d7e19-e360-46af-bfbf-eb35ec14548b" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.08658466523944" lon="4.390176379264867"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.086558294989445" lon="4.386013537838319"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe2_ret" length="500.00" related="Pipe2" outerDiameter="0.5" id="Pipe2_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="6e8476b4-7eff-4b03-8695-708dac6e7de3" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="In_ret" connectedTo="65f30879-5a89-4034-97bb-c52842fda0ef"/>
        <port xsi:type="esdl:OutPort" id="5bf80172-a781-4949-8873-b6424a555e87" connectedTo="e4b825f9-bb86-4a60-afca-e73479e488f1" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.086571480116405" lon="4.393856343516998"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.088338251891365" lon="4.391885765796076"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.08657807267842" lon="4.390197823679189"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.08658466523944" lon="4.390176379264867"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe3_ret" length="281.01" related="Pipe3" outerDiameter="0.5" id="Pipe3_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="47457db7-1f8a-4a86-b941-40b3ac0502e3" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="In_ret" connectedTo="eb68d4fe-b361-4e64-9f54-a1e05e5712ee"/>
        <port xsi:type="esdl:OutPort" id="852b8309-1d05-49fd-b75e-513765e71613" connectedTo="ff22fb00-4105-4553-ae79-9bf19216d951" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.086676960991774" lon="4.39796569977892"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.086571480116405" lon="4.393867072353058"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.086571480116405" lon="4.393856343516998"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe4_ret" length="100000.00" related="Pipe4" outerDiameter="0.5" id="Pipe4_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="4c542131-05d5-4d44-9dba-7515b0f98eb2" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="In_ret" connectedTo="65f30879-5a89-4034-97bb-c52842fda0ef"/>
        <port xsi:type="esdl:OutPort" id="071264ec-7ecb-419c-8206-729eb3767145" connectedTo="e4b825f9-bb86-4a60-afca-e73479e488f1" carrier="435a0034-fab0-4e7e-9a17-edf8de9a2b11_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.086571480116405" lon="4.393856343516998"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.083617914378806" lon="4.391758208579233"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.08658466523944" lon="4.390176379264867"/>
        </geometry>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
