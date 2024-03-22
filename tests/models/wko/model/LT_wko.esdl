<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" esdlVersion="v2401" name="Untitled EnergySystem with return network" version="1" id="03386224-685c-4f3b-bcc0-3c9dc63110de_with_return_network" description="">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="ad1a8ca8-b21b-4ad3-aa3e-756cf321bd4d">
    <carriers xsi:type="esdl:Carriers" id="553fa302-8878-4fa9-b0ca-6ee6123bf2c9">
      <carrier xsi:type="esdl:HeatCommodity" id="13db0822-98da-4ca8-9d48-868653ae06af" supplyTemperature="25.0" name="LT"/>
      <carrier xsi:type="esdl:HeatCommodity" id="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="LT_ret" returnTemperature="6.0"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="632648b0-a234-4e0c-b3a2-f18a6dee2700">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b" unit="WATT" description="Power in MW" physicalQuantity="POWER" multiplier="MEGA"/>
    </quantityAndUnits>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" id="d1a23619-2ef0-4172-8bab-25dfc28a64e1" name="Untitled Instance">
    <area xsi:type="esdl:Area" name="Untitled Area" id="4f36ae5d-6dc1-465d-8fc0-f5459d5c3e27">
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_9b90" id="9b904958-feed-40d0-934e-fe4e4420d916" power="5000000.0">
        <port xsi:type="esdl:InPort" id="62f09402-d0ef-455c-86e7-e1b8f39a5542" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="2272c6aa-15ad-4ce1-81b5-720b8bf741a2"/>
        <port xsi:type="esdl:OutPort" id="8e360471-beba-4bcf-bbd3-466259c51b5b" connectedTo="d490f22d-4a73-46a8-a208-8b1eaf709c2a" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.079796120944025" lon="4.41802740097046"/>
      </asset>
      <asset xsi:type="esdl:CoolingDemand" name="CoolingDemand_15e8" id="15e803b4-1224-4cac-979f-87747a656741" power="5000000.0">
        <port xsi:type="esdl:InPort" id="4e4b0784-2205-4937-af8c-35f33f7c20b8" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="2f34df96-a54e-4c16-9b2d-7b8a95bb1c51"/>
        <port xsi:type="esdl:OutPort" id="167a5468-c9b4-46c4-9815-1fbdeeb50420" connectedTo="2b30a20f-5f48-4957-9cc0-dceb5df79020" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.07984886931361" lon="4.4191861152648935"/>
      </asset>
      <asset xsi:type="esdl:Losses" name="Losses_109f" id="109f946b-373c-4510-bedb-30752a0cd576" power="100000.0">
        <port xsi:type="esdl:InPort" id="52a0a5b2-8798-4fcd-a2e2-361720ff6d11" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="71466743-5d37-4ec2-8935-eb6fb226bbaf"/>
        <port xsi:type="esdl:OutPort" id="44332191-50e2-4e34-b318-2090863ae0d6" connectedTo="2ddf7e12-8df0-4de8-846b-f8e1db77fb14" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out"/>
        <geometry xsi:type="esdl:Point" lat="52.07975655962592" lon="4.416847229003907"/>
      </asset>
      <asset xsi:type="esdl:ATES" name="ATES_226d" id="226d58d1-28e5-4d73-9e72-3aaf3a5c67ff" maxStorageTemperature="30.0" maxChargeRate="1000000.0" minStorageTemperature="6.0" maxDischargeRate="1000000.0">
        <port xsi:type="esdl:InPort" id="39616896-fb2c-470e-9567-25233db228af" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="435c948d-3801-4ffd-9a37-8ffb1d046390"/>
        <port xsi:type="esdl:OutPort" id="316be022-cc67-4336-9d1b-6898bda3cd96" connectedTo="b69790e9-58be-4c88-b8b3-4238b27cd314" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.077079495639495" lon="4.418177604675294"/>
      </asset>
      <asset xsi:type="esdl:HeatPump" name="HeatPump_b97e" COP="4.0" id="b97e7c4f-fff5-4e4a-bc64-830563f94e4c" power="5000000.0">
        <port xsi:type="esdl:InPort" id="2083140e-6ddb-4d26-a788-a36ebcf65b80" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="SecIn" connectedTo="679b496b-30c7-461d-bec0-e0e85f512989"/>
        <port xsi:type="esdl:OutPort" id="ad87a98f-e6a7-4688-9989-ea8a17a85afc" connectedTo="89d279c5-f072-4a0d-9117-4274b0c7eaaa" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="SecOut"/>
        <geometry xsi:type="esdl:Point" lat="52.07816089056798" lon="4.413928985595704"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_913e" id="913edda0-8cf9-4291-9ade-011751929a4b">
        <port xsi:type="esdl:InPort" id="58635b77-b097-459d-995d-58eedde3a267" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="362e9c25-54dd-426b-981f-0cde947a035b"/>
        <port xsi:type="esdl:OutPort" id="2925bc5d-077c-4603-9d9e-1f452926f504" connectedTo="59f66704-9491-4240-b0dd-0df181ac24a9 7962e4fc-2ca8-4c23-9641-0fb46c77dd26 0006482e-db8a-4260-9739-02971e0bef3c 4e659925-f8ba-4653-83c0-abee4af6afc1" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.078371891401986" lon="4.417941570281983"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe1" length="275.2" related="Pipe1_ret" outerDiameter="0.45" id="Pipe1" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="df0bd066-2e08-4235-8b55-249b7477e21a">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="89d279c5-f072-4a0d-9117-4274b0c7eaaa" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="ad87a98f-e6a7-4688-9989-ea8a17a85afc"/>
        <port xsi:type="esdl:OutPort" id="362e9c25-54dd-426b-981f-0cde947a035b" connectedTo="58635b77-b097-459d-995d-58eedde3a267" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.07816089056798" lon="4.413928985595704"/>
          <point xsi:type="esdl:Point" lat="52.078371891401986" lon="4.417941570281983"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe2" length="144.6" related="Pipe2_ret" outerDiameter="0.45" id="Pipe2" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="bccd9ceb-e3a0-4314-a223-25947bd976fb">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="59f66704-9491-4240-b0dd-0df181ac24a9" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="2925bc5d-077c-4603-9d9e-1f452926f504"/>
        <port xsi:type="esdl:OutPort" id="435c948d-3801-4ffd-9a37-8ffb1d046390" connectedTo="39616896-fb2c-470e-9567-25233db228af" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.078371891401986" lon="4.417941570281983"/>
          <point xsi:type="esdl:Point" lat="52.077079495639495" lon="4.418177604675294"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe4" length="171.2" related="Pipe4_ret" outerDiameter="0.45" id="Pipe4" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="5ae81141-3307-4758-8379-19eab097ead7">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="7962e4fc-2ca8-4c23-9641-0fb46c77dd26" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="2925bc5d-077c-4603-9d9e-1f452926f504"/>
        <port xsi:type="esdl:OutPort" id="71466743-5d37-4ec2-8935-eb6fb226bbaf" connectedTo="52a0a5b2-8798-4fcd-a2e2-361720ff6d11" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.078371891401986" lon="4.417941570281983"/>
          <point xsi:type="esdl:Point" lat="52.07975655962592" lon="4.416847229003907"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe3" length="158.5" related="Pipe3_ret" outerDiameter="0.45" id="Pipe3" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="1a6b0452-459c-45ed-a2cb-093f71a79780">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="0006482e-db8a-4260-9739-02971e0bef3c" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="2925bc5d-077c-4603-9d9e-1f452926f504"/>
        <port xsi:type="esdl:OutPort" id="2272c6aa-15ad-4ce1-81b5-720b8bf741a2" connectedTo="62f09402-d0ef-455c-86e7-e1b8f39a5542" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.078371891401986" lon="4.417941570281983"/>
          <point xsi:type="esdl:Point" lat="52.079796120944025" lon="4.41802740097046"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe5" length="184.9" related="Pipe5_ret" outerDiameter="0.45" id="Pipe5" diameter="DN300" innerDiameter="0.3127">
        <costInformation xsi:type="esdl:CostInformation" id="c77c2731-8d9f-475e-9af0-38b8fffff72c">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST" perUnit="METRE"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="4e659925-f8ba-4653-83c0-abee4af6afc1" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="In" connectedTo="2925bc5d-077c-4603-9d9e-1f452926f504"/>
        <port xsi:type="esdl:OutPort" id="2f34df96-a54e-4c16-9b2d-7b8a95bb1c51" connectedTo="4e4b0784-2205-4937-af8c-35f33f7c20b8" carrier="13db0822-98da-4ca8-9d48-868653ae06af" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.078371891401986" lon="4.417941570281983"/>
          <point xsi:type="esdl:Point" lat="52.07984886931361" lon="4.4191861152648935"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_913e_ret" id="d7f51234-09e4-4441-9ce9-f7dbca035bf8">
        <port xsi:type="esdl:OutPort" id="98d160f7-9307-44be-99a4-5d0bc2e0d5ea" connectedTo="ff09f46c-4269-4d4e-88a3-eeeab46448c8" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="ret_port"/>
        <port xsi:type="esdl:InPort" id="fb43509e-552d-4c76-a2e7-2706691c0851" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="ret_port" connectedTo="e4f073eb-04e8-43ed-8599-19e112d7a758 4dd7277d-c9cf-42b3-8f21-195be0c95d6f 6ceff0d3-fb78-4c87-b477-bcd2ec283051 6185d23a-aca6-4962-bb09-07f451e7d043"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.07846189149198" lon="4.417410734103896"/>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe1_ret" length="275.2" related="Pipe1" outerDiameter="0.45" id="Pipe1_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="ff09f46c-4269-4d4e-88a3-eeeab46448c8" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="In_ret" connectedTo="98d160f7-9307-44be-99a4-5d0bc2e0d5ea"/>
        <port xsi:type="esdl:OutPort" id="679b496b-30c7-461d-bec0-e0e85f512989" connectedTo="2083140e-6ddb-4d26-a788-a36ebcf65b80" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07846189149198" lon="4.417410734103896"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.078250890657976" lon="4.413397695431759"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe2_ret" length="144.6" related="Pipe2" outerDiameter="0.45" id="Pipe2_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="b69790e9-58be-4c88-b8b3-4238b27cd314" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="In_ret" connectedTo="316be022-cc67-4336-9d1b-6898bda3cd96"/>
        <port xsi:type="esdl:OutPort" id="e4f073eb-04e8-43ed-8599-19e112d7a758" connectedTo="fb43509e-552d-4c76-a2e7-2706691c0851" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07716949572949" lon="4.417643975184299"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07846189149198" lon="4.417410734103896"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe3_ret" length="158.5" related="Pipe3" outerDiameter="0.45" id="Pipe3_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="d490f22d-4a73-46a8-a208-8b1eaf709c2a" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="In_ret" connectedTo="8e360471-beba-4bcf-bbd3-466259c51b5b"/>
        <port xsi:type="esdl:OutPort" id="4dd7277d-c9cf-42b3-8f21-195be0c95d6f" connectedTo="fb43509e-552d-4c76-a2e7-2706691c0851" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07988612103402" lon="4.417499608352213"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07846189149198" lon="4.417410734103896"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe4_ret" length="171.2" related="Pipe4" outerDiameter="0.45" id="Pipe4_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="2ddf7e12-8df0-4de8-846b-f8e1db77fb14" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="In_ret" connectedTo="44332191-50e2-4e34-b318-2090863ae0d6"/>
        <port xsi:type="esdl:OutPort" id="6ceff0d3-fb78-4c87-b477-bcd2ec283051" connectedTo="fb43509e-552d-4c76-a2e7-2706691c0851" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07984655971592" lon="4.41631935232941"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07846189149198" lon="4.417410734103896"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe5_ret" length="184.9" related="Pipe5" outerDiameter="0.45" id="Pipe5_ret" diameter="DN300" innerDiameter="0.3127">
        <port xsi:type="esdl:InPort" id="2b30a20f-5f48-4957-9cc0-dceb5df79020" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="In_ret" connectedTo="167a5468-c9b4-46c4-9815-1fbdeeb50420"/>
        <port xsi:type="esdl:OutPort" id="6185d23a-aca6-4962-bb09-07f451e7d043" connectedTo="fb43509e-552d-4c76-a2e7-2706691c0851" carrier="13db0822-98da-4ca8-9d48-868653ae06af_ret" name="Out_ret"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.079938869403605" lon="4.418658434678616"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.07846189149198" lon="4.417410734103896"/>
        </geometry>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
