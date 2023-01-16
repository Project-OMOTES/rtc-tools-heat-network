<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" version="5" name="Untitled EnergySystem with return network" id="0735d19f-39a4-463d-bf28-cf0b28b88bef_with_return_network" esdlVersion="v2207" description="">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="b7ebaafe-597c-4055-bc88-b47cdfa84e34">
    <carriers xsi:type="esdl:Carriers" id="5d5909a4-18ec-4aa3-a08a-1e9539a20be2">
      <carrier xsi:type="esdl:HeatCommodity" name="heat1" id="d336e381-ca6f-442e-985e-9f4c2bec1efe" supplyTemperature="70.0"/>
      <carrier xsi:type="esdl:HeatCommodity" name="heat2" id="72126c73-87e9-4bf6-99cf-d02a6c07010c" supplyTemperature="90.0"/>
      <carrier xsi:type="esdl:HeatCommodity" name="heat1_ret" returnTemperature="40.0" id="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      <carrier xsi:type="esdl:HeatCommodity" name="heat2_ret" returnTemperature="50.0" id="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
    </carriers>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" name="MultipleCarrierTest" id="82399ebf-5e52-465d-830a-b502d6c1012b">
    <area xsi:type="esdl:Area" id="e91b7896-ae7a-4136-8e96-d98d40c49fc2" name="MultipleCarrierTest">
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_3322" id="3322fe41-f73b-4ba4-b87b-bf8315aa69e4" power="10000000.0">
        <geometry xsi:type="esdl:Point" lon="4.37633514404297" CRS="WGS84" lat="52.09026545046112"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="43e98a06-8db5-43a1-913b-e8e7f255fc3f" id="06b6b748-d052-4fd5-a017-ff76321284a9" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="01ea44bf-6f10-4913-bd41-5e5b210d3b40" connectedTo="46e659aa-0e2b-43af-b38c-242cb7fdc480" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_18aa" id="18aabae2-c92a-4f55-ad80-4dce8be795e0" power="10000000.0">
        <geometry xsi:type="esdl:Point" lon="4.375905990600587" CRS="WGS84" lat="52.08245983569832"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9b760b7c-1e69-4653-b322-1aa3d4040be2" id="eb6769ba-4a7d-4e1a-8f89-0e1f60d86f32" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a79a74b2-6ca2-4098-827b-7895d4ebb631" connectedTo="48427c64-4346-46db-989a-d75df7faff3e" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="10000000.0" name="ResidualHeatSource_61b8" id="61b8d7e1-aa99-439e-86b2-ea596e728932">
        <geometry xsi:type="esdl:Point" lon="4.400968551635743" CRS="WGS84" lat="52.09084554299605"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c932c77b-d5e5-40eb-b9c8-c67b8d0e25e9" connectedTo="309f0881-e58e-4a63-a3c2-c9ca971d2150" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="167022d9-0487-4acf-80ae-c7268c44724a" id="7c3e792b-9c73-49af-9845-1ee16dcb2a27" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="100000.0" name="ResidualHeatSource_aec9" id="aec9bf5f-ce40-4a6e-b2d6-d64e566ce147">
        <geometry xsi:type="esdl:Point" lon="4.400711059570313" CRS="WGS84" lat="52.08245983569832"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b466eb33-ed48-4685-bd88-63499e8e36db" connectedTo="5692ae05-f9c4-4f28-8a53-21bcafedf9a9" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5f1b694c-4562-469e-a8b8-531cc9ea143b" id="12e2b556-699f-497e-aa86-2284936c3658" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ffe" id="7ffe304e-0363-4634-aef5-a5da3343d7f5">
        <geometry xsi:type="esdl:Point" lon="4.388008117675782" CRS="WGS84" lat="52.082407090414286"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="28a0d33d-8b5e-431c-8e2a-c78a6fca3c31 1fb45b83-fc31-475f-9ced-0f9f17a2c454" id="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2950515a-1a1e-4e7b-a53b-65af13db7a45" connectedTo="dd5fcdd7-6779-4743-bee1-439e3df7d5ab" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8581" id="8581fd27-994d-4848-a458-1dd8c8f1f684">
        <geometry xsi:type="esdl:Point" lon="4.387664794921876" CRS="WGS84" lat="52.09084554299605"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="1eac327d-7ab8-48b8-8d33-6445416dd352" id="b0954a57-c90f-4601-837f-e593b3e90b51" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="76930ec6-55eb-4a0e-8880-3ea0ef74a327" connectedTo="c5ff5fe1-8e51-4292-b51a-599d10d1683d 7206613a-713e-42af-8c0e-ff16195ec3ad" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1_ret" diameter="DN300" length="908.9" name="Pipe1" innerDiameter="0.3127" id="Pipe1" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.400968551635743" lat="52.09084554299605"/>
          <point xsi:type="esdl:Point" lon="4.387664794921876" lat="52.09084554299605"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In" id="309f0881-e58e-4a63-a3c2-c9ca971d2150" connectedTo="c932c77b-d5e5-40eb-b9c8-c67b8d0e25e9" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="1eac327d-7ab8-48b8-8d33-6445416dd352" connectedTo="b0954a57-c90f-4601-837f-e593b3e90b51" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="febeba99-31d3-4dd3-bfad-0b95be773496">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2_ret" diameter="DN300" length="776.7" name="Pipe2" innerDiameter="0.3127" id="Pipe2" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.387664794921876" lat="52.09084554299605"/>
          <point xsi:type="esdl:Point" lon="4.37633514404297" lat="52.09026545046112"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In" id="c5ff5fe1-8e51-4292-b51a-599d10d1683d" connectedTo="76930ec6-55eb-4a0e-8880-3ea0ef74a327" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="43e98a06-8db5-43a1-913b-e8e7f255fc3f" connectedTo="06b6b748-d052-4fd5-a017-ff76321284a9" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="1def15d6-d08b-488a-9448-51abda40cba3">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3_ret" diameter="DN300" length="868.0" name="Pipe3" innerDiameter="0.3127" id="Pipe3" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.400711059570313" lat="52.08245983569832"/>
          <point xsi:type="esdl:Point" lon="4.388008117675782" lat="52.082407090414286"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In" id="5692ae05-f9c4-4f28-8a53-21bcafedf9a9" connectedTo="b466eb33-ed48-4685-bd88-63499e8e36db" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="28a0d33d-8b5e-431c-8e2a-c78a6fca3c31" connectedTo="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="56ae609b-7db1-4709-a514-6b3457f3509d">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4_ret" diameter="DN300" length="827.0" name="Pipe4" innerDiameter="0.3127" id="Pipe4" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.388008117675782" lat="52.082407090414286"/>
          <point xsi:type="esdl:Point" lon="4.375905990600587" lat="52.08245983569832"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In" id="dd5fcdd7-6779-4743-bee1-439e3df7d5ab" connectedTo="2950515a-1a1e-4e7b-a53b-65af13db7a45" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9b760b7c-1e69-4653-b322-1aa3d4040be2" connectedTo="eb6769ba-4a7d-4e1a-8f89-0e1f60d86f32" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="47ce9c1a-2eaa-4905-9fd5-00f1ce0c5413">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ffe_ret" id="0554c01f-71e6-4736-abc0-31cda8d834a7">
        <geometry xsi:type="esdl:Point" lon="4.387485812242726" CRS="WGS84" lat="52.08249709050428"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="dee6c805-0d6e-40cc-ac57-836481b371e7" id="e9e1badb-a5a2-4637-8ce2-94a7b1564a54" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="616137cf-6758-4091-abf1-220d21169b7a" connectedTo="bd02ded1-f0ea-4192-b304-7d8c128eee0b f7beb290-c113-4f55-ad8b-7a3a6d096b23" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8581_ret" id="cea5d37b-7edd-488f-a1d7-b37b9451459a">
        <geometry xsi:type="esdl:Point" lon="4.38715944579282" CRS="WGS84" lat="52.09093554308605"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="6f5b062b-a1e9-4d70-b743-1e43501a59fe" connectedTo="3a4a6955-9073-46ee-bbc7-52b894bc21d1" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="d4c09201-f88b-4c3a-99cf-594d5e188605 b7e1cb61-6809-4a3e-8053-3eac95fadcd4" id="17cc66e1-5c3f-48d6-8aa3-3a19fb9b0c8e" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1" diameter="DN300" name="Pipe1_ret" length="908.9" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe1_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.38715944579282" CRS="WGS84" lat="52.09093554308605"/>
          <point xsi:type="esdl:Point" lon="4.400463202506687" CRS="WGS84" lat="52.09093554308605"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="3a4a6955-9073-46ee-bbc7-52b894bc21d1" connectedTo="6f5b062b-a1e9-4d70-b743-1e43501a59fe" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="167022d9-0487-4acf-80ae-c7268c44724a" connectedTo="7c3e792b-9c73-49af-9845-1ee16dcb2a27" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2" diameter="DN300" name="Pipe2_ret" length="776.7" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe2_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.375828665733543" CRS="WGS84" lat="52.090355450551115"/>
          <point xsi:type="esdl:Point" lon="4.38715944579282" CRS="WGS84" lat="52.09093554308605"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="46e659aa-0e2b-43af-b38c-242cb7fdc480" connectedTo="01ea44bf-6f10-4913-bd41-5e5b210d3b40" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d4c09201-f88b-4c3a-99cf-594d5e188605" connectedTo="17cc66e1-5c3f-48d6-8aa3-3a19fb9b0c8e" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3" diameter="DN300" name="Pipe3_ret" length="868.0" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe3_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.387485812242726" CRS="WGS84" lat="52.08249709050428"/>
          <point xsi:type="esdl:Point" lon="4.400188863774152" CRS="WGS84" lat="52.08254983578832"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="bd02ded1-f0ea-4192-b304-7d8c128eee0b" connectedTo="616137cf-6758-4091-abf1-220d21169b7a" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="5f1b694c-4562-469e-a8b8-531cc9ea143b" connectedTo="12e2b556-699f-497e-aa86-2284936c3658" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4" diameter="DN300" name="Pipe4_ret" length="827.0" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe4_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.375383794804425" CRS="WGS84" lat="52.08254983578832"/>
          <point xsi:type="esdl:Point" lon="4.387485812242726" CRS="WGS84" lat="52.08249709050428"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="48427c64-4346-46db-989a-d75df7faff3e" connectedTo="a79a74b2-6ca2-4098-827b-7895d4ebb631" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="dee6c805-0d6e-40cc-ac57-836481b371e7" connectedTo="e9e1badb-a5a2-4637-8ce2-94a7b1564a54" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" diameter="DN300" name="Pipe_ce68" length="455.2" id="ce6861d7-f06f-4391-adcf-bb1284c24718" outerDiameter="0.45" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.09084554299605" lon="4.387664794921876"/>
          <point xsi:type="esdl:Point" lat="52.08675177427041" lon="4.387589693069459"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" id="7206613a-713e-42af-8c0e-ff16195ec3ad" connectedTo="76930ec6-55eb-4a0e-8880-3ea0ef74a327" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="22daf91b-d674-4fe9-b35b-360a731c173b" connectedTo="59f661b5-9a21-4054-8af9-3ee8858f9960" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="5888dfba-51d2-4562-b1f4-6b3965f20eba">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" diameter="DN300" name="Pipe_408e" length="404.9" id="408e117a-809b-458e-bd9f-1ead868fc84a" outerDiameter="0.45" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.08633644341566" lon="4.387664794921876"/>
          <point xsi:type="esdl:Point" lat="52.08269718870518" lon="4.387879371643067"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" id="5a64ebd4-90d9-471e-b413-5948c25cf9c8" connectedTo="80cfef3d-79d3-4d2c-b78d-701264cf3313" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="1fb45b83-fc31-475f-9ced-0f9f17a2c454" connectedTo="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3efcdead-3f3b-4972-b7e9-dd3dbebabb81">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" diameter="DN300" name="Pipe_408e_ret" length="413.5" id="9a7df578-3001-4e0b-a902-fca2665bef9c" outerDiameter="0.45" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.08268400246011" lon="4.3873000144958505"/>
          <point xsi:type="esdl:Point" lat="52.086402369206255" lon="4.387278556823731"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" id="f7beb290-c113-4f55-ad8b-7a3a6d096b23" connectedTo="616137cf-6758-4091-abf1-220d21169b7a" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="Out" id="96a5a980-06b9-4519-bc9b-c61e2f86537d" connectedTo="0500e037-457c-418b-9a55-d7ed0d34a98a" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="0970189c-8baa-4c65-83ab-d13e0b7c03c0">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" diameter="DN300" name="Pipe_ce68_ret" length="436.9" id="71ef008d-7412-4db6-9d42-adba1a7f59d1" outerDiameter="0.45" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0867319966983" lon="4.38732147216797"/>
          <point xsi:type="esdl:Point" lat="52.09066096891663" lon="4.387235641479493"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" thermalConductivity="52.15" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" thermalConductivity="0.027" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" thermalConductivity="0.4" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" id="ef6d471c-ead8-4541-853e-f85dea09e2be" connectedTo="76ab7560-39a3-4095-81a2-6afd87603fcd" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b7e1cb61-6809-4a3e-8053-3eac95fadcd4" connectedTo="17cc66e1-5c3f-48d6-8aa3-3a19fb9b0c8e" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="52bf58f4-7c46-48f7-9f28-4590b6719641">
          <investmentCosts xsi:type="esdl:SingleValue" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:GenericConversion" name="GenericConversion_3d3f" id="3d3f6e8e-a8cb-48c4-9c6a-fc0d7848e549" power="10000000.0" efficiency="0.9">
        <geometry xsi:type="esdl:Point" lat="52.08654740560273" lon="4.387439489364625" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="HeatInPrimary" id="59f661b5-9a21-4054-8af9-3ee8858f9960" connectedTo="22daf91b-d674-4fe9-b35b-360a731c173b" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:InPort" name="HeatInSecondary" id="0500e037-457c-418b-9a55-d7ed0d34a98a" connectedTo="96a5a980-06b9-4519-bc9b-c61e2f86537d" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="HeatOutPrimary" id="76ab7560-39a3-4095-81a2-6afd87603fcd" connectedTo="ef6d471c-ead8-4541-853e-f85dea09e2be" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="HeatOutSecondary" id="80cfef3d-79d3-4d2c-b78d-701264cf3313" connectedTo="5a64ebd4-90d9-471e-b413-5948c25cf9c8" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
