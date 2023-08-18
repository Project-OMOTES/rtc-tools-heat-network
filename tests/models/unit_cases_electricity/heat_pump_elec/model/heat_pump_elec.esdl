<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" name="Untitled EnergySystem with return network" id="0735d19f-39a4-463d-bf28-cf0b28b88bef_with_return_network" description="" esdlVersion="v2207" version="7">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="b7ebaafe-597c-4055-bc88-b47cdfa84e34">
    <carriers xsi:type="esdl:Carriers" id="5d5909a4-18ec-4aa3-a08a-1e9539a20be2">
      <carrier xsi:type="esdl:HeatCommodity" id="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="heat1" supplyTemperature="70.0"/>
      <carrier xsi:type="esdl:HeatCommodity" id="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="heat2" supplyTemperature="90.0"/>
      <carrier xsi:type="esdl:HeatCommodity" returnTemperature="40.0" id="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="heat1_ret"/>
      <carrier xsi:type="esdl:HeatCommodity" returnTemperature="50.0" id="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="heat2_ret"/>
      <carrier xsi:type="esdl:ElectricityCommodity" voltage="230.0" id="3f149631-f5b5-48b4-934c-bb0faf3711b1" name="Electr"/>
    </carriers>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" name="MultipleCarrierTest" id="82399ebf-5e52-465d-830a-b502d6c1012b">
    <area xsi:type="esdl:Area" id="e91b7896-ae7a-4136-8e96-d98d40c49fc2" name="MultipleCarrierTest">
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_3322" id="3322fe41-f73b-4ba4-b87b-bf8315aa69e4" power="10000000.0">
        <geometry xsi:type="esdl:Point" lon="4.37633514404297" lat="52.09026545046112" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" connectedTo="43e98a06-8db5-43a1-913b-e8e7f255fc3f" id="06b6b748-d052-4fd5-a017-ff76321284a9" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="In"/>
        <port xsi:type="esdl:OutPort" id="01ea44bf-6f10-4913-bd41-5e5b210d3b40" connectedTo="46e659aa-0e2b-43af-b38c-242cb7fdc480" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="Out"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_18aa" id="18aabae2-c92a-4f55-ad80-4dce8be795e0" power="10000000.0">
        <geometry xsi:type="esdl:Point" lon="4.375905990600587" lat="52.08245983569832" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" connectedTo="9b760b7c-1e69-4653-b322-1aa3d4040be2" id="eb6769ba-4a7d-4e1a-8f89-0e1f60d86f32" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="In"/>
        <port xsi:type="esdl:OutPort" id="a79a74b2-6ca2-4098-827b-7895d4ebb631" connectedTo="48427c64-4346-46db-989a-d75df7faff3e" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="Out"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="10000000.0" name="ResidualHeatSource_61b8" id="61b8d7e1-aa99-439e-86b2-ea596e728932">
        <geometry xsi:type="esdl:Point" lon="4.400968551635743" lat="52.09084554299605" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" id="c932c77b-d5e5-40eb-b9c8-c67b8d0e25e9" connectedTo="309f0881-e58e-4a63-a3c2-c9ca971d2150" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="Out"/>
        <port xsi:type="esdl:InPort" connectedTo="167022d9-0487-4acf-80ae-c7268c44724a" id="7c3e792b-9c73-49af-9845-1ee16dcb2a27" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="In"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="10000000.0" name="ResidualHeatSource_aec9"
      id="aec9bf5f-ce40-4a6e-b2d6-d64e566ce147">
        <geometry xsi:type="esdl:Point" lon="4.400711059570313" lat="52.08245983569832" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" id="b466eb33-ed48-4685-bd88-63499e8e36db" connectedTo="5692ae05-f9c4-4f28-8a53-21bcafedf9a9" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="Out"/>
        <port xsi:type="esdl:InPort" connectedTo="5f1b694c-4562-469e-a8b8-531cc9ea143b" id="12e2b556-699f-497e-aa86-2284936c3658" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="In"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ffe" id="7ffe304e-0363-4634-aef5-a5da3343d7f5">
        <geometry xsi:type="esdl:Point" lon="4.388008117675782" lat="52.082407090414286" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" connectedTo="28a0d33d-8b5e-431c-8e2a-c78a6fca3c31 1fb45b83-fc31-475f-9ced-0f9f17a2c454" id="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="In"/>
        <port xsi:type="esdl:OutPort" id="2950515a-1a1e-4e7b-a53b-65af13db7a45" connectedTo="dd5fcdd7-6779-4743-bee1-439e3df7d5ab" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="Out"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8581" id="8581fd27-994d-4848-a458-1dd8c8f1f684">
        <geometry xsi:type="esdl:Point" lon="4.387664794921876" lat="52.09084554299605" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" connectedTo="1eac327d-7ab8-48b8-8d33-6445416dd352" id="b0954a57-c90f-4601-837f-e593b3e90b51" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="In"/>
        <port xsi:type="esdl:OutPort" id="76930ec6-55eb-4a0e-8880-3ea0ef74a327" connectedTo="c5ff5fe1-8e51-4292-b51a-599d10d1683d 7206613a-713e-42af-8c0e-ff16195ec3ad" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="Out"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1_ret" diameter="DN300" name="Pipe1" length="908.9" id="Pipe1" innerDiameter="0.3127" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.400968551635743" lat="52.09084554299605"/>
          <point xsi:type="esdl:Point" lon="4.387664794921876" lat="52.09084554299605"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="c932c77b-d5e5-40eb-b9c8-c67b8d0e25e9" id="309f0881-e58e-4a63-a3c2-c9ca971d2150" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="In"/>
        <port xsi:type="esdl:OutPort" id="1eac327d-7ab8-48b8-8d33-6445416dd352" connectedTo="b0954a57-c90f-4601-837f-e593b3e90b51" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="Out"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="febeba99-31d3-4dd3-bfad-0b95be773496">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2_ret" diameter="DN300" name="Pipe2" length="776.7" id="Pipe2" innerDiameter="0.3127" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.387664794921876" lat="52.09084554299605"/>
          <point xsi:type="esdl:Point" lon="4.37633514404297" lat="52.09026545046112"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="76930ec6-55eb-4a0e-8880-3ea0ef74a327" id="c5ff5fe1-8e51-4292-b51a-599d10d1683d" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="In"/>
        <port xsi:type="esdl:OutPort" id="43e98a06-8db5-43a1-913b-e8e7f255fc3f" connectedTo="06b6b748-d052-4fd5-a017-ff76321284a9" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="Out"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="1def15d6-d08b-488a-9448-51abda40cba3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3_ret" diameter="DN300" name="Pipe3" length="868.0" id="Pipe3" innerDiameter="0.3127" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.400711059570313" lat="52.08245983569832"/>
          <point xsi:type="esdl:Point" lon="4.388008117675782" lat="52.082407090414286"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="b466eb33-ed48-4685-bd88-63499e8e36db" id="5692ae05-f9c4-4f28-8a53-21bcafedf9a9" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="In"/>
        <port xsi:type="esdl:OutPort" id="28a0d33d-8b5e-431c-8e2a-c78a6fca3c31" connectedTo="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="Out"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="56ae609b-7db1-4709-a514-6b3457f3509d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4_ret" diameter="DN300" name="Pipe4" length="827.0" id="Pipe4" innerDiameter="0.3127" outerDiameter="0.45">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.388008117675782" lat="52.082407090414286"/>
          <point xsi:type="esdl:Point" lon="4.375905990600587" lat="52.08245983569832"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="2950515a-1a1e-4e7b-a53b-65af13db7a45" id="dd5fcdd7-6779-4743-bee1-439e3df7d5ab" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="In"/>
        <port xsi:type="esdl:OutPort" id="9b760b7c-1e69-4653-b322-1aa3d4040be2" connectedTo="eb6769ba-4a7d-4e1a-8f89-0e1f60d86f32" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="Out"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="47ce9c1a-2eaa-4905-9fd5-00f1ce0c5413">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ffe_ret" id="0554c01f-71e6-4736-abc0-31cda8d834a7">
        <geometry xsi:type="esdl:Point" lon="4.387485812242726" lat="52.08249709050428" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" connectedTo="dee6c805-0d6e-40cc-ac57-836481b371e7" id="e9e1badb-a5a2-4637-8ce2-94a7b1564a54" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="ret_port"/>
        <port xsi:type="esdl:OutPort" id="616137cf-6758-4091-abf1-220d21169b7a" connectedTo="bd02ded1-f0ea-4192-b304-7d8c128eee0b f7beb290-c113-4f55-ad8b-7a3a6d096b23" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="ret_port"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8581_ret" id="cea5d37b-7edd-488f-a1d7-b37b9451459a">
        <geometry xsi:type="esdl:Point" lon="4.38715944579282" lat="52.09093554308605" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" id="6f5b062b-a1e9-4d70-b743-1e43501a59fe" connectedTo="3a4a6955-9073-46ee-bbc7-52b894bc21d1" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="ret_port"/>
        <port xsi:type="esdl:InPort" connectedTo="d4c09201-f88b-4c3a-99cf-594d5e188605 b7e1cb61-6809-4a3e-8053-3eac95fadcd4" id="17cc66e1-5c3f-48d6-8aa3-3a19fb9b0c8e" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="ret_port"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1" diameter="DN300" name="Pipe1_ret" length="908.9" outerDiameter="0.45" innerDiameter="0.3127" id="Pipe1_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.38715944579282" lat="52.09093554308605" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lon="4.400463202506687" lat="52.09093554308605" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="6f5b062b-a1e9-4d70-b743-1e43501a59fe" id="3a4a6955-9073-46ee-bbc7-52b894bc21d1" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="In_ret"/>
        <port xsi:type="esdl:OutPort" id="167022d9-0487-4acf-80ae-c7268c44724a" connectedTo="7c3e792b-9c73-49af-9845-1ee16dcb2a27" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="Out_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2" diameter="DN300" name="Pipe2_ret" length="776.7" outerDiameter="0.45" innerDiameter="0.3127" id="Pipe2_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.375828665733543" lat="52.090355450551115" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lon="4.38715944579282" lat="52.09093554308605" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="01ea44bf-6f10-4913-bd41-5e5b210d3b40" id="46e659aa-0e2b-43af-b38c-242cb7fdc480" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="In_ret"/>
        <port xsi:type="esdl:OutPort" id="d4c09201-f88b-4c3a-99cf-594d5e188605" connectedTo="17cc66e1-5c3f-48d6-8aa3-3a19fb9b0c8e" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="Out_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3" diameter="DN300" name="Pipe3_ret" length="868.0" outerDiameter="0.45" innerDiameter="0.3127" id="Pipe3_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.387485812242726" lat="52.08249709050428" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lon="4.400188863774152" lat="52.08254983578832" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="616137cf-6758-4091-abf1-220d21169b7a" id="bd02ded1-f0ea-4192-b304-7d8c128eee0b" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="In_ret"/>
        <port xsi:type="esdl:OutPort" id="5f1b694c-4562-469e-a8b8-531cc9ea143b" connectedTo="12e2b556-699f-497e-aa86-2284936c3658" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="Out_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4" diameter="DN300" name="Pipe4_ret" length="827.0" outerDiameter="0.45" innerDiameter="0.3127" id="Pipe4_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.375383794804425" lat="52.08254983578832" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lon="4.387485812242726" lat="52.08249709050428" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="a79a74b2-6ca2-4098-827b-7895d4ebb631" id="48427c64-4346-46db-989a-d75df7faff3e" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="In_ret"/>
        <port xsi:type="esdl:OutPort" id="dee6c805-0d6e-40cc-ac57-836481b371e7" connectedTo="e9e1badb-a5a2-4637-8ce2-94a7b1564a54" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="Out_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" diameter="DN300" name="Pipe_ce68" length="455.2" id="ce6861d7-f06f-4391-adcf-bb1284c24718" outerDiameter="0.45" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.09084554299605" lon="4.387664794921876"/>
          <point xsi:type="esdl:Point" lat="52.08675177427041" lon="4.387589693069459"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" connectedTo="76930ec6-55eb-4a0e-8880-3ea0ef74a327" id="7206613a-713e-42af-8c0e-ff16195ec3ad" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="In"/>
        <port xsi:type="esdl:OutPort" id="22daf91b-d674-4fe9-b35b-360a731c173b" connectedTo="59f661b5-9a21-4054-8af9-3ee8858f9960" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="Out"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="5888dfba-51d2-4562-b1f4-6b3965f20eba">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
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
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" connectedTo="80cfef3d-79d3-4d2c-b78d-701264cf3313" id="5a64ebd4-90d9-471e-b413-5948c25cf9c8" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="In"/>
        <port xsi:type="esdl:OutPort" id="1fb45b83-fc31-475f-9ced-0f9f17a2c454" connectedTo="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="Out"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3efcdead-3f3b-4972-b7e9-dd3dbebabb81">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
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
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" connectedTo="616137cf-6758-4091-abf1-220d21169b7a" id="f7beb290-c113-4f55-ad8b-7a3a6d096b23" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="In"/>
        <port xsi:type="esdl:OutPort" id="96a5a980-06b9-4519-bc9b-c61e2f86537d" connectedTo="0500e037-457c-418b-9a55-d7ed0d34a98a" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="Out"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="0970189c-8baa-4c65-83ab-d13e0b7c03c0">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
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
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" connectedTo="76ab7560-39a3-4095-81a2-6afd87603fcd" id="ef6d471c-ead8-4541-853e-f85dea09e2be" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="In"/>
        <port xsi:type="esdl:OutPort" id="b7e1cb61-6809-4a3e-8053-3eac95fadcd4" connectedTo="17cc66e1-5c3f-48d6-8aa3-3a19fb9b0c8e" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="Out"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="52bf58f4-7c46-48f7-9f28-4590b6719641">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" perUnit="METRE" physicalQuantity="COST" unit="EURO"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatPump" name="GenericConversion_3d3f" id="3d3f6e8e-a8cb-48c4-9c6a-fc0d7848e549" power="10000000.0" COP="4.0">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.08654740560273" lon="4.387439489364625"/>
        <port xsi:type="esdl:InPort" connectedTo="22daf91b-d674-4fe9-b35b-360a731c173b" id="59f661b5-9a21-4054-8af9-3ee8858f9960" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe" name="HeatInPrimary"/>
        <port xsi:type="esdl:InPort" connectedTo="96a5a980-06b9-4519-bc9b-c61e2f86537d" id="0500e037-457c-418b-9a55-d7ed0d34a98a" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret" name="HeatInSecondary"/>
        <port xsi:type="esdl:OutPort" id="76ab7560-39a3-4095-81a2-6afd87603fcd" connectedTo="ef6d471c-ead8-4541-853e-f85dea09e2be" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret" name="HeatOutPrimary"/>
        <port xsi:type="esdl:OutPort" id="80cfef3d-79d3-4d2c-b78d-701264cf3313" connectedTo="5a64ebd4-90d9-471e-b413-5948c25cf9c8" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c" name="HeatOutSecondary"/>
        <port xsi:type="esdl:InPort" connectedTo="b876ee97-9035-418c-bcaa-23d478872951" id="abb58ad2-4393-4309-bc34-162df5045186" carrier="3f149631-f5b5-48b4-934c-bb0faf3711b1" name="Elec_InPort"/>
      </asset>
      <asset xsi:type="esdl:ElectricityProducer" power="1000000.0" name="ElectricityProducer_ac2e"
      id="ac2e9d27-fd81-4661-8487-642ef930d33c">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.08654042953765" lon="4.385293722152711"/>
        <port xsi:type="esdl:OutPort" id="7f499c86-9c84-409b-8cff-68180b0e55b5" connectedTo="a7c8326f-36a3-4040-a37c-406685e438be" carrier="3f149631-f5b5-48b4-934c-bb0faf3711b1" name="Out"/>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" name="ElectricityCable_9d3b" id="9d3befbb-16a9-416c-8ffc-215b4aab7f40" length="146.6">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.08654042953765" lon="4.385293722152711"/>
          <point xsi:type="esdl:Point" lat="52.08654740560273" lon="4.387439489364625"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="7f499c86-9c84-409b-8cff-68180b0e55b5" id="a7c8326f-36a3-4040-a37c-406685e438be" carrier="3f149631-f5b5-48b4-934c-bb0faf3711b1" name="In"/>
        <port xsi:type="esdl:OutPort" id="b876ee97-9035-418c-bcaa-23d478872951" connectedTo="abb58ad2-4393-4309-bc34-162df5045186" carrier="3f149631-f5b5-48b4-934c-bb0faf3711b1" name="Out"/>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
