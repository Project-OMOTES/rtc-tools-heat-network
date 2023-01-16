<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" version="2" name="Untitled EnergySystem with return network" id="0735d19f-39a4-463d-bf28-cf0b28b88bef_with_return_network" esdlVersion="v2207" description="">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="b7ebaafe-597c-4055-bc88-b47cdfa84e34">
    <carriers xsi:type="esdl:Carriers" id="5d5909a4-18ec-4aa3-a08a-1e9539a20be2">
      <carrier xsi:type="esdl:HeatCommodity" name="heat1" id="d336e381-ca6f-442e-985e-9f4c2bec1efe" supplyTemperature="70.0"/>
      <carrier xsi:type="esdl:HeatCommodity" name="heat2" id="72126c73-87e9-4bf6-99cf-d02a6c07010c" supplyTemperature="90.0"/>
      <carrier xsi:type="esdl:HeatCommodity" name="heat1_ret" returnTemperature="40.0" id="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      <carrier xsi:type="esdl:HeatCommodity" name="heat2_ret" returnTemperature="50.0" id="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
    </carriers>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" id="82399ebf-5e52-465d-830a-b502d6c1012b" name="MultipleCarrierTest">
    <area xsi:type="esdl:Area" id="e91b7896-ae7a-4136-8e96-d98d40c49fc2" name="MultipleCarrierTest">
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_3322" id="3322fe41-f73b-4ba4-b87b-bf8315aa69e4" power="10000000.0">
        <geometry xsi:type="esdl:Point" lon="4.37633514404297" CRS="WGS84" lat="52.09026545046112"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="43e98a06-8db5-43a1-913b-e8e7f255fc3f" id="06b6b748-d052-4fd5-a017-ff76321284a9" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="01ea44bf-6f10-4913-bd41-5e5b210d3b40" connectedTo="5910dfdb-0ffa-4e3c-b4d4-f4ea9137202a" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_18aa" id="18aabae2-c92a-4f55-ad80-4dce8be795e0" power="10000000.0">
        <geometry xsi:type="esdl:Point" lon="4.375905990600587" CRS="WGS84" lat="52.08245983569832"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9b760b7c-1e69-4653-b322-1aa3d4040be2" id="eb6769ba-4a7d-4e1a-8f89-0e1f60d86f32" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a79a74b2-6ca2-4098-827b-7895d4ebb631" connectedTo="0e7ce77f-0159-4230-aacd-da324b75d351" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="10000000.0" name="ResidualHeatSource_61b8" id="61b8d7e1-aa99-439e-86b2-ea596e728932">
        <geometry xsi:type="esdl:Point" lon="4.400968551635743" CRS="WGS84" lat="52.09084554299605"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c932c77b-d5e5-40eb-b9c8-c67b8d0e25e9" connectedTo="309f0881-e58e-4a63-a3c2-c9ca971d2150" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="1b7c3efb-7890-4861-8d34-e2b9f93f986a" id="7c3e792b-9c73-49af-9845-1ee16dcb2a27" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="10000000.0" name="ResidualHeatSource_aec9" id="aec9bf5f-ce40-4a6e-b2d6-d64e566ce147">
        <geometry xsi:type="esdl:Point" lon="4.400711059570313" CRS="WGS84" lat="52.08245983569832"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b466eb33-ed48-4685-bd88-63499e8e36db" connectedTo="5692ae05-f9c4-4f28-8a53-21bcafedf9a9" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d654de1a-d05c-4ff6-8916-afb72393a3b2" id="12e2b556-699f-497e-aa86-2284936c3658" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ffe" id="7ffe304e-0363-4634-aef5-a5da3343d7f5">
        <geometry xsi:type="esdl:Point" lon="4.388008117675782" CRS="WGS84" lat="52.082407090414286"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="28a0d33d-8b5e-431c-8e2a-c78a6fca3c31" id="5d0f5b7b-53b9-4999-82d1-0eee38a37791" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2950515a-1a1e-4e7b-a53b-65af13db7a45" connectedTo="dd5fcdd7-6779-4743-bee1-439e3df7d5ab" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8581" id="8581fd27-994d-4848-a458-1dd8c8f1f684">
        <geometry xsi:type="esdl:Point" lon="4.387664794921876" CRS="WGS84" lat="52.09084554299605"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="1eac327d-7ab8-48b8-8d33-6445416dd352" id="b0954a57-c90f-4601-837f-e593b3e90b51" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
        <port xsi:type="esdl:OutPort" name="Out" id="76930ec6-55eb-4a0e-8880-3ea0ef74a327" connectedTo="c5ff5fe1-8e51-4292-b51a-599d10d1683d" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe"/>
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
      <asset xsi:type="esdl:Joint" name="Joint_8581_ret" id="77422cbf-a0e6-4a1a-8a41-6f6c5f94f6b7">
        <geometry xsi:type="esdl:Point" lon="4.38715944579282" CRS="WGS84" lat="52.09093554308605"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="3dea3b55-c03e-42a2-b1e6-5435bb7e8d6a" id="76330c4c-b7ec-4427-91b6-d7623282cdd2" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="215f384b-8284-4b41-a7d9-f2d5f83cfb30" connectedTo="5e8beb0c-7bc8-49eb-9b21-3d50cb1576fa" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7ffe_ret" id="449673b4-a8c1-449d-be02-dab669c0168c">
        <geometry xsi:type="esdl:Point" lon="4.387485812242726" CRS="WGS84" lat="52.08249709050428"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="0fe867bc-2b82-4698-80c9-cd1a62a976fd" id="b004c8ab-0190-487f-b53e-29b304a91f11" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="3445e5f6-a022-4ce1-9d8b-c4d479ff56ca" connectedTo="a7f30208-50fb-4676-a798-45b8e7696a54" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1" diameter="DN300" name="Pipe1_ret" length="908.9" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe1_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.38715944579282" CRS="WGS84" lat="52.09093554308605"/>
          <point xsi:type="esdl:Point" lon="4.400463202506687" CRS="WGS84" lat="52.09093554308605"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="5e8beb0c-7bc8-49eb-9b21-3d50cb1576fa" connectedTo="215f384b-8284-4b41-a7d9-f2d5f83cfb30" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="1b7c3efb-7890-4861-8d34-e2b9f93f986a" connectedTo="7c3e792b-9c73-49af-9845-1ee16dcb2a27" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2" diameter="DN300" name="Pipe2_ret" length="776.7" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe2_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.375828665733543" CRS="WGS84" lat="52.090355450551115"/>
          <point xsi:type="esdl:Point" lon="4.38715944579282" CRS="WGS84" lat="52.09093554308605"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="5910dfdb-0ffa-4e3c-b4d4-f4ea9137202a" connectedTo="01ea44bf-6f10-4913-bd41-5e5b210d3b40" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="3dea3b55-c03e-42a2-b1e6-5435bb7e8d6a" connectedTo="76330c4c-b7ec-4427-91b6-d7623282cdd2" carrier="d336e381-ca6f-442e-985e-9f4c2bec1efe_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3" diameter="DN300" name="Pipe3_ret" length="868.0" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe3_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.387485812242726" CRS="WGS84" lat="52.08249709050428"/>
          <point xsi:type="esdl:Point" lon="4.400188863774152" CRS="WGS84" lat="52.08254983578832"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="a7f30208-50fb-4676-a798-45b8e7696a54" connectedTo="3445e5f6-a022-4ce1-9d8b-c4d479ff56ca" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d654de1a-d05c-4ff6-8916-afb72393a3b2" connectedTo="12e2b556-699f-497e-aa86-2284936c3658" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4" diameter="DN300" name="Pipe4_ret" length="827.0" innerDiameter="0.3127" outerDiameter="0.45" id="Pipe4_ret">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.375383794804425" CRS="WGS84" lat="52.08254983578832"/>
          <point xsi:type="esdl:Point" lon="4.387485812242726" CRS="WGS84" lat="52.08249709050428"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" id="0e7ce77f-0159-4230-aacd-da324b75d351" connectedTo="a79a74b2-6ca2-4098-827b-7895d4ebb631" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="0fe867bc-2b82-4698-80c9-cd1a62a976fd" connectedTo="b004c8ab-0190-487f-b53e-29b304a91f11" carrier="72126c73-87e9-4bf6-99cf-d02a6c07010c_ret"/>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
