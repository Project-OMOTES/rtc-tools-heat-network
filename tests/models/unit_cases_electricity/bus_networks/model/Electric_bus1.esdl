<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" id="81af5735-c13a-4fa4-aef1-e35676858b7a" name="Electric_bus1" description="Test network for the bus" esdlVersion="v2109" version="2">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="2dda7438-1efd-4c4d-8115-495032858aa9">
    <carriers xsi:type="esdl:Carriers" id="55ef6a87-d7c5-4871-83b3-ee33f5c0e0d6">
      <carrier xsi:type="esdl:ElectricityCommodity" id="48c94e57-26f7-4926-bf77-95437cebd000" name="electricity" voltage="230.0"/>
    </carriers>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" id="cc5c7fa0-e917-4a2c-b272-074d7969e6d1" name="Untitled instance">
    <area xsi:type="esdl:Area" id="b6f69416-2004-45f3-af7b-b3884edfe599" name="Untitled area">
      <asset xsi:type="esdl:ElectricityProducer" id="ElectricityProducer_17a1" power="1000.0" name="ElectricityProducer_17a1">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lon="5.119628906250001" lat="52.08610776589893"/>
        <port xsi:type="esdl:OutPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" connectedTo="a9f61f0f-960b-432d-a880-a56acd2f013e" name="Out" id="3c904675-33f6-4d01-8ab0-427f25cb8eb8"/>
      </asset>
      <asset xsi:type="esdl:ElectricityDemand" id="ElectricityDemand_e527" power="1000.0" name="ElectricityDemand_e527">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lon="5.1258301734924325" lat="52.084367319956606"/>
        <port xsi:type="esdl:InPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" name="In" id="d9ad4682-4324-484c-bd3a-ee14f51cdc3f" connectedTo="75f03edd-149a-4bdb-adda-39e1f6e5ca7a"/>
      </asset>
      <asset xsi:type="esdl:Bus" id="f262233c-e1a9-46ca-8c54-812cb64263b8" name="Bus_f262">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lon="5.122611522674561" lat="52.085052958279135"/>
        <port xsi:type="esdl:InPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" name="In" id="2404d6a4-c7a1-433a-b0f9-ba3845ae706b" connectedTo="589a97da-a84f-4614-819d-ab123d66f5f1"/>
        <port xsi:type="esdl:OutPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" connectedTo="a2d49319-d387-49e3-aa58-2d159314e7d8" name="Out" id="9f455f7a-d82d-4c50-9120-9442ef4af39b"/>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" id="1ad063bc-0e08-4570-8c1e-839bc6f591e3" length="235.1" name="ElectricityCable_1ad0">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="5.119628906250001" lat="52.08610776589893"/>
          <point xsi:type="esdl:Point" lon="5.122611522674561" lat="52.085052958279135"/>
        </geometry>
        <port xsi:type="esdl:InPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" name="In" id="a9f61f0f-960b-432d-a880-a56acd2f013e" connectedTo="3c904675-33f6-4d01-8ab0-427f25cb8eb8"/>
        <port xsi:type="esdl:OutPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" connectedTo="2404d6a4-c7a1-433a-b0f9-ba3845ae706b" name="Out" id="589a97da-a84f-4614-819d-ab123d66f5f1"/>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" id="de9a9562-dbca-462c-95a1-4e03cb6fabc2" length="232.8" name="ElectricityCable_de9a">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="5.122611522674561" lat="52.085052958279135"/>
          <point xsi:type="esdl:Point" lon="5.1258301734924325" lat="52.084367319956606"/>
        </geometry>
        <port xsi:type="esdl:InPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" name="In" id="a2d49319-d387-49e3-aa58-2d159314e7d8" connectedTo="9f455f7a-d82d-4c50-9120-9442ef4af39b"/>
        <port xsi:type="esdl:OutPort" carrier="48c94e57-26f7-4926-bf77-95437cebd000" connectedTo="d9ad4682-4324-484c-bd3a-ee14f51cdc3f" name="Out" id="75f03edd-149a-4bdb-adda-39e1f6e5ca7a"/>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
