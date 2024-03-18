<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" esdlVersion="v2303" name="h2" version="8" id="c7ad2e58-111f-4e42-9e86-15516d1f5b29" description="">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="268b080c-6ee2-4498-9e7e-2c0105376c39">
    <carriers xsi:type="esdl:Carriers" id="3ef72f2f-98da-4818-a63b-ac13b04c897c">
      <carrier xsi:type="esdl:ElectricityCommodity" id="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="elec" voltage="50000.0"/>
      <carrier xsi:type="esdl:GasCommodity" id="68904785-3ba5-4894-8751-78d5883dc372" pressure="15.0" name="gas"/>
    </carriers>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" id="cea2b988-2dbc-4f9b-b816-685f307d174c" name="Untitled instance">
    <area xsi:type="esdl:Area" id="5f49c2c8-6193-45a2-ad7f-457b44a81d93" name="Untitled area">
      <asset xsi:type="esdl:Electrolyzer" name="Electrolyzer_fc66" effMaxLoad="68.0" maxLoad="500000000" efficiency="63.0" id="fc6644db-15b9-4a3f-9637-044644b496e9" power="500000000.0" effMinLoad="67.0" minLoad="50000000">
        <costInformation xsi:type="esdl:CostInformation" id="7428c1cc-3b5f-47c9-aa8e-ab8e7ded85a2">
          <fixedOperationalCosts xsi:type="esdl:SingleValue" id="336d392b-dae9-4073-859d-bbb818de1537" value="0.001">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" unit="EURO" id="65892160-442b-4615-a675-ec1b53f984e1" description="Cost in EUR/W/yr" physicalQuantity="COST" perTimeUnit="YEAR" perUnit="WATT"/>
          </fixedOperationalCosts>
          <investmentCosts xsi:type="esdl:SingleValue" id="5ee09361-d48b-4775-b5b1-182be6373d97" value="20.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" unit="EURO" id="abbe4b85-715a-4715-a81c-4b8dc53dbeac" description="Cost in EUR/kW" perMultiplier="KILO" physicalQuantity="COST" perUnit="WATT"/>
          </investmentCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="b3233805-b82b-406c-98fb-fc7bac66bb78" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="In" connectedTo="ce1699b8-34d4-43b9-a77a-1bfb08a9c36c"/>
        <port xsi:type="esdl:OutPort" id="5079eeaa-5083-49ce-94d3-07e55348e747" connectedTo="175baa37-443c-4b4f-8e07-ce9707f27bac" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.91407805051185" lon="4.718627929687501"/>
      </asset>
      <asset xsi:type="esdl:ElectricityDemand" name="ElectricityDemand_9d15" id="9d15a76b-cfc3-4abf-a166-6edc19b1264e" power="1000000000.0">
        <port xsi:type="esdl:InPort" id="8ed762e6-b62d-4dd0-b84d-d3759af40b3d" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="In" connectedTo="487db503-69b4-4604-9896-563634352699"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.92463517826793" lon="4.727210998535157"/>
      </asset>
      <asset xsi:type="esdl:GasDemand" name="GasDemand_0cf3" id="0cf3a097-01ea-4fea-a6af-f141a6885445" power="1000000000.0">
        <costInformation xsi:type="esdl:CostInformation" id="14964bd8-8eb0-4525-bcb4-0d03c1252a2b">
          <variableOperationalCosts xsi:type="esdl:SingleValue" id="5c63dce1-8a70-44f3-b6b9-476587ff73bb" value="0.0001">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" unit="EURO" id="4a4516c3-372a-4a6c-b25a-dcc8dca30545" description="Cost in EUR/kg" physicalQuantity="COST" multiplier="KILO" perUnit="GRAM"/>
          </variableOperationalCosts>
        </costInformation>
        <port xsi:type="esdl:InPort" id="119be8fa-50a9-463b-ba1e-6b97cea86f0e" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="In" connectedTo="a75218ac-9a5b-4ee4-b0ad-ae1262329442"/>
        <geometry xsi:type="esdl:Point" lat="52.91987443929076" lon="4.728240966796876"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_c156" id="c15601d5-a9ce-41b4-84eb-1a7658d65c72">
        <port xsi:type="esdl:InPort" id="f1e5ab4c-7d5f-4a8d-9bd9-e5131b34dd7d" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="In" connectedTo="8e1dbe29-588b-41e0-8fd5-c76bff01b1b0"/>
        <port xsi:type="esdl:OutPort" id="71538b02-85df-4e97-b647-64c058a57cc4" connectedTo="85d05e29-2096-489c-ae87-48f1a558c627" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.91490615355209" lon="4.724249839782716"/>
      </asset>
      <asset xsi:type="esdl:Bus" name="Bus_0694" id="0694b3aa-d613-40c4-96b2-8d447ab1fe96">
        <port xsi:type="esdl:InPort" id="d3860fc6-eb6c-4508-b77a-0c5327d9bb62" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="In" connectedTo="9050262b-fbbe-416f-9082-4988bd5787cf"/>
        <port xsi:type="esdl:OutPort" id="aa560077-d0ec-4391-9a48-97650436a326" connectedTo="50d85125-418b-48a8-a7e9-86f9de3106ae 0ad5e2e3-ab3d-4afb-8a34-8fc7ddc5f93b" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="Out"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.92429883889503" lon="4.718542098999024"/>
      </asset>
      <asset xsi:type="esdl:WindPark" name="WindPark_7f14" surfaceArea="69301306" id="7f140121-76d6-4abe-aa6f-4c611664280a" power="1000000000.0">
        <port xsi:type="esdl:OutPort" id="40593cd4-e512-4ec4-b702-47f1cc7d1836" connectedTo="2e47d054-4ea8-4ca0-a5a2-0cb60d8dd338" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="Out"/>
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="52.912628832091194" lon="4.432296752929688"/>
            <point xsi:type="esdl:Point" lat="52.91345696284071" lon="4.528427124023438"/>
            <point xsi:type="esdl:Point" lat="52.99577682618084" lon="4.547653198242188"/>
            <point xsi:type="esdl:Point" lat="53.008586419806925" lon="4.437103271484376"/>
          </exterior>
        </geometry>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" name="ElectricityCable_9c26" capacity="1000000000.0" length="16094.5" id="9c2688f7-e530-4ea4-b650-5c3a6a5998a5">
        <port xsi:type="esdl:InPort" id="2e47d054-4ea8-4ca0-a5a2-0cb60d8dd338" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="In" connectedTo="40593cd4-e512-4ec4-b702-47f1cc7d1836"/>
        <port xsi:type="esdl:OutPort" id="9050262b-fbbe-416f-9082-4988bd5787cf" connectedTo="d3860fc6-eb6c-4508-b77a-0c5327d9bb62" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.958842985141686" lon="4.485299213047531"/>
          <point xsi:type="esdl:Point" lat="52.92429883889503" lon="4.718542098999024"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" name="ElectricityCable_09d1" capacity="1000000000.0" length="582.3" id="09d101cd-1ead-4c9e-b732-992299af350c">
        <port xsi:type="esdl:InPort" id="50d85125-418b-48a8-a7e9-86f9de3106ae" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="In" connectedTo="aa560077-d0ec-4391-9a48-97650436a326"/>
        <port xsi:type="esdl:OutPort" id="487db503-69b4-4604-9896-563634352699" connectedTo="8ed762e6-b62d-4dd0-b84d-d3759af40b3d" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.92429883889503" lon="4.718542098999024"/>
          <point xsi:type="esdl:Point" lat="52.92463517826793" lon="4.727210998535157"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" name="ElectricityCable_591d" capacity="1000000000.0" length="1136.5" id="591d5c32-d7ff-401c-a4f1-a618d77870e2">
        <port xsi:type="esdl:InPort" id="0ad5e2e3-ab3d-4afb-8a34-8fc7ddc5f93b" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="In" connectedTo="aa560077-d0ec-4391-9a48-97650436a326"/>
        <port xsi:type="esdl:OutPort" id="ce1699b8-34d4-43b9-a77a-1bfb08a9c36c" connectedTo="b3233805-b82b-406c-98fb-fc7bac66bb78" carrier="d7bfb1ae-b0ea-4d66-98a2-b7cf2b0f094d" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.92429883889503" lon="4.718542098999024"/>
          <point xsi:type="esdl:Point" lat="52.91407805051185" lon="4.718627929687501"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe_772f" id="772fe539-7453-436b-b48e-2a39abc5ffb4" diameter="DN1200" length="388.0">
        <port xsi:type="esdl:InPort" id="175baa37-443c-4b4f-8e07-ce9707f27bac" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="In" connectedTo="5079eeaa-5083-49ce-94d3-07e55348e747"/>
        <port xsi:type="esdl:OutPort" id="8e1dbe29-588b-41e0-8fd5-c76bff01b1b0" connectedTo="f1e5ab4c-7d5f-4a8d-9bd9-e5131b34dd7d" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.91407805051185" lon="4.718627929687501"/>
          <point xsi:type="esdl:Point" lat="52.91490615355209" lon="4.724249839782716"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" name="Pipe_6ba6" id="6ba60fbf-6647-494c-b7f3-146bcb263dc0" diameter="DN1200" length="790.5">
        <port xsi:type="esdl:InPort" id="85d05e29-2096-489c-ae87-48f1a558c627" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="In" connectedTo="71538b02-85df-4e97-b647-64c058a57cc4"/>
        <port xsi:type="esdl:OutPort" id="a75218ac-9a5b-4ee4-b0ad-ae1262329442" connectedTo="119be8fa-50a9-463b-ba1e-6b97cea86f0e" carrier="68904785-3ba5-4894-8751-78d5883dc372" name="Out"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.91490615355209" lon="4.724249839782716"/>
          <point xsi:type="esdl:Point" lat="52.91940868671637" lon="4.723992347717286"/>
          <point xsi:type="esdl:Point" lat="52.91987443929076" lon="4.728240966796876"/>
        </geometry>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
