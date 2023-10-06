<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" name="Demo_areas" description="" version="9" esdlVersion="v2303" id="483e3ff5-7ff2-4b50-b7a3-c66cffe49703">
  <instance xsi:type="esdl:Instance" name="Untitled instance" id="411b572c-f96b-49fa-8a0a-389705cbf58d">
    <area xsi:type="esdl:Area" name="Delft" id="1e502f41-ce24-4609-b518-cd76a8aafb59">
      <area xsi:type="esdl:Area" id="72effb55-fb15-45a4-9e91-ed56104d777f" name="TUwijk">
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="52.002003665853785" lon="4.366979598999024"/>
            <point xsi:type="esdl:Point" lat="52.0030604590385" lon="4.371356964111329"/>
            <point xsi:type="esdl:Point" lat="52.000127796500934" lon="4.373674392700196"/>
            <point xsi:type="esdl:Point" lat="51.99875387048352" lon="4.374532699584962"/>
            <point xsi:type="esdl:Point" lat="51.999837161821546" lon="4.378180503845216"/>
            <point xsi:type="esdl:Point" lat="51.997353479419196" lon="4.379682540893556"/>
            <point xsi:type="esdl:Point" lat="51.99674574891218" lon="4.375905990600587"/>
            <point xsi:type="esdl:Point" lat="51.99809331411895" lon="4.37483310699463"/>
            <point xsi:type="esdl:Point" lat="51.996957134372074" lon="4.370155334472657"/>
          </exterior>
        </geometry>
        <asset xsi:type="esdl:HeatingDemand" power="25000000.0" name="HeatingDemand_c5c8" id="c5c8678d-e624-4878-95cc-baa9e8809d5e">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.000365587107815" lon="4.371979236602784"/>
          <port xsi:type="esdl:InPort" connectedTo="3574833a-a997-42aa-b93f-2feb32350f9c" name="In" id="751b6c76-92d3-4d7c-83a9-ca90ddc73b0e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1">
            <profile xsi:type="esdl:InfluxDBProfile" multiplier="8.0" filters="" startDate="2018-12-31T23:00:00.000000+0000" field="demand2_MW" database="energy_profiles" host="profiles.warmingup.info" port="443" endDate="2019-12-31T22:00:00.000000+0000" id="f1e57252-159c-4a34-bf52-a1d1a4a3eade" measurement="WarmingUp default profiles">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" id="c81b81a7-2ed3-4c20-9c4b-ecc05eff5627" name="Out"/>
          <costInformation xsi:type="esdl:CostInformation" id="1f927790-982a-4ef5-b99a-4e9e9aceabd2">
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="ae0d67c2-2e06-48fd-a150-2fabe403a076">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="51492d75-208e-4a02-a6c9-a1c2e5be6a5f"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:HeatStorage" maxDischargeRate="1000000.0" capacity="3600000000.0" maxChargeRate="1000000.0" name="HeatStorage_171d" id="171dd49c-b2f2-47d0-99d8-a845ad9ee33b">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99872744841605" lon="4.376206398010255"/>
          <port xsi:type="esdl:InPort" connectedTo="c9bd9209-6b43-46b0-b315-6f465c950a94" name="In" id="61f03284-a3b3-4748-b34d-31bc2bdcdd3d" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" id="96cdb905-3636-4bea-b989-d51afcac0d97" name="Out"/>
          <costInformation xsi:type="esdl:CostInformation" id="200a0bf3-3e61-452f-a01c-e6f2e13b04c8">
            <investmentCosts xsi:type="esdl:SingleValue" value="500.0" id="07eb7de0-5274-4697-948f-9e434657ea4c">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="CUBIC_METRE" description="Cost in EUR/m3" unit="EURO" physicalQuantity="COST" id="a5e1e949-0e02-4c21-a02c-fe0dd79ed1a3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="344.13" innerDiameter="0.3127" outerDiameter="0.45" id="065afef2-68e8-408f-80b8-bef8b8f10806" name="Pipe_065a" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
            <point xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
          <port xsi:type="esdl:InPort" connectedTo="e585e343-cb4d-45c4-bc16-0a7e862f8a75" name="In" id="86838250-3430-4918-9806-e3b5dc400055" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="459b9e2d-3588-4652-9ad3-b4c7845728d0" name="Out" id="920fda15-8b06-40ab-9c25-70c4ec4af6c9" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
              <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
              <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
              <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="030b9982-3c2c-4e5e-9ad0-79d3b61612d6">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" physicalQuantity="COST" unit="EURO" description="Costs in EUR/m" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="80.38" innerDiameter="0.3127" outerDiameter="0.45" id="4a1a2610-652c-4a56-bb46-339f555e6514" name="Pipe_4a1a" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
            <point xsi:type="esdl:Point" lat="51.99969184377424" lon="4.372311830520631"/>
            <point xsi:type="esdl:Point" lat="52.000365587107815" lon="4.371979236602784"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
          <port xsi:type="esdl:InPort" connectedTo="acb56bd8-eb5f-456e-923a-c5656848a3a8" name="In" id="1ba3bfca-5ef8-41ff-ab57-b3d5b18a4560" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="751b6c76-92d3-4d7c-83a9-ca90ddc73b0e" name="Out" id="3574833a-a997-42aa-b93f-2feb32350f9c" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
              <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
              <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
              <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="030b9982-3c2c-4e5e-9ad0-79d3b61612d6">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" physicalQuantity="COST" unit="EURO" description="Costs in EUR/m" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Joint" name="Joint_1591" id="1591bb7b-d143-44fb-95b3-e78f2c45144c">
          <geometry xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
          <port xsi:type="esdl:InPort" connectedTo="920fda15-8b06-40ab-9c25-70c4ec4af6c9" name="In" id="459b9e2d-3588-4652-9ad3-b4c7845728d0" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="1ba3bfca-5ef8-41ff-ab57-b3d5b18a4560 424c2407-d687-4c13-87a8-c9b5ad3dc334" name="Out" id="acb56bd8-eb5f-456e-923a-c5656848a3a8" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="288.2" innerDiameter="0.3127" outerDiameter="0.45" id="92e2a3b4-2106-4bea-b01c-89c38b2a7b3d" name="Pipe_92e2" diameter="DN300">
          <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
            <point xsi:type="esdl:Point" lat="51.99872744841605" lon="4.376206398010255"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="acb56bd8-eb5f-456e-923a-c5656848a3a8" name="In" id="424c2407-d687-4c13-87a8-c9b5ad3dc334" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="61f03284-a3b3-4748-b34d-31bc2bdcdd3d" name="Out" id="c9bd9209-6b43-46b0-b315-6f465c950a94" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
              <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
              <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
              <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="442e7c97-693f-49db-b2ef-3a61a1c84ccc">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
      </area>
      <area xsi:type="esdl:Area" id="4d129d60-44fb-49f1-b440-ee83b3396912" name="Xsport">
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="51.99669290239127" lon="4.370155334472657"/>
            <point xsi:type="esdl:Point" lat="51.997749820957424" lon="4.374661445617677"/>
            <point xsi:type="esdl:Point" lat="51.99539814314251" lon="4.376292228698731"/>
            <point xsi:type="esdl:Point" lat="51.99642866885095" lon="4.380154609680177"/>
            <point xsi:type="esdl:Point" lat="51.99534529503086" lon="4.381270408630372"/>
            <point xsi:type="esdl:Point" lat="51.99291421446105" lon="4.372386932373048"/>
          </exterior>
        </geometry>
        <asset xsi:type="esdl:HeatingDemand" power="25000000.0" name="HeatingDemand_5d98" id="5d98329e-e593-4b67-aa8b-40ce2a9531db">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99461862717003" lon="4.374768733978272"/>
          <port xsi:type="esdl:InPort" connectedTo="0dfd35e1-3273-452b-a8e9-0edc558ad204" name="In" id="6d7f2662-daac-4f09-ad23-94069b7429b0" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1">
            <profile xsi:type="esdl:InfluxDBProfile" multiplier="8.0" filters="" startDate="2018-12-31T23:00:00.000000+0000" field="demand4_MW" database="energy_profiles" host="profiles.warmingup.info" port="443" endDate="2019-12-31T22:00:00.000000+0000" id="05459e5b-d8b9-48e3-bc7a-fa6f1f2f7f68" measurement="WarmingUp default profiles">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" id="906f06cf-0b5a-4d32-9c89-2483b0880f73" name="Out"/>
          <costInformation xsi:type="esdl:CostInformation" id="95b2b2cb-6787-4f7e-8bd1-0d46e21d29ea">
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="41f92a36-2b13-41a4-a654-abc3e04f4737">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="a013a4eb-bc48-45b4-9f79-5faf32835cf8"/>
            </investmentCosts>
          </costInformation>
        </asset>
      </area>
      <area xsi:type="esdl:Area" id="815405ce-057a-4229-9bcf-9c515813f780" name="Kluyverpark">
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="51.992279997820766" lon="4.373159408569337"/>
            <point xsi:type="esdl:Point" lat="51.99402407196604" lon="4.3819570541381845"/>
            <point xsi:type="esdl:Point" lat="51.98961090434427" lon="4.385046958923341"/>
            <point xsi:type="esdl:Point" lat="51.98712665460617" lon="4.377150535583497"/>
          </exterior>
        </geometry>
        <asset xsi:type="esdl:HeatingDemand" power="25000000.0" name="HeatingDemand_13d1" id="13d18f34-caa6-4fef-ba8e-8e2bbe3b184f">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99118331035038" lon="4.380176067352296"/>
          <port xsi:type="esdl:InPort" connectedTo="3367f7bc-73da-4669-8d92-817a6c355777" name="In" id="1c5f5573-fbd8-436b-b602-e8d085b38fae" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069">
            <profile xsi:type="esdl:InfluxDBProfile" multiplier="10.0" filters="" startDate="2018-12-31T23:00:00.000000+0000" field="demand5_MW" database="energy_profiles" host="profiles.warmingup.info" port="443" endDate="2019-12-31T22:00:00.000000+0000" id="8840231a-bee6-4b93-a735-e429b3ad472c" measurement="WarmingUp default profiles">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" id="45555242-5bef-42d6-bd46-ab4b81aa1189" name="Out"/>
          <costInformation xsi:type="esdl:CostInformation" id="893354c9-2516-4e5d-adf3-26283292dfdd">
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="7fd12178-8441-46ac-bf53-fcf81243dcc0">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="952228d7-bb23-4980-8824-34da8f6bdf77"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:GenericProducer" power="10000000.0" name="GenericProducer_e7d4" id="e7d42e39-275b-4723-88ce-6f53de43a1f9">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.98972982773691" lon="4.381334781646729"/>
          <port xsi:type="esdl:OutPort" connectedTo="f181c6c2-30d8-411f-9281-a2148ea1392d" name="Out" id="7e98c563-5b4f-408d-9bc6-4c18ed60ffb8" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:InPort" id="3894549d-0548-4860-99e9-f2f8c6999ae1" name="In"/>
          <costInformation xsi:type="esdl:CostInformation" id="3b3ad03a-e25e-4a72-bedb-2d9e2f072c1f">
            <variableOperationalCosts xsi:type="esdl:SingleValue" value="100.0" id="86000dd6-1343-4cec-ae78-2bf64531d12b">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" perMultiplier="MEGA" description="Cost in EUR/MWh" unit="EURO" physicalQuantity="COST" id="46e3a198-0f0c-4e86-8a2f-dd1f07ef5d91"/>
            </variableOperationalCosts>
            <installationCosts xsi:type="esdl:SingleValue" value="500000.0" id="ad6887fe-665e-4cb7-a253-b02e634ee62a">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" unit="EURO" physicalQuantity="COST" id="ab86121c-4475-429f-8e92-62b2dc2a7c9f"/>
            </installationCosts>
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="403c649d-2a50-47dc-a41d-dd9cd1d00a9b">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="68819dd5-2e8e-4e6c-9a38-2dbcf4381fc0"/>
            </investmentCosts>
            <fixedOperationalCosts xsi:type="esdl:SingleValue" value="10000.0" id="8bbcf767-982c-49f1-b24c-4a4609b3a86d">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="df9e8833-bbdc-4a6e-8148-394317ac6039"/>
            </fixedOperationalCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:GenericConversion" power="15000000.0" efficiency="0.99" id="fa3307a7-9422-4051-8e9f-c0bd4b612690" name="GenericConversion_fa33">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.989994100812105" lon="4.375991821289063"/>
          <port xsi:type="esdl:InPort" connectedTo="a897a7ca-5b60-4552-80aa-8333587eca41" name="PrimIn" id="d0a39b01-2cc1-4869-a853-a2a1e65c9c1e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" id="b3c6c264-b0e9-47f7-88ff-d18ed934ef3e" name="PrimOut"/>
          <port xsi:type="esdl:InPort" id="eda6adb8-a88c-4fd0-ac40-45d2e1f3578b" name="SecIn"/>
          <port xsi:type="esdl:OutPort" connectedTo="d3f37217-014b-40e4-a19a-92c54c8e4d3e" name="SecOut" id="4a31d9cb-8ec9-487c-8678-0e373ea0e9d7" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <costInformation xsi:type="esdl:CostInformation" id="1815f197-424d-473b-b930-a91591a1694a">
            <fixedOperationalCosts xsi:type="esdl:SingleValue" value="1500.0" id="7e0988d1-b0f7-40ab-99ba-2c416c0acd0f">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="00fadb02-4dae-4ad9-8b42-e8a856737e29"/>
            </fixedOperationalCosts>
            <investmentCosts xsi:type="esdl:SingleValue" value="75000.0" id="d839ff1d-e696-4df0-b6c9-f6e0bb0d1600">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="f830b95f-8369-435f-91a9-9bde58a97761"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="283.4" innerDiameter="0.3127" outerDiameter="0.45" id="16d9d3fd-a715-48fb-a471-85104c85c477" name="Pipe_16d9" diameter="DN300">
          <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.989994100812105" lon="4.375991821289063"/>
            <point xsi:type="esdl:Point" lat="51.98968027636168" lon="4.377381205558778"/>
            <point xsi:type="esdl:Point" lat="51.98997428038556" lon="4.379172921180726"/>
            <point xsi:type="esdl:Point" lat="51.99034095685646" lon="4.379714727401734"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="4a31d9cb-8ec9-487c-8678-0e373ea0e9d7" name="In" id="d3f37217-014b-40e4-a19a-92c54c8e4d3e" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="18029b4e-46e5-41e4-94f0-dcfce11609b4" name="Out" id="b51ab54d-a158-470b-b397-8375d3a62430" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
              <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
              <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
              <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="7af869d5-da87-41d4-b2b1-026f2a846fd2">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Joint" id="f0da2ffb-9f5e-4832-90bf-bce36224ceb4" name="Joint_f0da">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.990358299588166" lon="4.379737526178361"/>
          <port xsi:type="esdl:InPort" connectedTo="b51ab54d-a158-470b-b397-8375d3a62430 50ee156e-a49c-4b81-91f7-8e395ef773fd" name="In" id="18029b4e-46e5-41e4-94f0-dcfce11609b4" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="94810176-f76d-4ec2-8318-344584593538" name="Out" id="7497949d-2872-4ab6-b144-e7e0ffc2640b" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="96.5" innerDiameter="0.3127" outerDiameter="0.45" id="7535912b-0436-41f7-8436-91577b7ae5d7" name="Pipe_7535" diameter="DN300">
          <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.990358299588166" lon="4.379737526178361"/>
            <point xsi:type="esdl:Point" lat="51.99118331035038" lon="4.380176067352296"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="7497949d-2872-4ab6-b144-e7e0ffc2640b" name="In" id="94810176-f76d-4ec2-8318-344584593538" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="1c5f5573-fbd8-436b-b602-e8d085b38fae" name="Out" id="3367f7bc-73da-4669-8d92-817a6c355777" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
              <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
              <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
              <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="b2bd6b3b-fe6a-4353-a575-65dcb504a281">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="129.8" innerDiameter="0.3127" outerDiameter="0.45" id="26b644e1-95f2-4320-ad4f-6495c8d3b630" name="Pipe_26b6" diameter="DN300">
          <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.98972982773691" lon="4.381334781646729"/>
            <point xsi:type="esdl:Point" lat="51.990358299588166" lon="4.379737526178361"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="7e98c563-5b4f-408d-9bc6-4c18ed60ffb8" name="In" id="f181c6c2-30d8-411f-9281-a2148ea1392d" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="18029b4e-46e5-41e4-94f0-dcfce11609b4" name="Out" id="50ee156e-a49c-4b81-91f7-8e395ef773fd" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
              <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
              <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
              <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="8b5b5cb5-324d-4d82-937b-3177a3a65e34">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
      </area>
      <asset xsi:type="esdl:ResidualHeatSource" power="30000000.0" name="ResidualHeatSource_ec0a" id="ec0a2222-d6fe-4cb6-aff0-237e08174fa8">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.994420442979276" lon="4.364640712738038"/>
        <port xsi:type="esdl:OutPort" connectedTo="24d83ef4-39c1-4b43-b541-889d70c7d996" name="Out" id="13305198-7cb2-4432-be42-bb9397acda0e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:InPort" id="8d27a37a-082e-49af-923d-b878fd19cf3f" name="In"/>
        <costInformation xsi:type="esdl:CostInformation" id="df01f903-c501-4fb3-9216-744e490d92cb">
          <variableOperationalCosts xsi:type="esdl:SingleValue" value="20.0" id="6895c9ee-eb7d-4f30-86cc-483210f055fd">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" perMultiplier="MEGA" description="Cost in EUR/MWh" unit="EURO" physicalQuantity="COST" id="5b5536c9-b0f3-4d81-9ce4-8325a1170aaa"/>
          </variableOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="2b3593ea-6359-4cc1-b099-53a6d3af7ec5">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" unit="EURO" physicalQuantity="COST" id="eb86cc68-4c3b-4767-ba2d-2fda540637d2"/>
          </installationCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="200000.0" id="e29c3cbe-f718-4a84-8ce3-1aa6bfe35d34">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="b02b49f1-fb38-4154-989e-00e2531ca68c"/>
          </investmentCosts>
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="60000.0" id="8c5955a8-ef9b-4fe9-9f22-4c1d2bc32385">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="6b1ef088-0a9f-4f5b-af72-75d9d412f0f3"/>
          </fixedOperationalCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="50000000.0" name="ResidualHeatSource_54b1" id="54b13f38-8835-4e75-8acc-780a14fa8dbd">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.004328577925094" lon="4.370477199554444"/>
        <port xsi:type="esdl:OutPort" connectedTo="a17c3c44-2296-47e7-b16e-e72593d46cc5" name="Out" id="010265a0-43b7-44dc-bf06-72519dac9ee5" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:InPort" id="2b4fdbd3-6904-4e60-b08e-9bcf6448e265" name="In"/>
        <costInformation xsi:type="esdl:CostInformation" id="ed861601-3428-4381-88cb-9a1452aed93e">
          <variableOperationalCosts xsi:type="esdl:SingleValue" value="40.0" id="08aab920-91f7-4f75-9fc7-b46c0c69c69d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" perMultiplier="MEGA" description="Cost in EUR/MWh" unit="EURO" physicalQuantity="COST" id="b5b04d25-41a0-4f14-abb5-bf8d98c7f0dd"/>
          </variableOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="bad6f69c-e77b-497e-b0be-83df5b7680df">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" unit="EURO" physicalQuantity="COST" id="8abbad22-0502-4c21-bbf9-7a7c0baa4341"/>
          </installationCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="500000.0" id="0d14d98e-acc3-473b-aca6-76e85fa0922f">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="2a9d1538-b945-4b1a-b951-e71e38f8c9ef"/>
          </investmentCosts>
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="120000.0" id="26ce5018-fb93-4ed0-ba48-c04470a0cf78">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="6dc663e0-9f60-49e3-aa91-eb0e05a338fd"/>
          </fixedOperationalCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" length="843.42" innerDiameter="0.3938" outerDiameter="0.56" id="90d7afb4-486c-4340-a698-da12137e0306" name="Pipe_90d7" diameter="DN400">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.004328577925094" lon="4.370477199554444"/>
          <point xsi:type="esdl:Point" lat="52.00332465343612" lon="4.370799064636231"/>
          <point xsi:type="esdl:Point" lat="52.0026509546404" lon="4.366292953491212"/>
          <point xsi:type="esdl:Point" lat="52.00084116453207" lon="4.366314411163331"/>
          <point xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <port xsi:type="esdl:InPort" connectedTo="010265a0-43b7-44dc-bf06-72519dac9ee5" name="In" id="a17c3c44-2296-47e7-b16e-e72593d46cc5" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="e78c77ed-4744-499e-a6cf-dabb06f5bfac" name="Out" id="b9abe6ee-3285-4d34-b602-82dc0650ef14" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="80dfa3cf-155f-41de-93c4-fb8acafe2fae">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" value="2840.6">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" physicalQuantity="COST" unit="EURO" description="Costs in EUR/m" id="9169bd50-197f-4d6b-aaac-b383a59c815d"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_b3e4" id="b3e415da-242d-4003-8a26-447529a7a9e4">
        <geometry xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
        <port xsi:type="esdl:InPort" connectedTo="b9abe6ee-3285-4d34-b602-82dc0650ef14" name="In" id="e78c77ed-4744-499e-a6cf-dabb06f5bfac" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="86838250-3430-4918-9806-e3b5dc400055 a5ba6248-a54b-4430-b873-158497eb0876" name="Out" id="e585e343-cb4d-45c4-bc16-0a7e862f8a75" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="588.97" innerDiameter="0.3938" outerDiameter="0.56" id="de369aac-a41a-4965-9b46-2e95adc23aa1" name="Pipe_de36" diameter="DN400">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <port xsi:type="esdl:InPort" connectedTo="e585e343-cb4d-45c4-bc16-0a7e862f8a75" name="In" id="a5ba6248-a54b-4430-b873-158497eb0876" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="4f7746c7-53a6-431b-bfaf-04ae200b712c" name="Out" id="fee183b9-5c67-48d1-9994-470733a41fb9" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="80dfa3cf-155f-41de-93c4-fb8acafe2fae">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" value="2840.6">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" physicalQuantity="COST" unit="EURO" description="Costs in EUR/m" id="9169bd50-197f-4d6b-aaac-b383a59c815d"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" length="734.05" innerDiameter="0.3938" outerDiameter="0.56" id="9669d949-974e-428e-a261-bc62731fd5fb" name="Pipe_9669" diameter="DN400">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
          <point xsi:type="esdl:Point" lat="51.99245176571451" lon="4.371185302734376"/>
          <point xsi:type="esdl:Point" lat="51.989386270407564" lon="4.3740177154541025"/>
          <point xsi:type="esdl:Point" lat="51.989994100812105" lon="4.375991821289063"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <port xsi:type="esdl:InPort" connectedTo="ae7152ae-f443-4478-97ae-2a4e4abccd63" name="In" id="702894a7-7da7-42dc-99c9-a86352a0d42b" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="d0a39b01-2cc1-4869-a853-a2a1e65c9c1e" name="Out" id="a897a7ca-5b60-4552-80aa-8333587eca41" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="80dfa3cf-155f-41de-93c4-fb8acafe2fae">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" value="2840.6">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" physicalQuantity="COST" unit="EURO" description="Costs in EUR/m" id="9169bd50-197f-4d6b-aaac-b383a59c815d"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_21a0" id="21a04f29-2a92-4e5f-8160-072fc4db05fc">
        <geometry xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
        <port xsi:type="esdl:InPort" connectedTo="fee183b9-5c67-48d1-9994-470733a41fb9 5e7c7a5b-c57a-4551-89c8-174c4db754e7" name="In" id="4f7746c7-53a6-431b-bfaf-04ae200b712c" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="702894a7-7da7-42dc-99c9-a86352a0d42b 0bc8b306-7783-459f-bdbf-cb4bfb4ca200" name="Out" id="ae7152ae-f443-4478-97ae-2a4e4abccd63" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="314.5" innerDiameter="0.3127" outerDiameter="0.45" id="5ec66dff-0b2e-479e-969d-93074270b9ec" name="Pipe_5ec6" diameter="DN300">
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
          <point xsi:type="esdl:Point" lat="51.99461862717003" lon="4.374768733978272"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="ae7152ae-f443-4478-97ae-2a4e4abccd63" name="In" id="0bc8b306-7783-459f-bdbf-cb4bfb4ca200" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="6d7f2662-daac-4f09-ad23-94069b7429b0" name="Out" id="0dfd35e1-3273-452b-a8e9-0edc558ad204" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="8d87e825-fcc5-4db9-9f69-85e9bea755a2">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" length="386.9" innerDiameter="0.3127" outerDiameter="0.45" id="11bb13db-a1fb-4dbd-a6a4-927a2ef4b68b" name="Pipe_11bb" diameter="DN300">
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.994420442979276" lon="4.364640712738038"/>
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="13305198-7cb2-4432-be42-bb9397acda0e" name="In" id="24d83ef4-39c1-4b43-b541-889d70c7d996" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="4f7746c7-53a6-431b-bfaf-04ae200b712c" name="Out" id="5e7c7a5b-c57a-4551-89c8-174c4db754e7" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" name="steel" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" name="PUR" id="e4c0350c-cd79-45b4-a45c-6259c750b478" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" name="HDPE" id="9a97f588-10fe-4a34-b0f2-277862151763" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="97fd42ee-48c3-4fed-baa8-57a1a0205861">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" value="1962.1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
          </investmentCosts>
        </costInformation>
      </asset>
    </area>
  </instance>
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="587aaec6-59e1-42a3-9d05-fda822e823c7">
    <carriers xsi:type="esdl:Carriers" id="6072772c-ea2b-4161-a0f6-cdaf0319f3c2">
      <carrier xsi:type="esdl:HeatCommodity" id="9e201a90-71c0-45bc-a683-6ae7e68b67d1" name="Primary" supplyTemperature="90.0" returnTemperature="50.0"/>
      <carrier xsi:type="esdl:HeatCommodity" id="e32b730d-4c75-4610-8ef7-dfcf461cc069" name="Secondary" supplyTemperature="70.0" returnTemperature="40.0"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="013d8f95-ae1c-448b-83eb-136a19150707">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Power in MW" unit="WATT" physicalQuantity="POWER" multiplier="MEGA" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
    </quantityAndUnits>
  </energySystemInformation>
</esdl:EnergySystem>
