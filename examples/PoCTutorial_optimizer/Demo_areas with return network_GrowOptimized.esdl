<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" name="Demo_areas with return network_GrowOptimized" description="" version="16" esdlVersion="v2303" id="d68e80a0-2499-4c60-87d9-3e212bbdafe2">
  <instance xsi:type="esdl:Instance" id="411b572c-f96b-49fa-8a0a-389705cbf58d" name="Untitled instance">
    <area xsi:type="esdl:Area" id="1e502f41-ce24-4609-b518-cd76a8aafb59" name="Delft">
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
        <KPIs xsi:type="esdl:KPIs" id="e28368c7-add0-4251-b0fd-2a220b63382a">
          <kpi xsi:type="esdl:DoubleKPI" name="Investment" value="4.825963520739156">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
          <kpi xsi:type="esdl:DoubleKPI" name="Installation">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
        </KPIs>
        <asset xsi:type="esdl:HeatingDemand" power="25000000.0" name="HeatingDemand_c5c8" id="c5c8678d-e624-4878-95cc-baa9e8809d5e">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.000365587107815" lon="4.371979236602784"/>
          <port xsi:type="esdl:InPort" connectedTo="3574833a-a997-42aa-b93f-2feb32350f9c" name="In" id="751b6c76-92d3-4d7c-83a9-ca90ddc73b0e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1">
            <profile xsi:type="esdl:InfluxDBProfile" multiplier="8.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" database="energy_profiles" field="demand2_MW" host="profiles.warmingup.info" port="443" endDate="2019-12-31T22:00:00.000000+0000" id="5645e777-6500-4b99-80eb-6f09f8d8910e" measurement="WarmingUp default profiles">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" connectedTo="880b4ede-37b5-4849-83e8-f1ed0f2d5a1a" name="Out" id="c81b81a7-2ed3-4c20-9c4b-ecc05eff5627" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <costInformation xsi:type="esdl:CostInformation" id="1f927790-982a-4ef5-b99a-4e9e9aceabd2">
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="ae0d67c2-2e06-48fd-a150-2fabe403a076">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="51492d75-208e-4a02-a6c9-a1c2e5be6a5f"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:HeatStorage" capacity="3600000000.0" maxDischargeRate="1000000.0" maxChargeRate="1000000.0" name="HeatStorage_171d" id="171dd49c-b2f2-47d0-99d8-a845ad9ee33b">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99872744841605" lon="4.376206398010255"/>
          <port xsi:type="esdl:InPort" connectedTo="c9bd9209-6b43-46b0-b315-6f465c950a94" name="In" id="61f03284-a3b3-4748-b34d-31bc2bdcdd3d" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="374aa328-c36e-4b1a-a530-70fb96205bb3" name="Out" id="96cdb905-3636-4bea-b989-d51afcac0d97" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <costInformation xsi:type="esdl:CostInformation" id="200a0bf3-3e61-452f-a01c-e6f2e13b04c8">
            <investmentCosts xsi:type="esdl:SingleValue" value="500.0" id="07eb7de0-5274-4697-948f-9e434657ea4c">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="CUBIC_METRE" description="Cost in EUR/m3" unit="EURO" physicalQuantity="COST" id="a5e1e949-0e02-4c21-a02c-fe0dd79ed1a3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="344.13" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe6" id="Pipe6" related="Pipe6_ret" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
            <point xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
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
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="80.38" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe7" id="Pipe7" related="Pipe7_ret" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
            <point xsi:type="esdl:Point" lat="51.99969184377424" lon="4.372311830520631"/>
            <point xsi:type="esdl:Point" lat="52.000365587107815" lon="4.371979236602784"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
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
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Joint" name="Joint_1591" id="1591bb7b-d143-44fb-95b3-e78f2c45144c">
          <geometry xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
          <port xsi:type="esdl:InPort" connectedTo="920fda15-8b06-40ab-9c25-70c4ec4af6c9" name="In" id="459b9e2d-3588-4652-9ad3-b4c7845728d0" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="1ba3bfca-5ef8-41ff-ab57-b3d5b18a4560 424c2407-d687-4c13-87a8-c9b5ad3dc334" name="Out" id="acb56bd8-eb5f-456e-923a-c5656848a3a8" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="288.2" innerDiameter="0.1603" outerDiameter="0.25" name="Pipe8" id="Pipe8" related="Pipe8_ret" diameter="DN150">
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.99967863301928" lon="4.372290372848512"/>
            <point xsi:type="esdl:Point" lat="51.99872744841605" lon="4.376206398010255"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
          <port xsi:type="esdl:InPort" connectedTo="acb56bd8-eb5f-456e-923a-c5656848a3a8" name="In" id="424c2407-d687-4c13-87a8-c9b5ad3dc334" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="61f03284-a3b3-4748-b34d-31bc2bdcdd3d" name="Out" id="c9bd9209-6b43-46b0-b315-6f465c950a94" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
              <matter xsi:type="esdl:Material" id="fa85538e-ebfa-4bce-8386-04980e793e18" name="steel" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
              <matter xsi:type="esdl:Material" id="3bafa031-f40f-42fc-b409-e35fffe5f457" name="PUR" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
              <matter xsi:type="esdl:Material" id="893337e3-58f1-4fb4-8c25-68d71b11fb71" name="HDPE" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="442e7c97-693f-49db-b2ef-3a61a1c84ccc">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Joint" name="Joint_1591_ret" id="e9f0319e-9509-4667-8f3b-b53427e93188">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99976863310928" lon="4.371508069839314"/>
          <port xsi:type="esdl:OutPort" connectedTo="3e11b654-c5ff-4cf5-a7ed-23e5758a84c3" name="ret_port" id="02e4b3f2-3b9f-499a-b6bc-e5af5d54197e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <port xsi:type="esdl:InPort" connectedTo="f145e3cd-ee40-49b9-a05a-75659fe099dd 2f6d8d3f-6f9c-443a-9401-350d944fa87f" name="ret_port" id="64490463-6f86-4924-8b85-38bafce82628" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="344.13" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe6_ret" id="Pipe6_ret" related="Pipe6" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99976863310928" lon="4.371508069839314"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99914772322822" lon="4.366580576596763"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="02e4b3f2-3b9f-499a-b6bc-e5af5d54197e" name="In_ret" id="3e11b654-c5ff-4cf5-a7ed-23e5758a84c3" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="a455fc93-a668-42f6-a014-70f482fb3efc" name="Out_ret" id="935f3cbd-95b4-4d81-be5f-ca68315c9f4f" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="80.38" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe7_ret" id="Pipe7_ret" related="Pipe7" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" CRS="WGS84" lat="52.00045558719781" lon="4.371200179465164"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.999781843864234" lon="4.371529590191046"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99976863310928" lon="4.371508069839314"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="c81b81a7-2ed3-4c20-9c4b-ecc05eff5627" name="In_ret" id="880b4ede-37b5-4849-83e8-f1ed0f2d5a1a" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="64490463-6f86-4924-8b85-38bafce82628" name="Out_ret" id="2f6d8d3f-6f9c-443a-9401-350d944fa87f" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="288.2" innerDiameter="0.1603" outerDiameter="0.25" name="Pipe8_ret" id="Pipe8_ret" related="Pipe8" diameter="DN150">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.998817448506045" lon="4.375419555106473"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99976863310928" lon="4.371508069839314"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="96cdb905-3636-4bea-b989-d51afcac0d97" name="In_ret" id="374aa328-c36e-4b1a-a530-70fb96205bb3" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="64490463-6f86-4924-8b85-38bafce82628" name="Out_ret" id="f145e3cd-ee40-49b9-a05a-75659fe099dd" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
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
        <KPIs xsi:type="esdl:KPIs" id="1608412a-a65d-4900-aeed-c6786207b5cb">
          <kpi xsi:type="esdl:DoubleKPI" name="Investment" value="2.5">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
          <kpi xsi:type="esdl:DoubleKPI" name="Installation">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
        </KPIs>
        <asset xsi:type="esdl:HeatingDemand" power="25000000.0" name="HeatingDemand_5d98" id="5d98329e-e593-4b67-aa8b-40ce2a9531db">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99461862717003" lon="4.374768733978272"/>
          <port xsi:type="esdl:InPort" connectedTo="0dfd35e1-3273-452b-a8e9-0edc558ad204" name="In" id="6d7f2662-daac-4f09-ad23-94069b7429b0" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1">
            <profile xsi:type="esdl:InfluxDBProfile" multiplier="8.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" database="energy_profiles" field="demand4_MW" host="profiles.warmingup.info" port="443" endDate="2019-12-31T22:00:00.000000+0000" id="e92f1544-5170-4922-b4fe-3ae0a298d0de" measurement="WarmingUp default profiles">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" connectedTo="4d4ea4f1-04fb-4d25-8aaf-9e1223651f81" name="Out" id="906f06cf-0b5a-4d32-9c89-2483b0880f73" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
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
        <KPIs xsi:type="esdl:KPIs" id="3b6bd0b4-109a-48a4-a29d-e8521652213e">
          <kpi xsi:type="esdl:DoubleKPI" name="Investment" value="6.268551939999998">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
          <kpi xsi:type="esdl:DoubleKPI" name="Installation" value="0.5">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
          <kpi xsi:type="esdl:DoubleKPI" name="Variable OPEX" value="0.033231041684931496">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
          <kpi xsi:type="esdl:DoubleKPI" name="Fixed OPEX" value="0.1">
            <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" multiplier="MEGA" unit="EURO"/>
          </kpi>
        </KPIs>
        <asset xsi:type="esdl:HeatingDemand" power="25000000.0" name="HeatingDemand_13d1" id="13d18f34-caa6-4fef-ba8e-8e2bbe3b184f">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99118331035038" lon="4.380176067352296"/>
          <port xsi:type="esdl:InPort" connectedTo="3367f7bc-73da-4669-8d92-817a6c355777" name="In" id="1c5f5573-fbd8-436b-b602-e8d085b38fae" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069">
            <profile xsi:type="esdl:InfluxDBProfile" multiplier="10.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" database="energy_profiles" field="demand5_MW" host="profiles.warmingup.info" port="443" endDate="2019-12-31T22:00:00.000000+0000" id="10320809-979e-44e4-b39a-764ded8de3b7" measurement="WarmingUp default profiles">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" connectedTo="1794f187-75b5-4609-a4e8-f03a75066ef2" name="Out" id="45555242-5bef-42d6-bd46-ab4b81aa1189" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
          <costInformation xsi:type="esdl:CostInformation" id="893354c9-2516-4e5d-adf3-26283292dfdd">
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="7fd12178-8441-46ac-bf53-fcf81243dcc0">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="952228d7-bb23-4980-8824-34da8f6bdf77"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:GenericProducer" power="10000000.0" name="GenericProducer_e7d4" id="e7d42e39-275b-4723-88ce-6f53de43a1f9">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.98972982773691" lon="4.381334781646729"/>
          <port xsi:type="esdl:OutPort" connectedTo="f181c6c2-30d8-411f-9281-a2148ea1392d" name="Out" id="7e98c563-5b4f-408d-9bc6-4c18ed60ffb8" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:InPort" connectedTo="5b7d8822-58e5-4d67-8334-3a4f20b02efb" name="In" id="3894549d-0548-4860-99e9-f2f8c6999ae1" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
          <costInformation xsi:type="esdl:CostInformation" id="3b3ad03a-e25e-4a72-bedb-2d9e2f072c1f">
            <variableOperationalCosts xsi:type="esdl:SingleValue" value="100.0" id="86000dd6-1343-4cec-ae78-2bf64531d12b">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" perMultiplier="MEGA" description="Cost in EUR/MWh" unit="EURO" physicalQuantity="COST" id="46e3a198-0f0c-4e86-8a2f-dd1f07ef5d91"/>
            </variableOperationalCosts>
            <installationCosts xsi:type="esdl:SingleValue" value="500000.0" id="ad6887fe-665e-4cb7-a253-b02e634ee62a">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" id="ab86121c-4475-429f-8e92-62b2dc2a7c9f" description="Cost in EUR" unit="EURO"/>
            </installationCosts>
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="403c649d-2a50-47dc-a41d-dd9cd1d00a9b">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="68819dd5-2e8e-4e6c-9a38-2dbcf4381fc0"/>
            </investmentCosts>
            <fixedOperationalCosts xsi:type="esdl:SingleValue" value="10000.0" id="8bbcf767-982c-49f1-b24c-4a4609b3a86d">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="df9e8833-bbdc-4a6e-8148-394317ac6039"/>
            </fixedOperationalCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:GenericConversion" efficiency="0.99" power="15000000.0" id="fa3307a7-9422-4051-8e9f-c0bd4b612690" name="GenericConversion_fa33" state="OPTIONAL">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.989994100812105" lon="4.375991821289063"/>
          <port xsi:type="esdl:InPort" connectedTo="a897a7ca-5b60-4552-80aa-8333587eca41" name="PrimIn" id="d0a39b01-2cc1-4869-a853-a2a1e65c9c1e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
          <port xsi:type="esdl:OutPort" connectedTo="cae6d1fb-7402-4e24-8870-c7d1c4c86a8e" name="PrimOut" id="b3c6c264-b0e9-47f7-88ff-d18ed934ef3e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
          <port xsi:type="esdl:InPort" connectedTo="310a945c-43a6-45e0-aaa0-0e216702fa53" name="SecIn" id="eda6adb8-a88c-4fd0-ac40-45d2e1f3578b" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
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
        <asset xsi:type="esdl:Pipe" length="283.4" innerDiameter="0.263" outerDiameter="0.4" name="Pipe9" id="Pipe9" related="Pipe9_ret" diameter="DN250">
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.989994100812105" lon="4.375991821289063"/>
            <point xsi:type="esdl:Point" lat="51.98968027636168" lon="4.377381205558778"/>
            <point xsi:type="esdl:Point" lat="51.98997428038556" lon="4.379172921180726"/>
            <point xsi:type="esdl:Point" lat="51.99034095685646" lon="4.379714727401734"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
          <port xsi:type="esdl:InPort" connectedTo="4a31d9cb-8ec9-487c-8678-0e373ea0e9d7" name="In" id="d3f37217-014b-40e4-a19a-92c54c8e4d3e" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="18029b4e-46e5-41e4-94f0-dcfce11609b4" name="Out" id="b51ab54d-a158-470b-b397-8375d3a62430" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.005">
              <matter xsi:type="esdl:Material" id="faac539b-4b7c-43f8-abcd-f08fa2652b7b" name="steel" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0587">
              <matter xsi:type="esdl:Material" id="d23b4eeb-a419-4c16-bc7e-280a76116f04" name="PUR" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0048">
              <matter xsi:type="esdl:Material" id="a2b91e8d-471d-4276-a8f6-4efb01054b4e" name="HDPE" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="7af869d5-da87-41d4-b2b1-026f2a846fd2">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Joint" name="Joint_f0da" id="f0da2ffb-9f5e-4832-90bf-bce36224ceb4">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.990358299588166" lon="4.379737526178361"/>
          <port xsi:type="esdl:InPort" connectedTo="b51ab54d-a158-470b-b397-8375d3a62430 50ee156e-a49c-4b81-91f7-8e395ef773fd" name="In" id="18029b4e-46e5-41e4-94f0-dcfce11609b4" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="94810176-f76d-4ec2-8318-344584593538" name="Out" id="7497949d-2872-4ab6-b144-e7e0ffc2640b" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="96.5" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe10" id="Pipe10" related="Pipe10_ret" diameter="DN300">
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.990358299588166" lon="4.379737526178361"/>
            <point xsi:type="esdl:Point" lat="51.99118331035038" lon="4.380176067352296"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
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
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Pipe" length="129.8" innerDiameter="0.2101" outerDiameter="0.315" name="Pipe11" id="Pipe11" related="Pipe11_ret" diameter="DN200">
          <geometry xsi:type="esdl:Line" CRS="WGS84">
            <point xsi:type="esdl:Point" lat="51.98972982773691" lon="4.381334781646729"/>
            <point xsi:type="esdl:Point" lat="51.990358299588166" lon="4.379737526178361"/>
          </geometry>
          <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
          <port xsi:type="esdl:InPort" connectedTo="7e98c563-5b4f-408d-9bc6-4c18ed60ffb8" name="In" id="f181c6c2-30d8-411f-9281-a2148ea1392d" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <port xsi:type="esdl:OutPort" connectedTo="18029b4e-46e5-41e4-94f0-dcfce11609b4" name="Out" id="50ee156e-a49c-4b81-91f7-8e395ef773fd" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069"/>
          <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
              <matter xsi:type="esdl:Material" id="930aa5cf-b76e-4049-afa7-ea79445faf55" name="steel" thermalConductivity="52.15"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
              <matter xsi:type="esdl:Material" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b" name="PUR" thermalConductivity="0.027"/>
            </component>
            <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
              <matter xsi:type="esdl:Material" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3" name="HDPE" thermalConductivity="0.4"/>
            </component>
          </material>
          <costInformation xsi:type="esdl:CostInformation" id="8b5b5cb5-324d-4d82-937b-3177a3a65e34">
            <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <asset xsi:type="esdl:Joint" name="Joint_f0da_ret" id="58b8c00e-965a-457b-8cf9-13b26215f807">
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.990448299678164" lon="4.378908311071035"/>
          <port xsi:type="esdl:InPort" connectedTo="f3d0f5eb-a215-44ca-8911-9439dfa74633" name="ret_port" id="1a3d201e-9e7f-4467-a1ee-acf91b521766" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="da04a1c0-c18f-4fa5-9ac0-43e26de071df 9d4fa664-bc3e-41ce-a955-4c82f7cb8af8" name="ret_port" id="b1d7edff-1cdd-48bc-ae7b-8c2ec9653e0e" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="283.4" innerDiameter="0.263" outerDiameter="0.4" name="Pipe9_ret" id="Pipe9_ret" related="Pipe9" diameter="DN250">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.990430956946454" lon="4.378885419689728"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99006428047556" lon="4.378341650621005"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.98977027645168" lon="4.376548354367039"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.9900841009021" lon="4.3751606570700075"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="b1d7edff-1cdd-48bc-ae7b-8c2ec9653e0e" name="In_ret" id="da04a1c0-c18f-4fa5-9ac0-43e26de071df" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="eda6adb8-a88c-4fd0-ac40-45d2e1f3578b" name="Out_ret" id="310a945c-43a6-45e0-aaa0-0e216702fa53" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="96.5" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe10_ret" id="Pipe10_ret" related="Pipe10" diameter="DN300">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99127331044038" lon="4.379351233488201"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.990448299678164" lon="4.378908311071035"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="45555242-5bef-42d6-bd46-ab4b81aa1189" name="In_ret" id="1794f187-75b5-4609-a4e8-f03a75066ef2" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="1a3d201e-9e7f-4467-a1ee-acf91b521766" name="Out_ret" id="f3d0f5eb-a215-44ca-8911-9439dfa74633" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
        </asset>
        <asset xsi:type="esdl:Pipe" length="129.8" innerDiameter="0.2101" outerDiameter="0.315" name="Pipe11_ret" id="Pipe11_ret" related="Pipe11" diameter="DN200">
          <geometry xsi:type="esdl:Line">
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.990448299678164" lon="4.378908311071035"/>
            <point xsi:type="esdl:Point" CRS="WGS84" lat="51.989819827826906" lon="4.380502197280458"/>
          </geometry>
          <port xsi:type="esdl:InPort" connectedTo="b1d7edff-1cdd-48bc-ae7b-8c2ec9653e0e" name="In_ret" id="9d4fa664-bc3e-41ce-a955-4c82f7cb8af8" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
          <port xsi:type="esdl:OutPort" connectedTo="3894549d-0548-4860-99e9-f2f8c6999ae1" name="Out_ret" id="5b7d8822-58e5-4d67-8334-3a4f20b02efb" carrier="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret"/>
        </asset>
      </area>
      <KPIs xsi:type="esdl:KPIs" id="20590775-282f-4a4c-9146-0942d256b11d">
        <kpi xsi:type="esdl:DistributionKPI" name="High level cost breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" value="33061750.09410136" label="CAPEX"/>
            <stringItem xsi:type="esdl:StringItem" value="4421843.303524791" label="OPEX"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="Overall cost breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" value="2500000.0" label="Installation"/>
            <stringItem xsi:type="esdl:StringItem" value="30561750.09410136" label="Investment"/>
            <stringItem xsi:type="esdl:StringItem" value="2286890.051517862" label="Variable OPEX"/>
            <stringItem xsi:type="esdl:StringItem" value="2134953.2520069294" label="Fixed OPEX"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="CAPEX breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" value="8978971.883362204" label="ResidualHeatSource"/>
            <stringItem xsi:type="esdl:StringItem" value="13958183.792000003" label="Pipe"/>
            <stringItem xsi:type="esdl:StringItem" value="7500000.0" label="HeatingDemand"/>
            <stringItem xsi:type="esdl:StringItem" value="10844.41873915558" label="HeatStorage"/>
            <stringItem xsi:type="esdl:StringItem" value="1500000.0" label="GenericProducer"/>
            <stringItem xsi:type="esdl:StringItem" value="1113749.999999999" label="GenericConversion"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="OPEX breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" value="8341225.602787701" label="ResidualHeatSource"/>
            <stringItem xsi:type="esdl:StringItem" value="4421843.303524791" label="GenericProducer"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="Energy production [Wh]">
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" value="112630667047.39214" label="ResidualHeatSource_ec0a"/>
            <stringItem xsi:type="esdl:StringItem" value="26141722.12719077" label="ResidualHeatSource_54b1"/>
            <stringItem xsi:type="esdl:StringItem" value="332310416.8493149" label="GenericProducer_e7d4"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="TUwijk: Asset cost breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" label="Installation"/>
            <stringItem xsi:type="esdl:StringItem" value="4825963.520739156" label="Investment"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="Xsport: Asset cost breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" label="Installation"/>
            <stringItem xsi:type="esdl:StringItem" value="2500000.0" label="Investment"/>
          </distribution>
        </kpi>
        <kpi xsi:type="esdl:DistributionKPI" name="Kluyverpark: Asset cost breakdown [EUR]">
          <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" unit="EURO"/>
          <distribution xsi:type="esdl:StringLabelDistribution">
            <stringItem xsi:type="esdl:StringItem" value="500000.0" label="Installation"/>
            <stringItem xsi:type="esdl:StringItem" value="6268551.939999999" label="Investment"/>
          </distribution>
        </kpi>
      </KPIs>
      <asset xsi:type="esdl:ResidualHeatSource" power="30000000.0" name="ResidualHeatSource_ec0a" id="ec0a2222-d6fe-4cb6-aff0-237e08174fa8">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.994420442979276" lon="4.364640712738038"/>
        <port xsi:type="esdl:OutPort" connectedTo="24d83ef4-39c1-4b43-b541-889d70c7d996" name="Out" id="13305198-7cb2-4432-be42-bb9397acda0e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:InPort" connectedTo="62bf94ae-b3aa-4084-95ff-b052fe182df1" name="In" id="8d27a37a-082e-49af-923d-b878fd19cf3f" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="df01f903-c501-4fb3-9216-744e490d92cb">
          <variableOperationalCosts xsi:type="esdl:SingleValue" value="20.0" id="6895c9ee-eb7d-4f30-86cc-483210f055fd">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" perMultiplier="MEGA" description="Cost in EUR/MWh" unit="EURO" physicalQuantity="COST" id="5b5536c9-b0f3-4d81-9ce4-8325a1170aaa"/>
          </variableOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="2b3593ea-6359-4cc1-b099-53a6d3af7ec5">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" id="eb86cc68-4c3b-4767-ba2d-2fda540637d2" description="Cost in EUR" unit="EURO"/>
          </installationCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="200000.0" id="e29c3cbe-f718-4a84-8ce3-1aa6bfe35d34">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="b02b49f1-fb38-4154-989e-00e2531ca68c"/>
          </investmentCosts>
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="60000.0" id="8c5955a8-ef9b-4fe9-9f22-4c1d2bc32385">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="6b1ef088-0a9f-4f5b-af72-75d9d412f0f3"/>
          </fixedOperationalCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="1957943.7667244095" name="ResidualHeatSource_54b1" id="54b13f38-8835-4e75-8acc-780a14fa8dbd">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.004328577925094" lon="4.370477199554444"/>
        <port xsi:type="esdl:OutPort" connectedTo="a17c3c44-2296-47e7-b16e-e72593d46cc5" name="Out" id="010265a0-43b7-44dc-bf06-72519dac9ee5" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:InPort" connectedTo="3ca327dc-8876-4b78-be0c-206a0eafe345" name="In" id="2b4fdbd3-6904-4e60-b08e-9bcf6448e265" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="ed861601-3428-4381-88cb-9a1452aed93e">
          <variableOperationalCosts xsi:type="esdl:SingleValue" value="40.0" id="08aab920-91f7-4f75-9fc7-b46c0c69c69d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" perMultiplier="MEGA" description="Cost in EUR/MWh" unit="EURO" physicalQuantity="COST" id="b5b04d25-41a0-4f14-abb5-bf8d98c7f0dd"/>
          </variableOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="bad6f69c-e77b-497e-b0be-83df5b7680df">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" physicalQuantity="COST" id="8abbad22-0502-4c21-bbf9-7a7c0baa4341" description="Cost in EUR" unit="EURO"/>
          </installationCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="500000.0" id="0d14d98e-acc3-473b-aca6-76e85fa0922f">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="2a9d1538-b945-4b1a-b951-e71e38f8c9ef"/>
          </investmentCosts>
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="120000.0" id="26ce5018-fb93-4ed0-ba48-c04470a0cf78">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" perMultiplier="MEGA" description="Cost in EUR/MW" unit="EURO" physicalQuantity="COST" id="6dc663e0-9f60-49e3-aa91-eb0e05a338fd"/>
          </fixedOperationalCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" length="843.42" innerDiameter="0.1603" outerDiameter="0.25" name="Pipe1" id="Pipe1" related="Pipe1_ret" diameter="DN150">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.004328577925094" lon="4.370477199554444"/>
          <point xsi:type="esdl:Point" lat="52.00332465343612" lon="4.370799064636231"/>
          <point xsi:type="esdl:Point" lat="52.0026509546404" lon="4.366292953491212"/>
          <point xsi:type="esdl:Point" lat="52.00084116453207" lon="4.366314411163331"/>
          <point xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
        <port xsi:type="esdl:InPort" connectedTo="010265a0-43b7-44dc-bf06-72519dac9ee5" name="In" id="a17c3c44-2296-47e7-b16e-e72593d46cc5" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="e78c77ed-4744-499e-a6cf-dabb06f5bfac" name="Out" id="b9abe6ee-3285-4d34-b602-82dc0650ef14" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" id="fa85538e-ebfa-4bce-8386-04980e793e18" name="steel" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" id="3bafa031-f40f-42fc-b409-e35fffe5f457" name="PUR" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" id="893337e3-58f1-4fb4-8c25-68d71b11fb71" name="HDPE" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="80dfa3cf-155f-41de-93c4-fb8acafe2fae">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="9169bd50-197f-4d6b-aaac-b383a59c815d"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_b3e4" id="b3e415da-242d-4003-8a26-447529a7a9e4">
        <geometry xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
        <port xsi:type="esdl:InPort" connectedTo="b9abe6ee-3285-4d34-b602-82dc0650ef14" name="In" id="e78c77ed-4744-499e-a6cf-dabb06f5bfac" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="86838250-3430-4918-9806-e3b5dc400055 a5ba6248-a54b-4430-b873-158497eb0876" name="Out" id="e585e343-cb4d-45c4-bc16-0a7e862f8a75" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="588.97" innerDiameter="0.3938" outerDiameter="0.56" name="Pipe2" id="Pipe2" related="Pipe2_ret" diameter="DN400">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.99905772313822" lon="4.367365837097169"/>
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
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
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="9169bd50-197f-4d6b-aaac-b383a59c815d"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" length="734.05" innerDiameter="0.2101" outerDiameter="0.315" name="Pipe3" id="Pipe3" related="Pipe3_ret" diameter="DN200">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
          <point xsi:type="esdl:Point" lat="51.99245176571451" lon="4.371185302734376"/>
          <point xsi:type="esdl:Point" lat="51.989386270407564" lon="4.3740177154541025"/>
          <point xsi:type="esdl:Point" lat="51.989994100812105" lon="4.375991821289063"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
        <port xsi:type="esdl:InPort" connectedTo="ae7152ae-f443-4478-97ae-2a4e4abccd63" name="In" id="702894a7-7da7-42dc-99c9-a86352a0d42b" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="d0a39b01-2cc1-4869-a853-a2a1e65c9c1e" name="Out" id="a897a7ca-5b60-4552-80aa-8333587eca41" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" id="930aa5cf-b76e-4049-afa7-ea79445faf55" name="steel" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b" name="PUR" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3" name="HDPE" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="80dfa3cf-155f-41de-93c4-fb8acafe2fae">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="9169bd50-197f-4d6b-aaac-b383a59c815d"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_21a0" id="21a04f29-2a92-4e5f-8160-072fc4db05fc">
        <geometry xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
        <port xsi:type="esdl:InPort" connectedTo="fee183b9-5c67-48d1-9994-470733a41fb9 5e7c7a5b-c57a-4551-89c8-174c4db754e7" name="In" id="4f7746c7-53a6-431b-bfaf-04ae200b712c" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="702894a7-7da7-42dc-99c9-a86352a0d42b 0bc8b306-7783-459f-bdbf-cb4bfb4ca200" name="Out" id="ae7152ae-f443-4478-97ae-2a4e4abccd63" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="314.5" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe4" id="Pipe4" related="Pipe4_ret" diameter="DN300">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
          <point xsi:type="esdl:Point" lat="51.99461862717003" lon="4.374768733978272"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
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
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" length="386.9" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe5" id="Pipe5" related="Pipe5_ret" diameter="DN300">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.994420442979276" lon="4.364640712738038"/>
          <point xsi:type="esdl:Point" lat="51.99407031543172" lon="4.370262622833253"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
        <port xsi:type="esdl:InPort" connectedTo="13305198-7cb2-4432-be42-bb9397acda0e" name="In" id="24d83ef4-39c1-4b43-b541-889d70c7d996" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <port xsi:type="esdl:OutPort" connectedTo="4f7746c7-53a6-431b-bfaf-04ae200b712c" name="Out" id="5e7c7a5b-c57a-4551-89c8-174c4db754e7" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" id="f4cee538-cc3b-4809-bd66-979f2ce9649b" name="steel" thermalConductivity="52.15"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.05785">
            <matter xsi:type="esdl:Material" id="e4c0350c-cd79-45b4-a45c-6259c750b478" name="PUR" thermalConductivity="0.027"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0052">
            <matter xsi:type="esdl:Material" id="9a97f588-10fe-4a34-b0f2-277862151763" name="HDPE" thermalConductivity="0.4"/>
          </component>
        </material>
        <costInformation xsi:type="esdl:CostInformation" id="97fd42ee-48c3-4fed-baa8-57a1a0205861">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" description="Costs in EUR/m" unit="EURO" physicalQuantity="COST" id="983f0959-8566-43ce-a380-782d29406ed3"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_b3e4_ret" id="6df97adc-beff-4925-a7a1-394d7013af7c">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99914772322822" lon="4.366580576596763"/>
        <port xsi:type="esdl:OutPort" connectedTo="02ecbc14-ff04-4e0f-86b0-346f229b7fac" name="ret_port" id="78ac7bde-62b4-4587-8a3a-6893966fbd79" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:InPort" connectedTo="935f3cbd-95b4-4d81-be5f-ca68315c9f4f 2ece3d04-caba-4f59-b658-b4053a04d550" name="ret_port" id="a455fc93-a668-42f6-a014-70f482fb3efc" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_21a0_ret" id="0b02007d-8de3-4d3b-9501-f9d572241fec">
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99416031552172" lon="4.369452758355907"/>
        <port xsi:type="esdl:OutPort" connectedTo="f30e1d44-20de-4860-832c-9cf230dcb4f3 9871687a-b3a8-43a8-9d7b-c039ce331ed6" name="ret_port" id="b91359da-6f9e-454c-bc50-59067a8a13b2" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:InPort" connectedTo="9fbee64a-535c-49b2-9022-a6e61361073d f1ea42b8-db7d-44e4-b451-654b8394cba2" name="ret_port" id="ebb8ee4a-4d65-41ea-8899-cc7f95491bb0" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="843.42" innerDiameter="0.1603" outerDiameter="0.25" name="Pipe1_ret" id="Pipe1_ret" related="Pipe1" diameter="DN150">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99914772322822" lon="4.366580576596763"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.00093116462207" lon="4.36553758519132"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.0027409547304" lon="4.365524500958683"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.003414653526114" lon="4.3700336825792006"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.00441857801509" lon="4.369716347128475"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="78ac7bde-62b4-4587-8a3a-6893966fbd79" name="In_ret" id="02ecbc14-ff04-4e0f-86b0-346f229b7fac" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:OutPort" connectedTo="2b4fdbd3-6904-4e60-b08e-9bcf6448e265" name="Out_ret" id="3ca327dc-8876-4b78-be0c-206a0eafe345" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="588.97" innerDiameter="0.3938" outerDiameter="0.56" name="Pipe2_ret" id="Pipe2_ret" related="Pipe2" diameter="DN400">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99416031552172" lon="4.369452758355907"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99914772322822" lon="4.366580576596763"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="b91359da-6f9e-454c-bc50-59067a8a13b2" name="In_ret" id="f30e1d44-20de-4860-832c-9cf230dcb4f3" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:OutPort" connectedTo="a455fc93-a668-42f6-a014-70f482fb3efc" name="Out_ret" id="2ece3d04-caba-4f59-b658-b4053a04d550" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="734.05" innerDiameter="0.2101" outerDiameter="0.315" name="Pipe3_ret" id="Pipe3_ret" related="Pipe3" diameter="DN200">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.9900841009021" lon="4.3751606570700075"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.98947627049756" lon="4.37318327752516"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99254176580451" lon="4.370367114375066"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99416031552172" lon="4.369452758355907"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="b3c6c264-b0e9-47f7-88ff-d18ed934ef3e" name="In_ret" id="cae6d1fb-7402-4e24-8870-c7d1c4c86a8e" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:OutPort" connectedTo="ebb8ee4a-4d65-41ea-8899-cc7f95491bb0" name="Out_ret" id="9fbee64a-535c-49b2-9022-a6e61361073d" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="314.5" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe4_ret" id="Pipe4_ret" related="Pipe4" diameter="DN300">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.994708627260025" lon="4.373961650608693"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99416031552172" lon="4.369452758355907"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="906f06cf-0b5a-4d32-9c89-2483b0880f73" name="In_ret" id="4d4ea4f1-04fb-4d25-8aaf-9e1223651f81" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:OutPort" connectedTo="ebb8ee4a-4d65-41ea-8899-cc7f95491bb0" name="Out_ret" id="f1ea42b8-db7d-44e4-b451-654b8394cba2" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" length="386.9" innerDiameter="0.3127" outerDiameter="0.45" name="Pipe5_ret" id="Pipe5_ret" related="Pipe5" diameter="DN300">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99416031552172" lon="4.369452758355907"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.994510443069274" lon="4.363832626387745"/>
        </geometry>
        <port xsi:type="esdl:InPort" connectedTo="b91359da-6f9e-454c-bc50-59067a8a13b2" name="In_ret" id="9871687a-b3a8-43a8-9d7b-c039ce331ed6" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
        <port xsi:type="esdl:OutPort" connectedTo="8d27a37a-082e-49af-923d-b878fd19cf3f" name="Out_ret" id="62bf94ae-b3aa-4084-95ff-b052fe182df1" carrier="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret"/>
      </asset>
    </area>
  </instance>
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="587aaec6-59e1-42a3-9d05-fda822e823c7">
    <carriers xsi:type="esdl:Carriers" id="6072772c-ea2b-4161-a0f6-cdaf0319f3c2">
      <carrier xsi:type="esdl:HeatCommodity" id="9e201a90-71c0-45bc-a683-6ae7e68b67d1" name="Primary" supplyTemperature="90.0"/>
      <carrier xsi:type="esdl:HeatCommodity" id="e32b730d-4c75-4610-8ef7-dfcf461cc069" name="Secondary" supplyTemperature="70.0"/>
      <carrier xsi:type="esdl:HeatCommodity" name="Primary_ret" id="9e201a90-71c0-45bc-a683-6ae7e68b67d1_ret" returnTemperature="50.0"/>
      <carrier xsi:type="esdl:HeatCommodity" name="Secondary_ret" id="e32b730d-4c75-4610-8ef7-dfcf461cc069_ret" returnTemperature="40.0"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="013d8f95-ae1c-448b-83eb-136a19150707">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Power in MW" unit="WATT" physicalQuantity="POWER" multiplier="MEGA" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
    </quantityAndUnits>
  </energySystemInformation>
</esdl:EnergySystem>
