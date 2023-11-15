<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" version="5" id="efae9b58-8902-48cc-b223-3dd8b974a887_with_return_network" description="" esdlVersion="v2210" name="test_simple with return network">
  <instance xsi:type="esdl:Instance" id="4948fb33-ade0-4772-af3c-76a7a53cfb1b" name="Untitled instance">
    <area xsi:type="esdl:Area" name="test_simple" id="58cc169e-e080-4f29-968a-fcdc1b535005">
      <asset xsi:type="esdl:Pipe" related="Pipe1_ret" id="Pipe1" state="OPTIONAL" outerDiameter="0.315" diameter="DN1200" innerDiameter="0.2101" name="Pipe1" length="1065.5">
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" name="In" connectedTo="bde1c4d2-08e9-4bfa-932c-756918f75eeb" id="e4d00ff5-f0ad-4708-a507-3b65ea688b3d"/>
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" name="Out" id="50472f9c-f398-4910-b285-10f0726c5a91" connectedTo="78fcb7b2-1c2c-4c0a-a45c-4b70ef366883"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" id="930aa5cf-b76e-4049-afa7-ea79445faf55" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.3017053604125985" lat="52.040534056400794"/>
          <point xsi:type="esdl:Point" lon="4.317283630371095" lat="52.040586851155766"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf" name="Logstor Product Catalogue Version 2020.03"/>
        <costInformation xsi:type="esdl:CostInformation" id="d2479332-5c6e-4ffb-8c53-d5612fe69ec5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" id="f09748bb-db39-48c2-a8bd-cef12778812a" power="50000000.0" name="ResidualHeatSource_1">
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" name="Out" id="bde1c4d2-08e9-4bfa-932c-756918f75eeb" connectedTo="e4d00ff5-f0ad-4708-a507-3b65ea688b3d"/>
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" name="In" connectedTo="e9bf7ac3-8188-4c9d-b428-6cf352bd13d5" id="8c75a72c-a0aa-4bc0-8780-f6a9f91209dd"/>
        <geometry xsi:type="esdl:Point" lon="4.30016040802002" lat="52.04048126158346" CRS="WGS84"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" id="3ff29919-9b95-4f79-881a-554698744ffc" power="30000000.0" name="HeatingDemand_1">
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" name="In" connectedTo="50472f9c-f398-4910-b285-10f0726c5a91" id="78fcb7b2-1c2c-4c0a-a45c-4b70ef366883"/>
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" name="Out" id="a8efb7f4-529f-4d52-ad06-2709e5a400ad" connectedTo="56e88d83-a31b-494e-bb4f-06e90f948930"/>
        <geometry xsi:type="esdl:Point" lon="4.319128990173341" lat="52.04037567176172" CRS="WGS84"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1" id="Pipe1_ret" state="OPTIONAL" outerDiameter="0.315" diameter="DN1200" innerDiameter="0.2101" name="Pipe1_ret" length="1065.5">
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" name="In_ret" connectedTo="a8efb7f4-529f-4d52-ad06-2709e5a400ad" id="56e88d83-a31b-494e-bb4f-06e90f948930"/>
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" name="Out_ret" id="e9bf7ac3-8188-4c9d-b428-6cf352bd13d5" connectedTo="8c75a72c-a0aa-4bc0-8780-f6a9f91209dd"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lon="4.316656392279453" lat="52.04067685124576" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lon="4.3010779624577795" lat="52.04062405649079" CRS="WGS84"/>
        </geometry>
      </asset>
    </area>
  </instance>
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="5b94e081-1321-411a-a60c-626cee580861">
    <carriers xsi:type="esdl:Carriers" id="baa6a6c9-be81-4b43-ae48-3806d6d63011">
      <carrier xsi:type="esdl:HeatCommodity" id="08bc1f84-4a92-4966-8264-389231f57372" supplyTemperature="80.0" name="heat"/>
      <carrier xsi:type="esdl:HeatCommodity" id="08bc1f84-4a92-4966-8264-389231f57372_ret" returnTemperature="40.0" name="heat_ret"/>
    </carriers>
  </energySystemInformation>
</esdl:EnergySystem>
