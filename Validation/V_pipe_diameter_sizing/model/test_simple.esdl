<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" description="" esdlVersion="v2210" name="test_simple with return network" version="4" id="efae9b58-8902-48cc-b223-3dd8b974a887_with_return_network">
  <instance xsi:type="esdl:Instance" id="4948fb33-ade0-4772-af3c-76a7a53cfb1b" name="Untitled instance">
    <area xsi:type="esdl:Area" id="58cc169e-e080-4f29-968a-fcdc1b535005" name="test_simple">
      <asset xsi:type="esdl:Pipe" length="1065.5" innerDiameter="0.2101" diameter="DN200" outerDiameter="0.315" related="Pipe1_ret" name="Pipe1" id="Pipe1">
        <costInformation xsi:type="esdl:CostInformation" id="d2479332-5c6e-4ffb-8c53-d5612fe69ec5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" description="Costs in EUR/m" perUnit="METRE" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.3017053604125985" lat="52.040534056400794"/>
          <point xsi:type="esdl:Point" lon="4.317283630371095" lat="52.040586851155766"/>
        </geometry>
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" connectedTo="bde1c4d2-08e9-4bfa-932c-756918f75eeb" name="In" id="e4d00ff5-f0ad-4708-a507-3b65ea688b3d"/>
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" connectedTo="78fcb7b2-1c2c-4c0a-a45c-4b70ef366883" name="Out" id="50472f9c-f398-4910-b285-10f0726c5a91"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" id="930aa5cf-b76e-4049-afa7-ea79445faf55" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3" name="HDPE"/>
          </component>
        </material>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" name="ResidualHeatSource_1" id="f09748bb-db39-48c2-a8bd-cef12778812a" power="50000000.0">
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" connectedTo="e4d00ff5-f0ad-4708-a507-3b65ea688b3d" name="Out" id="bde1c4d2-08e9-4bfa-932c-756918f75eeb"/>
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" connectedTo="e9bf7ac3-8188-4c9d-b428-6cf352bd13d5" name="In" id="8c75a72c-a0aa-4bc0-8780-f6a9f91209dd"/>
        <geometry xsi:type="esdl:Point" lat="52.04048126158346" lon="4.30016040802002" CRS="WGS84"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" power="50000000.0" name="HeatingDemand_1" id="3ff29919-9b95-4f79-881a-554698744ffc">
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372" connectedTo="50472f9c-f398-4910-b285-10f0726c5a91" name="In" id="78fcb7b2-1c2c-4c0a-a45c-4b70ef366883"/>
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" connectedTo="56e88d83-a31b-494e-bb4f-06e90f948930" name="Out" id="a8efb7f4-529f-4d52-ad06-2709e5a400ad"/>
        <geometry xsi:type="esdl:Point" lat="52.04037567176172" lon="4.319128990173341" CRS="WGS84"/>
      </asset>
      <asset xsi:type="esdl:Pipe" id="Pipe1_ret" length="1065.5" outerDiameter="0.315" innerDiameter="0.2101" related="Pipe1" name="Pipe1_ret" diameter="DN200">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04067685124576" lon="4.316656392279453" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04062405649079" lon="4.3010779624577795" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" connectedTo="a8efb7f4-529f-4d52-ad06-2709e5a400ad" name="In_ret" id="56e88d83-a31b-494e-bb4f-06e90f948930"/>
        <port xsi:type="esdl:OutPort" carrier="08bc1f84-4a92-4966-8264-389231f57372_ret" connectedTo="8c75a72c-a0aa-4bc0-8780-f6a9f91209dd" name="Out_ret" id="e9bf7ac3-8188-4c9d-b428-6cf352bd13d5"/>
      </asset>
    </area>
  </instance>
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="5b94e081-1321-411a-a60c-626cee580861">
    <carriers xsi:type="esdl:Carriers" id="baa6a6c9-be81-4b43-ae48-3806d6d63011">
      <carrier xsi:type="esdl:HeatCommodity" id="08bc1f84-4a92-4966-8264-389231f57372" name="heat" supplyTemperature="80.0"/>
      <carrier xsi:type="esdl:HeatCommodity" returnTemperature="40.0" id="08bc1f84-4a92-4966-8264-389231f57372_ret" name="heat_ret"/>
    </carriers>
  </energySystemInformation>
</esdl:EnergySystem>
