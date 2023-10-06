<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" version="21" id="15174819-d1af-4ba6-9f1d-2cd07991f14a_with_return_network" description="" esdlVersion="v2207" name="PoC Tutorial_2sources_storage with return network">
  <instance xsi:type="esdl:Instance" id="9af52a01-24e2-4bbe-a7b8-f868161e6ca3" name="Untitled instance">
    <area xsi:type="esdl:Area" id="9edf5aa3-3855-4dac-b6f5-01c2edf1a43a" name="Untitled area">
      <area xsi:type="esdl:Area" name="Area_76a7" id="76a75122-8ea6-472d-8059-374830275d72">
        <asset xsi:type="esdl:HeatingDemand" id="b0ff0df6-4a47-43a5-a0a5-aa10975c0a5c" power="25000000.0" name="HeatingDemand_b0ff">
          <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="5169316d-ae93-4f04-9a34-7c776444b651" id="2c5a109b-0d98-47b6-acc1-05e1708f8b85">
            <profile xsi:type="esdl:InfluxDBProfile" startDate="2018-12-31T23:00:00.000000+0000" database="energy_profiles" host="profiles.warmingup.info" filters="" port="443" multiplier="8.0" endDate="2019-12-31T22:00:00.000000+0000" field="demand2_MW" measurement="WarmingUp default profiles" id="c4062edd-779b-44b9-87de-02ea9c1babfe">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" connectedTo="3585453d-8c3a-40ce-873c-22a1b2403a6a" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out" id="d7eb3623-7481-41bb-911d-2dd9bd67db39"/>
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.00105253065436" lon="4.373545646667481"/>
          <costInformation xsi:type="esdl:CostInformation" id="cd676c1c-6224-406f-a864-3e7ff55265e0">
            <installationCosts xsi:type="esdl:SingleValue" value="100000.0" id="829c1c62-4266-4560-ac77-b961b7a722cf">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="1c0a9d42-bb5a-44c1-9ed6-824d7c82e932" unit="EURO" physicalQuantity="COST"/>
            </installationCosts>
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="08f06cd1-471b-404a-8894-27b315775b59">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="f8e5bc00-6eae-47d6-b7ad-b9699879b14c" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="52.003232185574326" lon="4.376142024993897"/>
            <point xsi:type="esdl:Point" lat="52.002228236493224" lon="4.372107982635499"/>
            <point xsi:type="esdl:Point" lat="52.0031132980428" lon="4.3712711334228525"/>
            <point xsi:type="esdl:Point" lat="52.00197724570447" lon="4.366979598999024"/>
            <point xsi:type="esdl:Point" lat="51.99826505971136" lon="4.369254112243653"/>
            <point xsi:type="esdl:Point" lat="51.99945404958839" lon="4.373760223388673"/>
            <point xsi:type="esdl:Point" lat="51.995635958872874" lon="4.3776869773864755"/>
            <point xsi:type="esdl:Point" lat="51.9963626102222" lon="4.380712509155274"/>
          </exterior>
        </geometry>
      </area>
      <area xsi:type="esdl:Area" name="Area_9d0f" id="9d0fd9ed-662c-42ec-a417-7e789737084c">
        <asset xsi:type="esdl:HeatingDemand" id="08fd3385-681a-4211-a083-51775cc99daa" power="25000000.0" name="HeatingDemand_08fd">
          <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="787c6ae3-96da-41e1-af86-6e68a1e28cb1" id="01caa60f-1549-4f3f-817e-e4e6807b2398">
            <profile xsi:type="esdl:InfluxDBProfile" startDate="2018-12-31T23:00:00.000000+0000" database="energy_profiles" host="profiles.warmingup.info" filters="" port="443" multiplier="8.0" endDate="2019-12-31T22:00:00.000000+0000" field="demand4_MW" measurement="WarmingUp default profiles" id="a5728de4-65d9-4f91-9a77-0f13bf0cf240">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" connectedTo="244de5c4-7403-45c9-a983-3697afc4ed8d" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out" id="ca90e0a5-f866-4ec2-9b3b-07f054f1c2b2"/>
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99648151568376" lon="4.373245239257813"/>
          <costInformation xsi:type="esdl:CostInformation" id="7eac6b5e-ba11-4ef1-b90e-599dd2690d20">
            <installationCosts xsi:type="esdl:SingleValue" value="100000.0" id="3b93383a-2ec4-4b35-9335-ebd91f0de600">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="318840e0-6f48-46f4-b9c9-91315d7b45fd" unit="EURO" physicalQuantity="COST"/>
            </installationCosts>
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="26aba0b6-a1a0-4610-97b3-be763548c669">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="424508c6-24aa-4a35-81e9-7ae198cbaeb8" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="51.99817258139733" lon="4.3692970275878915"/>
            <point xsi:type="esdl:Point" lat="51.99932194116153" lon="4.373695850372315"/>
            <point xsi:type="esdl:Point" lat="51.99534529503086" lon="4.376678466796876"/>
            <point xsi:type="esdl:Point" lat="51.99624370444482" lon="4.38081979751587"/>
            <point xsi:type="esdl:Point" lat="51.99490929572965" lon="4.381613731384278"/>
            <point xsi:type="esdl:Point" lat="51.99272923553559" lon="4.372708797454835"/>
          </exterior>
        </geometry>
      </area>
      <area xsi:type="esdl:Area" name="Area_a58a" id="a58a988c-95c1-492b-bc63-b125d25f5108">
        <asset xsi:type="esdl:HeatingDemand" id="8fbe3d4e-5d5b-4489-9271-9969c2b9e589" power="25000000.0" name="HeatingDemand_8fbe">
          <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="13edd58c-4a04-4770-8aac-c6e8689acbba" id="23cdb929-5cfc-4b8d-963e-06b6e6cf3a5c">
            <profile xsi:type="esdl:InfluxDBProfile" startDate="2018-12-31T23:00:00.000000+0000" database="energy_profiles" host="profiles.warmingup.info" filters="" port="443" multiplier="10.0" endDate="2019-12-31T22:00:00.000000+0000" field="demand5_MW" measurement="WarmingUp default profiles" id="c2e1824f-34be-4dbb-9a6d-c4bb64ff3749">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
            </profile>
          </port>
          <port xsi:type="esdl:OutPort" connectedTo="976ff0fd-5c99-4044-a91e-9879ce306b5a" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out" id="0fd050fa-15ff-4f1e-b0bd-ef1823365eaa"/>
          <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99069441691871" lon="4.379038810729981"/>
          <costInformation xsi:type="esdl:CostInformation" id="32e59a3e-fe5e-4201-bbd6-4f26dc4c961e">
            <installationCosts xsi:type="esdl:SingleValue" value="100000.0" id="850701a9-ec9a-40c5-ab3e-e0dbe6165cb6">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="b18cdc99-6303-404c-958a-1a4c079f842d" unit="EURO" physicalQuantity="COST"/>
            </installationCosts>
            <investmentCosts xsi:type="esdl:SingleValue" value="100000.0" id="3971c64d-50e4-47ec-a769-1f91935065d9">
              <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="5a0e43e1-b141-4626-87eb-487cdb70e60e" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
            </investmentCosts>
          </costInformation>
        </asset>
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lat="51.992279997820766" lon="4.372987747192384"/>
            <point xsi:type="esdl:Point" lat="51.994235470276436" lon="4.382214546203614"/>
            <point xsi:type="esdl:Point" lat="51.989333415199866" lon="4.38581943511963"/>
            <point xsi:type="esdl:Point" lat="51.98688879367896" lon="4.37633514404297"/>
          </exterior>
        </geometry>
      </area>
      <asset xsi:type="esdl:ResidualHeatSource" id="72d74fb5-134f-4bfb-829e-220ab76a8a7b" state="OPTIONAL" power="50000000.0" name="ResidualHeatSource_72d7">
        <port xsi:type="esdl:OutPort" connectedTo="07c2f88e-85be-4b8b-a72e-14364a3810c9" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="d890f8aa-9b94-493d-b017-bd7cebaf8c77"/>
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In" connectedTo="e78131b7-4501-4115-940e-ccc76617bccb" id="4d6c18cd-cc52-443d-8e53-96cd188dd1a8"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.00467202060717" lon="4.372987747192384"/>
        <costInformation xsi:type="esdl:CostInformation" id="123cc46e-f8ef-42f9-842f-08e77dd06899">
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="120000.0" id="8ee43904-745d-4f5c-8b15-5a2adcde7e5f">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="4c3a445d-d13f-4f61-8d75-22280135830f" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
          </fixedOperationalCosts>
          <variableOperationalCosts xsi:type="esdl:SingleValue" value="40.0" id="e5bc1b5e-23b1-4985-94f2-bea25c12da14">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" unit="EURO" id="de56cc5d-2baa-451c-b38c-47abea1393c2" description="Cost in EUR/MWh" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalCosts>
          <marginalCosts xsi:type="esdl:SingleValue" value="2.0" id="e52de449-7340-4878-b582-d20711d9fb14">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="4fccbb6b-2e41-46d9-a92c-f94da7c3aca7" unit="EURO" physicalQuantity="COST"/>
          </marginalCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="500000.0" id="3102215e-50a4-4332-b16e-2fd6522df31b">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="69673880-fe81-4f34-8fd9-558cc693a251" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="12c24715-08dc-4b61-ae15-76541704c83a">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="89b04c83-eb32-49a3-adc4-492a719221d3" unit="EURO" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe1_1_ret" id="Pipe1_1" state="OPTIONAL" outerDiameter="0.56" diameter="DN400" name="Pipe1_1" length="818.07">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="d890f8aa-9b94-493d-b017-bd7cebaf8c77" id="07c2f88e-85be-4b8b-a72e-14364a3810c9"/>
        <port xsi:type="esdl:OutPort" connectedTo="d149871b-d76a-4cc3-8922-c8d99205f47e" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="fc2801d0-215b-4d2b-9846-ee4918b87e21"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.00467202060717" lon="4.372987747192384"/>
          <point xsi:type="esdl:Point" lat="52.00210934629504" lon="4.365863800048829"/>
          <point xsi:type="esdl:Point" lat="51.99996926872789" lon="4.367129802703858"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4a3d48d9-74bb-4407-b249-9e2af4d1037c">
          <investmentCosts xsi:type="esdl:SingleValue" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="9169bd50-197f-4d6b-aaac-b383a59c815d" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" id="a04cb927-426d-4016-a084-356002a85d6c" name="Joint_a04c">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="fc2801d0-215b-4d2b-9846-ee4918b87e21" id="d149871b-d76a-4cc3-8922-c8d99205f47e"/>
        <port xsi:type="esdl:OutPort" connectedTo="863d2ff2-7a68-4bfb-8aa5-dab741b72b25 7f18f0b7-fcf3-4d83-8a65-79cbd3273ea7" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="3535f436-1270-4b1b-a326-41d69cd6e330"/>
        <geometry xsi:type="esdl:Point" lat="51.99996926872789" lon="4.367129802703858"/>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe3_1_ret" id="Pipe3_1" outerDiameter="0.56" diameter="DN400" name="Pipe3_1" length="1182.23">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="a42acf83-361d-4d4d-8001-8617daad939a" id="e44060e9-8b0e-41e5-ae46-ce074c891c15"/>
        <port xsi:type="esdl:OutPort" connectedTo="23cdb929-5cfc-4b8d-963e-06b6e6cf3a5c" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="13edd58c-4a04-4770-8aac-c6e8689acbba"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.995622746921015" lon="4.36983346939087"/>
          <point xsi:type="esdl:Point" lat="51.989029496545015" lon="4.373888969421388"/>
          <point xsi:type="esdl:Point" lat="51.99069441691871" lon="4.379038810729981"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4a3d48d9-74bb-4407-b249-9e2af4d1037c">
          <investmentCosts xsi:type="esdl:SingleValue" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="9169bd50-197f-4d6b-aaac-b383a59c815d" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" id="95caf7c7-e89f-4378-96f2-f904f9255c83" name="Joint_95ca">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="8c38fc93-ed85-42ef-9be6-87d47c416e90 abfddfae-c090-4b8a-88b1-23d4c6adacc5" id="4d629e6e-5db4-4a8f-9945-934a24ede11a"/>
        <port xsi:type="esdl:OutPort" connectedTo="e44060e9-8b0e-41e5-ae46-ce074c891c15 20165ec3-cf86-41e4-976f-079ba0ca5be4" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="a42acf83-361d-4d4d-8001-8617daad939a"/>
        <geometry xsi:type="esdl:Point" lat="51.995622746921015" lon="4.36983346939087"/>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3127" related="Pipe4_1_ret" id="Pipe4_1" outerDiameter="0.45" diameter="DN300" name="Pipe4_1" length="455.4">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="3535f436-1270-4b1b-a326-41d69cd6e330" id="7f18f0b7-fcf3-4d83-8a65-79cbd3273ea7"/>
        <port xsi:type="esdl:OutPort" connectedTo="2c5a109b-0d98-47b6-acc1-05e1708f8b85" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="5169316d-ae93-4f04-9a34-7c776444b651"/>
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
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.99996926872789" lon="4.367129802703858"/>
          <point xsi:type="esdl:Point" lat="52.00105253065436" lon="4.373545646667481"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="384df56d-e3f0-4794-a0e2-bc0c71482e45">
          <investmentCosts xsi:type="esdl:SingleValue" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3127" related="Pipe5_1_ret" id="Pipe5_1" outerDiameter="0.45" diameter="DN300" name="Pipe5_1" length="252.3">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="a42acf83-361d-4d4d-8001-8617daad939a" id="20165ec3-cf86-41e4-976f-079ba0ca5be4"/>
        <port xsi:type="esdl:OutPort" connectedTo="01caa60f-1549-4f3f-817e-e4e6807b2398" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="787c6ae3-96da-41e1-af86-6e68a1e28cb1"/>
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
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.995622746921015" lon="4.36983346939087"/>
          <point xsi:type="esdl:Point" lat="51.99648151568376" lon="4.373245239257813"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4c8d9c8f-7e99-4402-8f66-d413e73db121">
          <investmentCosts xsi:type="esdl:SingleValue" value="1962.1" id="1e93bdda-8a74-42d5-960d-d64e4dff2025" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="983f0959-8566-43ce-a380-782d29406ed3" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" id="76f0817c-9f01-431e-be96-dbf3ee806d76" state="OPTIONAL" power="50000000.0" name="ResidualHeatSource_76f0">
        <port xsi:type="esdl:OutPort" connectedTo="74f2c723-08f3-4d44-871b-875aa721e055" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="81fbe527-8919-4864-b3fa-d3dbc223e4bb"/>
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In" connectedTo="9b7f473c-e432-44b5-9273-c50128b48218" id="4d708ad1-b369-47a3-abd3-26aa96173e71"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.995081053536964" lon="4.364705085754395"/>
        <costInformation xsi:type="esdl:CostInformation" id="8a3abfe2-d012-489d-87c3-2732a6d4b25f">
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="60000.0" id="b1b80150-e761-4633-b9d5-81b1d69a6db8">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="2c63c99b-cc4c-471a-a83e-a0d25cad5b7e" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
          </fixedOperationalCosts>
          <variableOperationalCosts xsi:type="esdl:SingleValue" value="20.0" id="233a17e9-7392-49fb-a833-f8761c322098">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATTHOUR" unit="EURO" id="6fad95f6-2d6b-4310-9f76-9b3a26b3bd40" description="Cost in EUR/MWh" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalCosts>
          <marginalCosts xsi:type="esdl:SingleValue" value="1.0" id="c36fe934-9dc9-4ef1-bd6b-6e1e4168ae91">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="4cfb1695-3026-49fa-a8ba-2f90450196d4" unit="EURO" physicalQuantity="COST"/>
          </marginalCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="250000.0" id="a572009d-83d3-48ba-8727-e41267ca5350">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="6862d724-1794-4bab-b666-258fd14150ac" description="Cost in EUR/MW" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="938b0cb2-d1e5-4fcc-bf45-6de3b5b06de2">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="44294593-d3b9-4940-9b58-699ccbb28b86" unit="EURO" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.263" related="Pipe1_ret" id="Pipe1" state="OPTIONAL" outerDiameter="0.4" diameter="DN250" name="Pipe1" length="356.2">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="81fbe527-8919-4864-b3fa-d3dbc223e4bb" id="74f2c723-08f3-4d44-871b-875aa721e055"/>
        <port xsi:type="esdl:OutPort" connectedTo="4d629e6e-5db4-4a8f-9945-934a24ede11a" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="abfddfae-c090-4b8a-88b1-23d4c6adacc5"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.005">
            <matter xsi:type="esdl:Material" id="faac539b-4b7c-43f8-abcd-f08fa2652b7b" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0587">
            <matter xsi:type="esdl:Material" id="d23b4eeb-a419-4c16-bc7e-280a76116f04" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0048">
            <matter xsi:type="esdl:Material" id="a2b91e8d-471d-4276-a8f6-4efb01054b4e" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.995081053536964" lon="4.364705085754395"/>
          <point xsi:type="esdl:Point" lat="51.995622746921015" lon="4.36983346939087"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="91120115-4997-441c-8732-827100c28f9b">
          <investmentCosts xsi:type="esdl:SingleValue" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="9169bd50-197f-4d6b-aaac-b383a59c815d" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:ATES" aquiferMidTemperature="17.0" aquiferPermeability="10000.0" id="94180b94-ab68-4b63-93a4-7fd53377fc39" wellCasingSize="13.0" aquiferTopDepth="300.0" aquiferNetToGross="1.0" maxDischargeRate="11610000.0" aquiferAnisotropy="4.0" name="ATES_9418" wellDistance="150.0" aquiferThickness="45.0" maxChargeRate="11610000.0" aquiferPorosity="0.3" salinity="10000.0">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="a087d17b-06b9-430d-a21e-4429c5ee8eb3" id="d8f59680-68e3-4c8c-af03-b6b93485bdea"/>
        <port xsi:type="esdl:OutPort" connectedTo="60f698fd-72a7-4fd8-9a7c-25830035a2ef" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out" id="a323b0d8-f467-4d65-b06e-a3e9c31cf7ee"/>
        <dataSource xsi:type="esdl:DataSource" name="WarmingUp factsheet: HT-ATES (high)" description="This data was generated using the 'kosten_per_asset.xslx' file in the 'Kentallen' directory of WarmingUp project 1D" attribution=""/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.996190857331285" lon="4.3641042709350595"/>
        <costInformation xsi:type="esdl:CostInformation">
          <fixedOperationalCosts xsi:type="esdl:SingleValue" value="10000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="574ef21d-681a-43ae-a1cb-f7b25d88defb" description="Cost in EUR/MW" perMultiplier="MEGA"/>
          </fixedOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" value="1000000.0" id="b599f18b-f06c-4176-82a6-01d54fa170f6">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR" id="d9663ce0-e713-4457-b0e2-0a898033105c" unit="EURO" physicalQuantity="COST"/>
          </installationCosts>
          <investmentCosts xsi:type="esdl:SingleValue" value="200000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="a3b5cdd9-364b-4262-bce5-4658c5f1bac9" description="Cost in EUR/MW" perMultiplier="MEGA"/>
          </investmentCosts>
          <fixedMaintenanceCosts xsi:type="esdl:SingleValue" value="10000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="WATT" unit="EURO" id="57537388-7fd7-40b3-a0c4-0ce65648eaab" description="Cost in EUR/MW" perMultiplier="MEGA"/>
          </fixedMaintenanceCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe2_ret" id="Pipe2" outerDiameter="0.56" diameter="DN400" name="Pipe2" length="371.04">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="3535f436-1270-4b1b-a326-41d69cd6e330" id="863d2ff2-7a68-4bfb-8aa5-dab741b72b25"/>
        <port xsi:type="esdl:OutPort" connectedTo="61955a8c-5e5d-4c18-8600-6c48fd2bfde6" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="c176f2c7-21a6-4257-842a-6dbd4d3b1216"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.99996926872789" lon="4.367129802703858"/>
          <point xsi:type="esdl:Point" lat="51.9968514417669" lon="4.369060993194581"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4a3d48d9-74bb-4407-b249-9e2af4d1037c">
          <investmentCosts xsi:type="esdl:SingleValue" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="9169bd50-197f-4d6b-aaac-b383a59c815d" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe3_ret" id="Pipe3" outerDiameter="0.56" diameter="DN400" name="Pipe3" length="146.5">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="f244349d-052f-45eb-82b9-013845d88213" id="83073b33-31e1-4a20-a005-b54a745d82d4"/>
        <port xsi:type="esdl:OutPort" connectedTo="4d629e6e-5db4-4a8f-9945-934a24ede11a" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="8c38fc93-ed85-42ef-9be6-87d47c416e90"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="51.9968514417669" lon="4.369060993194581"/>
          <point xsi:type="esdl:Point" lat="51.995622746921015" lon="4.36983346939087"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4a3d48d9-74bb-4407-b249-9e2af4d1037c">
          <investmentCosts xsi:type="esdl:SingleValue" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="9169bd50-197f-4d6b-aaac-b383a59c815d" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" id="54362e09-d7ae-4d79-9d1a-e1e796efb407" name="Joint_5436">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="c176f2c7-21a6-4257-842a-6dbd4d3b1216" id="61955a8c-5e5d-4c18-8600-6c48fd2bfde6"/>
        <port xsi:type="esdl:OutPort" connectedTo="83073b33-31e1-4a20-a005-b54a745d82d4 58458118-b835-4e8d-932a-c75e1b3ec7ec" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="f244349d-052f-45eb-82b9-013845d88213"/>
        <geometry xsi:type="esdl:Point" lat="51.9968514417669" lon="4.369060993194581"/>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe4_ret" id="Pipe4" state="OPTIONAL" outerDiameter="0.56" diameter="DN400" name="Pipe4" length="347.2">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="In" connectedTo="f244349d-052f-45eb-82b9-013845d88213" id="58458118-b835-4e8d-932a-c75e1b3ec7ec"/>
        <port xsi:type="esdl:OutPort" connectedTo="d8f59680-68e3-4c8c-af03-b6b93485bdea" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" name="Out" id="a087d17b-06b9-430d-a21e-4429c5ee8eb3"/>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" id="74d84321-1767-4cec-b6e7-b90c06020400" thermalConductivity="52.15" name="steel"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" id="95012cd9-7648-4df5-8c83-74dc73bc16ba" thermalConductivity="0.027" name="PUR"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be" thermalConductivity="0.4" name="HDPE"/>
          </component>
        </material>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="51.9968514417669" lon="4.369060993194581"/>
          <point xsi:type="esdl:Point" lat="51.996190857331285" lon="4.3641042709350595"/>
        </geometry>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="31686671-e1b7-4b4e-852c-1ba30e572636">
          <investmentCosts xsi:type="esdl:SingleValue" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86" name="Combined investment and installation costs">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" perUnit="METRE" unit="EURO" id="9169bd50-197f-4d6b-aaac-b383a59c815d" description="Costs in EUR/m" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" id="46b7c740-70de-40b1-a208-f656005bc703" name="Joint_a04c_ret">
        <port xsi:type="esdl:OutPort" connectedTo="31b83f9c-84a5-4993-866c-981a8e935a32" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="ret_port" id="3ca34f19-f97b-4524-99c1-d28d1ad218fe"/>
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="ret_port" connectedTo="e8df3fb4-0462-411a-a65f-1a72c770dfa9 38b6ee76-57a9-4601-b4dd-5f96e5824af4" id="849606dd-d795-411b-a19b-5b2f754dcd90"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="52.000059268817886" lon="4.3663488762941665"/>
      </asset>
      <asset xsi:type="esdl:Joint" id="7bf5e908-9681-4550-ba31-8a7f5da545de" name="Joint_95ca_ret">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="ret_port" connectedTo="bade7369-f079-4fdd-acae-a8a4707665a8 745a16cd-276c-4dd1-a827-a38254ffd76b" id="41d3d82c-ce6f-47c8-bd56-39903a177f31"/>
        <port xsi:type="esdl:OutPort" connectedTo="52480ed1-7b38-4aea-9848-0fa284c764d3 b566e44c-2d5e-4c91-83a0-7ce39143219e" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="ret_port" id="cd56a05e-59e9-4141-b193-525596370c92"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.99571274701101" lon="4.369031429215318"/>
      </asset>
      <asset xsi:type="esdl:Joint" id="359a1db6-3c95-4812-a672-01046b07dc8c" name="Joint_5436_ret">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="ret_port" connectedTo="25607d6d-e06d-44c3-9351-89ece62a0880 7a470258-76e6-4c4f-a054-62345806c3ef" id="95717d03-9439-4cb3-a5bc-b23375396103"/>
        <port xsi:type="esdl:OutPort" connectedTo="9c1dc996-4740-4839-8a55-e10d954502b3" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="ret_port" id="9e61ca18-25ad-4af2-a1db-6fefd3816146"/>
        <geometry xsi:type="esdl:Point" CRS="WGS84" lat="51.9969414418569" lon="4.36826503796172"/>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe1_1" id="Pipe1_1_ret" state="OPTIONAL" outerDiameter="0.56" diameter="DN400" name="Pipe1_1_ret" length="818.07">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="3ca34f19-f97b-4524-99c1-d28d1ad218fe" id="31b83f9c-84a5-4993-866c-981a8e935a32"/>
        <port xsi:type="esdl:OutPort" connectedTo="4d6c18cd-cc52-443d-8e53-96cd188dd1a8" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="e78131b7-4501-4115-940e-ccc76617bccb"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.000059268817886" lon="4.3663488762941665"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.00219934638504" lon="4.365092860885141"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.004762020697164" lon="4.372228431895539"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe3_1" id="Pipe3_1_ret" outerDiameter="0.56" diameter="DN400" name="Pipe3_1_ret" length="1182.23">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="0fd050fa-15ff-4f1e-b0bd-ef1823365eaa" id="976ff0fd-5c99-4044-a91e-9879ce306b5a"/>
        <port xsi:type="esdl:OutPort" connectedTo="41d3d82c-ce6f-47c8-bd56-39903a177f31" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="bade7369-f079-4fdd-acae-a8a4707665a8"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.990784417008705" lon="4.378211386257638"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.98911949663501" lon="4.37305259776345"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99571274701101" lon="4.369031429215318"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3127" related="Pipe4_1" id="Pipe4_1_ret" outerDiameter="0.45" diameter="DN300" name="Pipe4_1_ret" length="455.4">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="d7eb3623-7481-41bb-911d-2dd9bd67db39" id="3585453d-8c3a-40ce-873c-22a1b2403a6a"/>
        <port xsi:type="esdl:OutPort" connectedTo="849606dd-d795-411b-a19b-5b2f754dcd90" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="e8df3fb4-0462-411a-a65f-1a72c770dfa9"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.00114253074436" lon="4.372769808163894"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.000059268817886" lon="4.3663488762941665"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3127" related="Pipe5_1" id="Pipe5_1_ret" outerDiameter="0.45" diameter="DN300" name="Pipe5_1_ret" length="252.3">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="ca90e0a5-f866-4ec2-9b3b-07f054f1c2b2" id="244de5c4-7403-45c9-a983-3697afc4ed8d"/>
        <port xsi:type="esdl:OutPort" connectedTo="41d3d82c-ce6f-47c8-bd56-39903a177f31" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="745a16cd-276c-4dd1-a827-a38254ffd76b"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99657151577376" lon="4.372447461880088"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99571274701101" lon="4.369031429215318"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.263" related="Pipe1" id="Pipe1_ret" state="OPTIONAL" outerDiameter="0.4" diameter="DN250" name="Pipe1_ret" length="356.2">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="cd56a05e-59e9-4141-b193-525596370c92" id="52480ed1-7b38-4aea-9848-0fa284c764d3"/>
        <port xsi:type="esdl:OutPort" connectedTo="4d708ad1-b369-47a3-abd3-26aa96173e71" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="9b7f473c-e432-44b5-9273-c50128b48218"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99571274701101" lon="4.369031429215318"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99517105362696" lon="4.363900332878933"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe2" id="Pipe2_ret" outerDiameter="0.56" diameter="DN400" name="Pipe2_ret" length="371.04">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="9e61ca18-25ad-4af2-a1db-6fefd3816146" id="9c1dc996-4740-4839-8a55-e10d954502b3"/>
        <port xsi:type="esdl:OutPort" connectedTo="849606dd-d795-411b-a19b-5b2f754dcd90" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="38b6ee76-57a9-4601-b4dd-5f96e5824af4"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.9969414418569" lon="4.36826503796172"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="52.000059268817886" lon="4.3663488762941665"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe3" id="Pipe3_ret" outerDiameter="0.56" diameter="DN400" name="Pipe3_ret" length="146.5">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="cd56a05e-59e9-4141-b193-525596370c92" id="b566e44c-2d5e-4c91-83a0-7ce39143219e"/>
        <port xsi:type="esdl:OutPort" connectedTo="95717d03-9439-4cb3-a5bc-b23375396103" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="25607d6d-e06d-44c3-9351-89ece62a0880"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99571274701101" lon="4.369031429215318"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.9969414418569" lon="4.36826503796172"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" innerDiameter="0.3938" related="Pipe4" id="Pipe4_ret" state="OPTIONAL" outerDiameter="0.56" diameter="DN400" name="Pipe4_ret" length="347.2">
        <port xsi:type="esdl:InPort" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="In_ret" connectedTo="a323b0d8-f467-4d65-b06e-a3e9c31cf7ee" id="60f698fd-72a7-4fd8-9a7c-25830035a2ef"/>
        <port xsi:type="esdl:OutPort" connectedTo="95717d03-9439-4cb3-a5bc-b23375396103" carrier="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" name="Out_ret" id="7a470258-76e6-4c4f-a054-62345806c3ef"/>
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.99628085742128" lon="4.363305055921369"/>
          <point xsi:type="esdl:Point" CRS="WGS84" lat="51.9969414418569" lon="4.36826503796172"/>
        </geometry>
      </asset>
    </area>
  </instance>
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="5fa7e6e3-5701-48a7-bbbd-cad59a46f6c5">
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="38f7850a-2090-411e-a15a-84b1d7b78362">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" unit="WATT" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b" description="Power in MW" physicalQuantity="POWER" multiplier="MEGA"/>
    </quantityAndUnits>
    <carriers xsi:type="esdl:Carriers" id="42a694c6-9a1b-4cc7-bbae-0b44725f9434">
      <carrier xsi:type="esdl:HeatCommodity" id="9f6aeb1a-138b-4bb9-9a09-d524e94658e6" supplyTemperature="80.0" name="Primary"/>
      <carrier xsi:type="esdl:HeatCommodity" id="9f6aeb1a-138b-4bb9-9a09-d524e94658e6_ret" returnTemperature="40.0" name="Primary_ret"/>
    </carriers>
  </energySystemInformation>
</esdl:EnergySystem>
