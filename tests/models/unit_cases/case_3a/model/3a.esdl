<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem description="" id="2dc6d2a1-519b-4759-986a-f3d4007d8d67" name="3a" xmlns:esdl="http://www.tno.nl/esdl" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <energySystemInformation id="6a1c2e35-b383-4816-af26-14deb41e6d1d" xsi:type="esdl:EnergySystemInformation">
    <carriers id="3a4f662c-eb5e-4e08-8e02-bbf2607faf1c" xsi:type="esdl:Carriers">
      <carrier id="9ab126bd-7b66-4bed-82cc-4c04282fe859" name="Heat_cold" returnTemperature="40.0" xsi:type="esdl:HeatCommodity"/>
      <carrier id="419b5016-12c9-475a-b46e-9e474b60aa8f" name="Heat_hot" supplyTemperature="80.0" xsi:type="esdl:HeatCommodity"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="c7104449-b5eb-49c7-b064-a2a4b16ae6e4">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Power in MW" unit="WATT" physicalQuantity="POWER" multiplier="MEGA" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
    </quantityAndUnits>
  </energySystemInformation>
  <instance id="a6350368-fa00-4b9b-81b0-283d35b91de0" name="Untitled Instance" xsi:type="esdl:Instance">
    <area id="f280f1c3-8858-4336-906e-c7608b192bf6" name="Untitled Area" xsi:type="esdl:Area">
      <asset capacity="100000000000.0" id="4b0cd685-2219-4b02-ad4f-da3bc5453651" name="HeatStorage_4b0c" xsi:type="esdl:HeatStorage">
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="a7c97b65-6bdd-4a00-b9b7-6c8d214053bd" id="efad2169-5f95-4618-98ce-e89860e2897b" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="776e5a76-57a5-4c03-b361-409ccb548b88" id="db555a41-d15b-44fb-b994-f5004eb3bc85" name="out" xsi:type="esdl:OutPort"/>
        <costInformation id="a65059d4-48ab-47a6-a391-9a93134bbcfe" xsi:type="esdl:CostInformation">
          <variableMaintenanceCosts id="10a07c71-5d3b-43fd-a99d-d58acd0fe11e" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <investmentCosts id="b6a15da1-af36-4444-8134-20c97ae9c6cc" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedOperationalCosts id="7abef6d7-5eb9-448f-b842-69402037e68c" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedMaintenanceCosts id="a638e42a-fb7d-472d-bffb-f2e240295cc2" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <variableOperationalCosts id="f2e339d6-5c27-4293-aacd-e47457a673cc" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <installationCosts id="1466ee38-91b9-4245-a44e-83a3e5a86fba" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
        </costInformation>
        <geometry CRS="WGS84" lat="51.98322161118308" lon="4.383137226104737" xsi:type="esdl:Point"/>
      </asset>
      <asset flowRate="5.0" id="b702bda3-632c-43ff-9867-72cda41f442f" maxTemperature="80.0" minTemperature="80.0" name="GeothermalSource_b702" power="10000000.0" xsi:type="esdl:GeothermalSource">
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="7cec87b6-9928-4cfb-9b7b-fe7b51908911" id="f31879b6-efa1-4b2c-8740-80bebb9500a7" name="Out" xsi:type="esdl:OutPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="53542379-cead-4b7c-97f5-891e1985a8cc" id="02be741c-673b-4c30-8bad-dee320e26924" name="in" xsi:type="esdl:InPort"/>
        <costInformation id="1fd82474-556d-44ff-88a1-d5eda195b218" xsi:type="esdl:CostInformation">
          <variableMaintenanceCosts id="ecd886c4-566e-4419-a352-9d0e2ea1eb0f" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <investmentCosts id="3c4cf9fe-659a-417e-a38b-1fec58b903be" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedOperationalCosts id="8d2bde1f-708e-4e0f-a0eb-bf4e7d922f90" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedMaintenanceCosts id="e45de843-8f57-429e-a914-c12a4189aff0" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <variableOperationalCosts id="130deb1b-98aa-4215-9d28-675a4271cf67" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <installationCosts id="c684f250-61bc-4593-b480-bce9461dc572" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
        </costInformation>
        <geometry CRS="WGS84" lat="51.98491978527026" lon="4.3822574615478525" xsi:type="esdl:Point"/>
      </asset>
      <asset id="de7364b2-24e6-4ce3-84a3-d6870a9f93bf" innerDiameter="0.16030000000000003" length="202.6782401380455" name="Pipe_de73" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="c4af6ec7-a8da-4412-b4c4-62d3b5c8ecb8" name="steel" thermalConductivity="0.00014862917548188605" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="b74a8c0f-9c8a-4a02-8055-c77943b414c1" name="PUR" thermalConductivity="2.1603217928675567" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036000000000000003" xsi:type="esdl:CompoundMatterComponent">
            <matter id="aff626e3-66f1-470f-a86d-f1f6adb336cb" name="HDPE" thermalConductivity="0.011627406024262562" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="1a4a112f-dd25-4438-9741-9cdd34c23e73" id="750c4c71-06a5-4137-a540-ac49fd5c8e24" name="In" xsi:type="esdl:InPort"/>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="18b90d93-5005-45d0-b558-ece8994d39ae" id="0f61b1e9-cf22-403c-8788-a0b00963dc5f" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98443082345643" lon="4.385175704956056" xsi:type="esdl:Point"/>
          <point lat="51.9853294518694" lon="4.3877506256103525" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="2e4292aa-da7d-4e1a-a71f-d0058f06a5ae" innerDiameter="0.16030000000000003" length="191.59622393233496" name="Pipe_2e42" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="c4af6ec7-a8da-4412-b4c4-62d3b5c8ecb8" name="steel" thermalConductivity="0.00014862917548188605" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="b74a8c0f-9c8a-4a02-8055-c77943b414c1" name="PUR" thermalConductivity="2.1603217928675567" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036000000000000003" xsi:type="esdl:CompoundMatterComponent">
            <matter id="aff626e3-66f1-470f-a86d-f1f6adb336cb" name="HDPE" thermalConductivity="0.011627406024262562" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="1a4a112f-dd25-4438-9741-9cdd34c23e73" id="bff4b9c4-d3fc-475f-b57a-15006f7e2841" name="In" xsi:type="esdl:InPort"/>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="f5eb43f1-fb71-4d82-97cc-3fb0eb2c209d" id="618a8f33-60c9-4265-83e2-13916fc28e75" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.984166517550754" lon="4.385282993316651" xsi:type="esdl:Point"/>
          <point lat="51.98429867069855" lon="4.38807249069214" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="7fab434b-b488-43f2-826e-d9c46c0d1b98" innerDiameter="0.16030000000000003" length="192.56530875797927" name="Pipe_7fab" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="c4af6ec7-a8da-4412-b4c4-62d3b5c8ecb8" name="steel" thermalConductivity="0.00014862917548188605" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="b74a8c0f-9c8a-4a02-8055-c77943b414c1" name="PUR" thermalConductivity="2.1603217928675567" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036000000000000003" xsi:type="esdl:CompoundMatterComponent">
            <matter id="aff626e3-66f1-470f-a86d-f1f6adb336cb" name="HDPE" thermalConductivity="0.011627406024262562" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="1a4a112f-dd25-4438-9741-9cdd34c23e73" id="2a5d7b97-a173-4fcf-a317-669761abecde" name="In" xsi:type="esdl:InPort"/>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="889db852-2751-425f-ab8c-989ff878acf8" id="75895019-b7a1-45c4-8550-8f01bc3a6f2a" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98394185630461" lon="4.385218620300294" xsi:type="esdl:Point"/>
          <point lat="51.98346609935764" lon="4.387922286987306" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="e53abd80-80fa-416e-9950-94e149bcab8d" innerDiameter="0.16030000000000003" length="86.30055068000897" name="Pipe_e53a" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="c4af6ec7-a8da-4412-b4c4-62d3b5c8ecb8" name="steel" thermalConductivity="0.00014862917548188605" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="b74a8c0f-9c8a-4a02-8055-c77943b414c1" name="PUR" thermalConductivity="2.1603217928675567" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036000000000000003" xsi:type="esdl:CompoundMatterComponent">
            <matter id="aff626e3-66f1-470f-a86d-f1f6adb336cb" name="HDPE" thermalConductivity="0.011627406024262562" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="1a4a112f-dd25-4438-9741-9cdd34c23e73" id="99c1b5b6-f5e5-47dc-a0a8-a4b6ab877a4b" name="In" xsi:type="esdl:InPort"/>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="efad2169-5f95-4618-98ce-e89860e2897b" id="a7c97b65-6bdd-4a00-b9b7-6c8d214053bd" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.984034364013006" lon="4.384467601776124" xsi:type="esdl:Point"/>
          <point lat="51.98350574596288" lon="4.383544921875001" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="f1a51444-41aa-4b92-a3f5-05ba2d4de379" name="Joint_f1a5" xsi:type="esdl:Joint">
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="beb9f479-61b9-48d2-a492-850dedc2826e" id="4dc27c51-a337-402a-9fdb-376aa78b98bf" name="In" xsi:type="esdl:InPort"/>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="750c4c71-06a5-4137-a540-ac49fd5c8e24 bff4b9c4-d3fc-475f-b57a-15006f7e2841 2a5d7b97-a173-4fcf-a317-669761abecde 99c1b5b6-f5e5-47dc-a0a8-a4b6ab877a4b" id="1a4a112f-dd25-4438-9741-9cdd34c23e73" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" lat="51.9843184936371" lon="4.384735822677613" xsi:type="esdl:Point"/>
      </asset>
      <asset id="59de9666-e9f5-4526-a1eb-910f050d3056" name="Joint_59de" xsi:type="esdl:Joint">
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="2616e85e-2b1a-4ddc-8724-fc7de9173c02 e26af727-64fb-4b86-8b5e-00700334a2a1 613a36f5-a31f-498a-a33d-bfe3c9132ae3 5b87fbb9-44a9-41e1-9088-19561c4c5836" id="c43e1fb4-83c1-4830-afef-828db81160a7" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="b1476c2d-cb2c-4efe-b854-7435ee1789e8" id="06a60970-d7d0-4e0f-827c-b3accf4048ae" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" lat="51.98410374466893" lon="4.384773373603822" xsi:type="esdl:Point"/>
      </asset>
      <asset id="a3b88fb6-b4a7-4986-8233-32ca05a5df9f" minTemperature="70.0" name="HeatingDemand_a3b8" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="0f61b1e9-cf22-403c-8788-a0b00963dc5f" id="18b90d93-5005-45d0-b558-ece8994d39ae" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.3" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="5b25d760-cdb5-460e-bde2-283058a4b124" id="31ec92fd-18c8-4788-9d6d-b96fab55e6e5" name="out" xsi:type="esdl:OutPort"/>
        <KPIs id="da2ac53f-3c9c-46a8-a57c-dddb2d93d5a0" xsi:type="esdl:KPIs"/>
        <costInformation id="a6bf5ed3-d145-40e7-b058-98047b1baf65" xsi:type="esdl:CostInformation">
          <variableMaintenanceCosts id="bcbd02a0-10de-4cf3-ba37-b37cc55e4768" name="variablemaintenance" value="1.0" xsi:type="esdl:SingleValue"/>
          <investmentCosts id="a2b8f132-ddc5-4bcc-8005-4a61c2fd9c45" name="Investmentcost" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedOperationalCosts id="b6df3b60-02dd-4d43-a9b5-ee71eae05f23" name="fixedoperational" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedMaintenanceCosts id="97f326fc-4bd0-47ee-a79e-e1ef8fb6e85a" name="fixedmaintenance" value="1.0" xsi:type="esdl:SingleValue"/>
          <installationCosts id="657e1b24-5312-4424-8d65-c2dc240764f2" name="installationcost" value="1.0" xsi:type="esdl:SingleValue"/>
        </costInformation>
        <geometry CRS="WGS84" lat="51.98544177915329" lon="4.387986660003663" xsi:type="esdl:Point"/>
      </asset>
      <asset id="d1217097-9b71-4af9-8d14-df08d3ed1edb" minTemperature="70.0" name="HeatingDemand_d121" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="618a8f33-60c9-4265-83e2-13916fc28e75" id="f5eb43f1-fb71-4d82-97cc-3fb0eb2c209d" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.3" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="505a5bf8-8bb4-4615-9274-fb12a8209ff6" id="8480259e-a24f-406b-a14c-38aac26c6127" name="out" xsi:type="esdl:OutPort"/>
        <costInformation id="600defac-6d42-4405-af6a-d51469fe32c4" xsi:type="esdl:CostInformation">
          <variableMaintenanceCosts id="5873bf6f-27f4-47dc-b369-535019981a2e" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <investmentCosts id="913b674d-caeb-4c73-a88d-a6428aa96bfa" name="investmetcost" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedOperationalCosts id="a07891c6-d661-4973-b447-5264ea0ce01b" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <fixedMaintenanceCosts id="4c2ede2b-2503-43e9-aa9b-614cfc3b92ed" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <variableOperationalCosts id="352ef2c4-b465-4476-9d73-ef54a4d6507c" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
          <installationCosts id="84d54b33-4b9e-413c-9a41-613e9dc030c5" name="NewSingleValue" value="1.0" xsi:type="esdl:SingleValue"/>
        </costInformation>
        <geometry CRS="WGS84" lat="51.98435813948783" lon="4.3883728981018075" xsi:type="esdl:Point"/>
      </asset>
      <asset id="208d2055-e5cd-4382-a561-19ddedec4428" minTemperature="70.0" name="HeatingDemand_208d" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="75895019-b7a1-45c4-8550-8f01bc3a6f2a" id="889db852-2751-425f-ab8c-989ff878acf8" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.3" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="fc0a7022-4c09-4f5a-8215-aefc9f11d75c" id="1fe139dc-c68d-4ea5-948f-ee0c5df13d1b" name="out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" lat="51.983472707127596" lon="4.388158321380616" xsi:type="esdl:Point"/>
      </asset>
      <asset id="a0c9270c-d9d0-4cd8-9a1b-17567ef62316" innerDiameter="0.16030000000000003" length="125.03425563633269" name="Pipe_a0c9" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="c4af6ec7-a8da-4412-b4c4-62d3b5c8ecb8" name="steel" thermalConductivity="0.00014862917548188605" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="b74a8c0f-9c8a-4a02-8055-c77943b414c1" name="PUR" thermalConductivity="2.1603217928675567" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036000000000000003" xsi:type="esdl:CompoundMatterComponent">
            <matter id="aff626e3-66f1-470f-a86d-f1f6adb336cb" name="HDPE" thermalConductivity="0.011627406024262562" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="f31879b6-efa1-4b2c-8740-80bebb9500a7" id="7cec87b6-9928-4cfb-9b7b-fe7b51908911" name="In" xsi:type="esdl:InPort"/>
        <port carrier="419b5016-12c9-475a-b46e-9e474b60aa8f" connectedTo="4dc27c51-a337-402a-9fdb-376aa78b98bf" id="beb9f479-61b9-48d2-a492-850dedc2826e" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98474138090263" lon="4.382675886154176" xsi:type="esdl:Point"/>
          <point lat="51.9842920630504" lon="4.384349584579469" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="450923c5-762d-4166-82db-424c282e0a71" innerDiameter="0.16030000000000003" length="125.03425563633269" name="Pipe_a0c9_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="06a60970-d7d0-4e0f-827c-b3accf4048ae" id="b1476c2d-cb2c-4efe-b854-7435ee1789e8" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="02be741c-673b-4c30-8bad-dee320e26924" id="53542379-cead-4b7c-97f5-891e1985a8cc" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9842720630504" lon="4.3843295845794685" xsi:type="esdl:Point"/>
          <point lat="51.984721380902634" lon="4.382655886154176" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="03bc68c6-cead-40bc-93e2-70838ea463e9" innerDiameter="0.16030000000000003" length="86.30055068000897" name="Pipe_e53a_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="db555a41-d15b-44fb-b994-f5004eb3bc85" id="776e5a76-57a5-4c03-b361-409ccb548b88" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="c43e1fb4-83c1-4830-afef-828db81160a7" id="2616e85e-2b1a-4ddc-8724-fc7de9173c02" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98348574596288" lon="4.383524921875001" xsi:type="esdl:Point"/>
          <point lat="51.98401436401301" lon="4.384447601776124" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="9d0308e0-00ed-458a-a2bd-d18cbdd60184" innerDiameter="0.16030000000000003" length="192.56530875797927" name="Pipe_7fab_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="1fe139dc-c68d-4ea5-948f-ee0c5df13d1b" id="fc0a7022-4c09-4f5a-8215-aefc9f11d75c" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="c43e1fb4-83c1-4830-afef-828db81160a7" id="e26af727-64fb-4b86-8b5e-00700334a2a1" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98344609935764" lon="4.3879022869873054" xsi:type="esdl:Point"/>
          <point lat="51.98392185630461" lon="4.385198620300294" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="aee94265-d252-4f7f-bf61-9afdd1c2bc3c" innerDiameter="0.16030000000000003" length="191.59622393233496" name="Pipe_2e42_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="8480259e-a24f-406b-a14c-38aac26c6127" id="505a5bf8-8bb4-4615-9274-fb12a8209ff6" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="c43e1fb4-83c1-4830-afef-828db81160a7" id="613a36f5-a31f-498a-a33d-bfe3c9132ae3" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98427867069855" lon="4.388052490692139" xsi:type="esdl:Point"/>
          <point lat="51.984146517550755" lon="4.385262993316651" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
      <asset id="acd16776-5e03-46d5-adc3-6c1f280027bc" innerDiameter="0.16030000000000003" length="202.6782401380455" name="Pipe_de73_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="31ec92fd-18c8-4788-9d6d-b96fab55e6e5" id="5b25d760-cdb5-460e-bde2-283058a4b124" name="In" xsi:type="esdl:InPort"/>
        <port carrier="9ab126bd-7b66-4bed-82cc-4c04282fe859" connectedTo="c43e1fb4-83c1-4830-afef-828db81160a7" id="5b87fbb9-44a9-41e1-9088-19561c4c5836" name="Out" xsi:type="esdl:OutPort"/>
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9853094518694" lon="4.387730625610352" xsi:type="esdl:Point"/>
          <point lat="51.98441082345643" lon="4.3851557049560554" xsi:type="esdl:Point"/>
        </geometry>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>