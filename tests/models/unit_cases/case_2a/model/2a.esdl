<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem description="unit test case" esdlVersion="v2102" id="c9e01879-71f3-4911-a398-84f711e234bb" name="2a" version="3" xmlns:esdl="http://www.tno.nl/esdl" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <energySystemInformation id="b66ef2e0-2543-43a3-98b4-60f2cad50a9a" xsi:type="esdl:EnergySystemInformation">
    <carriers id="11edbe28-baa6-44c4-a876-c8f112213a28" xsi:type="esdl:Carriers">
      <carrier id="ddd48512-6feb-439f-ba46-5a9a334b5c42" name="Heat_cold" returnTemperature="45.0" xsi:type="esdl:HeatCommodity"/>
      <carrier id="76160f2d-374c-4df5-9bee-20ff805124f8" name="Heat_hot" supplyTemperature="75.0" xsi:type="esdl:HeatCommodity"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="c7104449-b5eb-49c7-b064-a2a4b16ae6e4">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Power in MW" unit="WATT" physicalQuantity="POWER" multiplier="MEGA" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
    </quantityAndUnits>
  </energySystemInformation>
  <instance id="9824a796-cf2b-4bdf-a676-294fd5312e3a" name="Untitled instance" xsi:type="esdl:Instance">
    <area id="a31b9962-cec8-4ee4-ad06-bebbc33d8ebe" name="Untitled area" xsi:type="esdl:Area">
      <asset id="96bc0bda-8111-4377-99b2-f46b8e5496e1" innerDiameter="0.1603" length="64.71961535135489" name="Pipe_96bc" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98704736777082" lon="4.37633514404297" xsi:type="esdl:Point"/>
          <point lat="51.98707379673152" lon="4.377279281616212" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="a74f0919-95d5-4eca-826a-b757452cba60" id="e1dbf59a-8ac9-4bc1-bbf9-0c367cbda5bf" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="ffb016da-8d1a-4522-b800-ca9f27c3e00e" id="a5ffe70b-95fd-4a02-ae60-4a236533cb1f" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="51e4de22-7d65-45ae-ab4b-aae3bc7223c0" innerDiameter="0.1603" length="282.86556498109013" name="Pipe_51e4" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98716629797122" lon="4.378309249877931" xsi:type="esdl:Point"/>
          <point lat="51.988540579598386" lon="4.381785392761231" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="40f3eac1-4f91-4d7f-8424-2aad687412ed" id="b0b62b87-f41f-4a13-9806-185d06390d72" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="5721f9a7-ac9a-492b-a5ac-aedd8367163a" id="5f124c30-06d9-496a-82f0-d06afcd312e6" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="6b39bb76-cc92-4c35-ad89-81b9f18581fc" innerDiameter="0.1603" length="242.6088250579639" name="Pipe_6b39" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98695486628545" lon="4.378459453582765" xsi:type="esdl:Point"/>
          <point lat="51.9870341532846" lon="4.381999969482423" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="40f3eac1-4f91-4d7f-8424-2aad687412ed" id="45b7d4b8-f5ff-49cf-ad67-b5a1a87b0979" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="2aa11517-ed66-4c01-8153-5dbb41656d12" id="98806314-ffd0-4630-ae80-e7dcba57d12d" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="f9b0efe5-be05-4106-b9a5-dbfe320365ee" innerDiameter="0.1603" length="272.73364059156825" name="Pipe_f9b0" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9866641460877" lon="4.378309249877931" xsi:type="esdl:Point"/>
          <point lat="51.9855937509125" lon="4.381892681121827" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="40f3eac1-4f91-4d7f-8424-2aad687412ed" id="e5d44b05-ef48-462e-b0a2-8ca4f2056fd2" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="efea5685-11f5-49a3-b0c3-f6ccb5722f66" id="48d0f571-420e-494d-afc1-c9157f9ca758" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="29276338-9c84-465b-8320-1a63bbbb3a3a" innerDiameter="0.1603" length="284.5424510210789" name="Pipe_2927_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98859343574185" lon="4.382836818695069" xsi:type="esdl:Point"/>
          <point lat="51.987470229270784" lon="4.3865704536438" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="c7ce4f71-bfbe-4168-9ed8-6f2932e53e59" id="ef635a38-22de-4e0a-b708-5b410ae7952f" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="97f93925-1b6a-4c69-a718-d111d99fdf87" id="44e9eeb5-1d92-4218-80f3-8b9531be82d3" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="9a6f55a9-7e11-4dda-8436-4d91aa892ae6" innerDiameter="0.1603" length="236.58901335863" name="Pipe_9a6f_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9870341532846" lon="4.382944107055665" xsi:type="esdl:Point"/>
          <point lat="51.9870605822531" lon="4.386398792266847" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="d024024d-d64e-4512-9aa3-b02de3d0d732" id="5112abf6-a036-467f-bb3b-4ee2db7f237c" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="97f93925-1b6a-4c69-a718-d111d99fdf87" id="28c1aa2c-df66-4ba4-b813-9c9a3d735c8e" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="a71819b2-eff9-48f7-b67f-9311c158d144" innerDiameter="0.1603" length="285.4445187317924" name="Pipe_a718_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.985540891228666" lon="4.382858276367188" xsi:type="esdl:Point"/>
          <point lat="51.9867830773058" lon="4.386506080627442" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="0791747e-cce6-4ab2-ac2f-87cf1b610781" id="559a6328-7fce-4737-9dcf-cae3f34e770d" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="97f93925-1b6a-4c69-a718-d111d99fdf87" id="52ee035f-154d-483e-86b0-38f57797c8e4" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="8592a065-d8cf-43f8-a4e9-029a3e24e9d1" innerDiameter="0.1603" length="73.4836667788432" name="Pipe_8592_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98716629797122" lon="4.387407302856446" xsi:type="esdl:Point"/>
          <point lat="51.987179512418464" lon="4.388480186462403" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="96249ee7-bc5c-45b8-8859-46193190ec4e" id="e4e79366-7e9d-408c-837f-59d56c160aa7" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="9a9cc77c-9dd7-4430-94ae-301bb6f1f95f" id="bc9ed99c-ab3a-4fc1-b0fb-9eef40810468" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="f3b9de9c-85d0-4cbb-8e86-2d0bd3dd6498" innerDiameter="0.1603" length="42.63700468550451" name="Pipe_f3b9" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.987324871080524" lon="4.382514953613282" xsi:type="esdl:Point"/>
          <point lat="51.987708087109915" lon="4.382493495941163" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="01b0f5de-4f35-496b-88c9-d0ae8c59aec7" id="1ae4c886-3078-4a01-a126-43e5a5738eac" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="874dc021-09f4-46ec-bb5a-c3276916fbfd" id="3f7958f8-88e8-47f3-ae8d-b5cd4ce92093" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="74846440-61d9-4f57-9bce-77e3664ba832" minTemperature="70.0" name="HeatingDemand_7484" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <geometry lat="51.9894259117724" lon="4.382386207580567" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="00f134de-804f-4ac4-b829-5e44fc433791" id="b145eba8-612d-42b7-bea7-7e22a2440509" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.3" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="acbb3410-c6a6-4495-bf02-2c2f259d40ca" id="a77a0773-d214-40bf-b424-9dfcb7b26667" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="c6c8d8f7-b6ae-4e8b-be9a-72282a5c3f1e" minTemperature="70.0" name="HeatingDemand_c6c8" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <geometry CRS="WGS84" lat="51.988144156533934" lon="4.382386207580567" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="3f7958f8-88e8-47f3-ae8d-b5cd4ce92093" id="874dc021-09f4-46ec-bb5a-c3276916fbfd" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.3" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="b43e5090-a0f4-4cfa-877f-c2e75afc2c2f" id="fc3a02e2-fe94-4641-b94b-99161215f0a8" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="6f99424a-004e-4fb4-8edb-17c374d5be6b" minTemperature="70.0" name="HeatingDemand_6f99" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <geometry lat="51.984576191039736" lon="4.382514953613282" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="3b1e06ac-9998-4c62-8764-7e582bd478bc" id="30713528-878b-46fd-82e2-9da8eb95075e" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.3" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="c02aa852-9471-480d-a9da-8d794a356285" id="d35984bb-f7da-4313-bfba-0e651dfe3ac6" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="5d926b6c-a5c3-4a84-9fd7-fb2b0a793c53" innerDiameter="0.1603" length="42.639244076735075" name="Pipe_5d92" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9853558818439" lon="4.382429122924806" xsi:type="esdl:Point"/>
          <point lat="51.984972645687144" lon="4.382450580596925" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="6620ebae-26d8-4a04-8e32-04a244a56144" id="ff5409f6-79ac-427a-ac8a-697927a33ef2" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="30713528-878b-46fd-82e2-9da8eb95075e" id="3b1e06ac-9998-4c62-8764-7e582bd478bc" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="6604187b-1d8f-4f4e-ab60-845ee74b3fa3" innerDiameter="0.1603" length="36.733067324883415" name="Pipe_6604" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9887387898147" lon="4.38232183456421" xsi:type="esdl:Point"/>
          <point lat="51.989069138225666" lon="4.38232183456421" xsi:type="esdl:Point"/>
        </geometry>
        <material compoundType="LAYERED" xsi:type="esdl:CompoundMatter">
          <component layerWidth="0.004" xsi:type="esdl:CompoundMatterComponent">
            <matter id="5d4601b4-12cc-4da9-956a-f58d746f870c" name="steel" thermalConductivity="52.15" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.03725" xsi:type="esdl:CompoundMatterComponent">
            <matter id="74926ea3-65c8-49d4-99e5-4733f851d1c8" name="PUR" thermalConductivity="0.027" xsi:type="esdl:Material"/>
          </component>
          <component layerWidth="0.0036" xsi:type="esdl:CompoundMatterComponent">
            <matter id="497f5bca-482e-4e23-9daf-ea5a9ba73bd4" name="HDPE" thermalConductivity="0.4" xsi:type="esdl:Material"/>
          </component>
        </material>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="ae672672-7869-4460-b39b-090ec4e1e96f" id="471c8497-b34b-449d-bbf9-4ec563090989" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="b145eba8-612d-42b7-bea7-7e22a2440509" id="00f134de-804f-4ac4-b829-5e44fc433791" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="f76693ae-be7f-4246-ba13-c78337ea0e54" maxTemperature="85.0" minTemperature="65.0" name="GeothermalSource_fafd" power="100000000.0" xsi:type="esdl:ResidualHeatSource">
        <geometry CRS="WGS84" lat="51.9870405822531" lon="4.375349548797608" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="e1dbf59a-8ac9-4bc1-bbf9-0c367cbda5bf" id="a74f0919-95d5-4eca-826a-b757452cba60" name="Out" xsi:type="esdl:OutPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="6be755c7-7cbf-497a-bcc8-c65b7e6e1097" id="32e53487-903e-425c-a40c-22db341df148" name="In" xsi:type="esdl:InPort"/>
      </asset>
      <asset id="06f4c7dd-2293-48a3-aa04-76e4d2ab2a6e" maxTemperature="85.0" minTemperature="65.0" name="GeothermalSource_27cb" power="100000000.0" xsi:type="esdl:ResidualHeatSource">
        <geometry lat="51.986925133624595" lon="4.389113187789918" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="22a22cb4-c95a-4a38-a1a4-148b0e1b14fd" id="45cc9a58-c769-471a-88e5-1f5b47a7b3c1" name="Out" xsi:type="esdl:OutPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="5a0078b9-e705-45af-b9d3-1ed7cc78100d" id="10fc289a-2780-4310-a1d6-2414180bdcdb" name="In" xsi:type="esdl:InPort"/>
      </asset>
      <asset id="1cc87169-3bbc-4e30-8c60-7b9c4fadbec4" name="Pump_1cc8" pumpCapacity="10.0" xsi:type="esdl:Pump">
        <geometry CRS="WGS84" lat="51.98723237016834" lon="4.388732314109803" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="bc9ed99c-ab3a-4fc1-b0fb-9eef40810468" id="9a9cc77c-9dd7-4430-94ae-301bb6f1f95f" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="10fc289a-2780-4310-a1d6-2414180bdcdb" id="5a0078b9-e705-45af-b9d3-1ed7cc78100d" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="c4a54822-fccb-40ec-9306-e3c3a4909196" name="Pump_c4a5" pumpCapacity="10.0" xsi:type="esdl:Pump">
        <geometry CRS="WGS84" lat="51.9869845989266" lon="4.375785291194917" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="dcebb219-1aae-4f7e-9440-b81e6954e279" id="327c6e94-b590-4fb7-af8a-d84384c8ce61" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="32e53487-903e-425c-a40c-22db341df148" id="6be755c7-7cbf-497a-bcc8-c65b7e6e1097" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="b060e8a6-cb4e-4f2d-a17f-cb922b2080dd" name="Joint_b060" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98725219180846" lon="4.3867743015289316" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="0fa3e162-c2f1-4851-b55e-d844aa031d23" id="b467ea4f-fa8a-4921-8ce2-aec564614219" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="7bff3cae-f595-4e84-a001-0c7cf50fdffd 525800f1-ac51-46d4-be02-b6bfe594676d ade8d4b8-7e71-4828-a484-f7ef9433cd76" id="74f7f26a-1327-4ed4-ac42-12e4fa759ad0" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="bfea62e9-81af-45b1-be77-7910c4466193" name="Joint_bfea" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98698129530076" lon="4.386785030364991" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="44e9eeb5-1d92-4218-80f3-8b9531be82d3 28c1aa2c-df66-4ba4-b813-9c9a3d735c8e 52ee035f-154d-483e-86b0-38f57797c8e4" id="97f93925-1b6a-4c69-a718-d111d99fdf87" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="e4e79366-7e9d-408c-837f-59d56c160aa7" id="96249ee7-bc5c-45b8-8859-46193190ec4e" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="0f8349e8-009c-4adc-a43c-8cfb5cedeb4c" name="Joint_0f83" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.987139869065096" lon="4.382375478744508" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="98806314-ffd0-4630-ae80-e7dcba57d12d 888bc8f9-0c5d-4534-b3a6-185f7518e218" id="2aa11517-ed66-4c01-8153-5dbb41656d12" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="1ae4c886-3078-4a01-a126-43e5a5738eac" id="01b0f5de-4f35-496b-88c9-d0ae8c59aec7" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="d63737c3-87c0-4720-b7f7-01de0c102434" name="Joint_d637" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98688879367896" lon="4.382514953613282" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="595907ec-d7d7-4211-9852-b1bd53d4e9f4" id="fc3843d8-afcc-4030-b0e5-88a2c19f2f8c" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="5112abf6-a036-467f-bb3b-4ee2db7f237c d500437d-e4cd-4c48-a921-919e175cf5b2" id="d024024d-d64e-4512-9aa3-b02de3d0d732" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="f3942dd0-8b7e-45b1-9714-5cc656145179" name="Joint_f394" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98570607753357" lon="4.382225275039674" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="0e19db7e-1dce-456f-b8e4-add92d7dc229" id="e5267f56-bbf4-4852-a25f-d3776fe38d16" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="559a6328-7fce-4737-9dcf-cae3f34e770d e4112faf-e3f7-4310-8952-b49f4efe4ea7" id="0791747e-cce6-4ab2-ac2f-87cf1b610781" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="d50228b8-e571-477d-9d6e-7593dcbea442" name="Joint_d502" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98553428376382" lon="4.382504224777223" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="48d0f571-420e-494d-afc1-c9157f9ca758 926f4acf-acbb-48c9-a1ff-cb732c3c3191" id="efea5685-11f5-49a3-b0c3-f6ccb5722f66" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="ff5409f6-79ac-427a-ac8a-697927a33ef2" id="6620ebae-26d8-4a04-8e32-04a244a56144" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="f54ebdd3-b008-45ec-ad9a-b8fc1d768586" name="Joint_f54e" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.987146476293084" lon="4.377590417861939" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="0a6a08d6-3271-42e5-814c-74081a0423cc a660e689-f2e4-4052-808f-b9270f987189 005cb3f2-c5a5-4415-8ace-f7602e91b7f2" id="318839fd-89d8-488e-b4e6-4dcb4a4ced2a" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="890bc53f-488a-4948-9825-d6481e309dcb" id="cbbb9bd2-4eae-4366-b574-8448f3c03654" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="2fd380b7-206d-4c2a-8db8-6f0466330dab" name="Joint_2fd3" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98687557914596" lon="4.377869367599488" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="a5ffe70b-95fd-4a02-ae60-4a236533cb1f" id="ffb016da-8d1a-4522-b800-ca9f27c3e00e" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="b0b62b87-f41f-4a13-9806-185d06390d72 45b7d4b8-f5ff-49cf-ad67-b5a1a87b0979 e5d44b05-ef48-462e-b0a2-8ca4f2056fd2" id="40f3eac1-4f91-4d7f-8424-2aad687412ed" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="4981f45b-3f7f-40a1-9ab9-767647333938" name="Joint_4981" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98862316729514" lon="4.38206434249878" xsi:type="esdl:Point"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="9a581781-6f74-49c6-8333-bcff43165805" id="1e88ea92-59da-4509-b575-6ffe6da304f7" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="ef635a38-22de-4e0a-b708-5b410ae7952f 41ca1e9a-d112-419a-90fc-8fa5e01fc07a" id="c7ce4f71-bfbe-4168-9ed8-6f2932e53e59" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="82278cef-24cf-4413-9860-8715f95bb007" name="Joint_8227" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.988487723392524" lon="4.382375478744508" xsi:type="esdl:Point"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="5f124c30-06d9-496a-82f0-d06afcd312e6 c0d628fd-5d78-4c15-b6e5-f863abb50137" id="5721f9a7-ac9a-492b-a5ac-aedd8367163a" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="471c8497-b34b-449d-bbf9-4ec563090989" id="ae672672-7869-4460-b39b-090ec4e1e96f" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="25c0ce8d-a7ef-4d87-8538-606d50dea54b" innerDiameter="0.1603" length="64.71961535135489" name="Pipe_96bc_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98705379673152" lon="4.377259281616212" xsi:type="esdl:Point"/>
          <point lat="51.98702736777082" lon="4.3763151440429695" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="cbbb9bd2-4eae-4366-b574-8448f3c03654" id="890bc53f-488a-4948-9825-d6481e309dcb" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="327c6e94-b590-4fb7-af8a-d84384c8ce61" id="dcebb219-1aae-4f7e-9440-b81e6954e279" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="0f1e9676-8f12-4e7a-8239-cfe836fa6d00" innerDiameter="0.1603" length="282.86556498109013" name="Pipe_51e4_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98852057959839" lon="4.381765392761231" xsi:type="esdl:Point"/>
          <point lat="51.98714629797122" lon="4.3782892498779304" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="c7ce4f71-bfbe-4168-9ed8-6f2932e53e59" id="41ca1e9a-d112-419a-90fc-8fa5e01fc07a" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="318839fd-89d8-488e-b4e6-4dcb4a4ced2a" id="0a6a08d6-3271-42e5-814c-74081a0423cc" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="8ef48aeb-50af-4f04-baae-ea8ced21ec13" innerDiameter="0.1603" length="242.6088250579639" name="Pipe_6b39_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9870141532846" lon="4.381979969482423" xsi:type="esdl:Point"/>
          <point lat="51.98693486628545" lon="4.378439453582764" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="d024024d-d64e-4512-9aa3-b02de3d0d732" id="d500437d-e4cd-4c48-a921-919e175cf5b2" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="318839fd-89d8-488e-b4e6-4dcb4a4ced2a" id="a660e689-f2e4-4052-808f-b9270f987189" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="d6d374b9-4e25-43e2-9bbd-37c580a1414b" innerDiameter="0.1603" length="272.73364059156825" name="Pipe_f9b0_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9855737509125" lon="4.381872681121827" xsi:type="esdl:Point"/>
          <point lat="51.9866441460877" lon="4.3782892498779304" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="0791747e-cce6-4ab2-ac2f-87cf1b610781" id="e4112faf-e3f7-4310-8952-b49f4efe4ea7" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="318839fd-89d8-488e-b4e6-4dcb4a4ced2a" id="005cb3f2-c5a5-4415-8ace-f7602e91b7f2" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="be98bc9a-55f6-47a5-83b6-f6f9d03ad8f8" innerDiameter="0.1603" length="36.733067324883415" name="Pipe_6604_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98904913822567" lon="4.38230183456421" xsi:type="esdl:Point"/>
          <point lat="51.9887187898147" lon="4.38230183456421" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="a77a0773-d214-40bf-b424-9dfcb7b26667" id="acbb3410-c6a6-4495-bf02-2c2f259d40ca" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="1e88ea92-59da-4509-b575-6ffe6da304f7" id="9a581781-6f74-49c6-8333-bcff43165805" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="3f21c500-7f55-45dd-b8d4-c90b02fcd6f1" innerDiameter="0.1603" length="42.63700468550451" name="Pipe_f3b9_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.987688087109916" lon="4.382473495941163" xsi:type="esdl:Point"/>
          <point lat="51.987304871080525" lon="4.382494953613282" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="fc3a02e2-fe94-4641-b94b-99161215f0a8" id="b43e5090-a0f4-4cfa-877f-c2e75afc2c2f" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="fc3843d8-afcc-4030-b0e5-88a2c19f2f8c" id="595907ec-d7d7-4211-9852-b1bd53d4e9f4" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="dec5d7e5-d1d2-4305-bf26-677bf91122ee" innerDiameter="0.1603" length="284.5424510210789" name="Pipe_2927" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.987450229270785" lon="4.3865504536438" xsi:type="esdl:Point"/>
          <point lat="51.98857343574185" lon="4.382816818695069" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="74f7f26a-1327-4ed4-ac42-12e4fa759ad0" id="7bff3cae-f595-4e84-a001-0c7cf50fdffd" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="5721f9a7-ac9a-492b-a5ac-aedd8367163a" id="c0d628fd-5d78-4c15-b6e5-f863abb50137" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="76a64b03-e5af-4a6c-acd3-70e846cd44c2" innerDiameter="0.1603" length="236.58901335863" name="Pipe_9a6f" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9870405822531" lon="4.3863787922668465" xsi:type="esdl:Point"/>
          <point lat="51.9870141532846" lon="4.382924107055665" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="74f7f26a-1327-4ed4-ac42-12e4fa759ad0" id="525800f1-ac51-46d4-be02-b6bfe594676d" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="2aa11517-ed66-4c01-8153-5dbb41656d12" id="888bc8f9-0c5d-4534-b3a6-185f7518e218" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="8fbb7fbc-3433-43c4-96fa-f6fafbddd511" innerDiameter="0.1603" length="285.4445187317924" name="Pipe_a718" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.9867630773058" lon="4.386486080627442" xsi:type="esdl:Point"/>
          <point lat="51.98552089122867" lon="4.382838276367188" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="74f7f26a-1327-4ed4-ac42-12e4fa759ad0" id="ade8d4b8-7e71-4828-a484-f7ef9433cd76" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="efea5685-11f5-49a3-b0c3-f6ccb5722f66" id="926f4acf-acbb-48c9-a1ff-cb732c3c3191" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="21b1b1d5-0aee-4bc2-9aca-050b374c6898" innerDiameter="0.1603" length="73.4836667788432" name="Pipe_8592" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.987159512418465" lon="4.388460186462403" xsi:type="esdl:Point"/>
          <point lat="51.98714629797122" lon="4.387387302856446" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="45cc9a58-c769-471a-88e5-1f5b47a7b3c1" id="22a22cb4-c95a-4a38-a1a4-148b0e1b14fd" name="In" xsi:type="esdl:InPort"/>
        <port carrier="76160f2d-374c-4df5-9bee-20ff805124f8" connectedTo="b467ea4f-fa8a-4921-8ce2-aec564614219" id="0fa3e162-c2f1-4851-b55e-d844aa031d23" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="16ae6491-df96-404a-a466-4a39c67e8e21" innerDiameter="0.1603" length="42.639244076735075" name="Pipe_5d92_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.984952645687144" lon="4.382430580596925" xsi:type="esdl:Point"/>
          <point lat="51.9853358818439" lon="4.3824091229248054" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="d35984bb-f7da-4313-bfba-0e651dfe3ac6" id="c02aa852-9471-480d-a9da-8d794a356285" name="In" xsi:type="esdl:InPort"/>
        <port carrier="ddd48512-6feb-439f-ba46-5a9a334b5c42" connectedTo="e5267f56-bbf4-4852-a25f-d3776fe38d16" id="0e19db7e-1dce-456f-b8e4-add92d7dc229" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
