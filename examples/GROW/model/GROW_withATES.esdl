<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" name="Untitled EnergySystem with return network with return network" esdlVersion="v2111" id="7cf3bc36-1398-4cde-bc73-3c82edf47ff3_with_return_network_with_return_network" description="" version="13">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="92cbf842-4d76-4b77-b099-f5ca80990993">
    <carriers xsi:type="esdl:Carriers" id="e5e0bad2-dd5e-4e1d-a1fe-d19fcc89d802">
      <carrier xsi:type="esdl:HeatCommodity" supplyTemperature="70.0" id="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75" name="Heat"/>
      <carrier xsi:type="esdl:HeatCommodity" returnTemperature="40.0" id="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret" name="Heat_ret"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="aea45f45-caf3-476a-98b3-4ebb49595202">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Power in MW" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b" unit="WATT" physicalQuantity="POWER" multiplier="MEGA"/>
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Energy in GJ" id="eb07bccb-203f-407e-af98-e687656a221d" unit="JOULE" physicalQuantity="ENERGY" multiplier="GIGA"/>
    </quantityAndUnits>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" name="Untitled Instance" id="71555e19-f22c-4963-9f22-86d3c1e732c5">
    <area xsi:type="esdl:Area" name="Untitled Area" id="409fd1ef-d637-4268-bca1-c33bf8cb8ecd">
      <asset xsi:type="esdl:Joint" name="Joint_bdf0" id="bdf0ba14-9269-43d5-bd14-6ec0ef190b3a">
        <geometry xsi:type="esdl:Point" lat="52.0443961581157" lon="4.3101945519447336" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c3e7a4d0-a8ce-4688-af7f-3ef98bcc7f06 9df74fea-c3d1-4733-8cbe-02f9ee031511 c668dd7b-24f6-4e05-97b1-5ba65a743f93" id="d44b1d62-d2be-482a-9c1b-29c7a5073711" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c709f60d-55ac-4791-8b4e-e8cd6733fc33" connectedTo="f55f315e-169c-46c9-84f7-a7f305dd7165 ec75d694-eb55-4409-880c-6c342f3abbfe cea9d279-8464-4774-b382-34e7b3e69c5a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:ResidualHeatSource" power="20000000.0" name="WarmteLink" id="fcce9b72-d99a-43b0-91d3-06818bcc354d">
        <geometry xsi:type="esdl:Point" lat="52.044602369454225" lon="4.310012161731721" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="6882fee0-bcea-4c96-9030-d8b70894d27a" connectedTo="0410dd85-26b3-4ceb-bbb9-a8a738a1a020" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d05ca38e-424c-4325-8e75-69379f90bf24" id="85c8013d-9a9b-4efc-87ba-cfd3d999adb6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="a0b76b52-a5ed-46a3-af24-625b16480fdb">
          <investmentCosts xsi:type="esdl:SingleValue" id="ee23c233-ea88-4f4f-9685-4187cf8814a6" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="5d37cf6d-ae69-47a1-8aa8-b59dd7676922" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="36133d10-eb52-4468-90ba-1a10bb15916b" value="32.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="4ff8b3a2-1a38-41ee-b08b-9d90cc327fef" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="2291626f-7d2b-4fe9-9816-55b04effd6b5" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="1ec0b548-0d2e-41dd-8c8d-97f5abac577f" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:GeothermalSource" power="20000000.0" name="GeoSource" id="8cf9a76a-37b3-46e2-8b49-9df756620be5">
        <geometry xsi:type="esdl:Point" lat="52.04469805119214" lon="4.3102964758872995" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="bd8a6c48-a664-4f32-bf2e-68046a3bfeb4" connectedTo="cc9f9cec-d565-422a-b645-29c24f85678d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f7d48b4e-2348-4b24-9af9-98598f270be2" id="3aacb2a9-615b-4047-8b01-04f5f1f32379" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="b0498a6e-b4d6-4ab5-96a7-c4e3918a3edb">
          <fixedOperationalCosts xsi:type="esdl:SingleValue" id="25fafbb2-9fcc-4ac6-904a-f8447e22765c" value="51367.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="539ee2d5-60b5-4e19-9e9e-384789b91e51" description="Cost in EUR/MW" perUnit="WATT" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </fixedOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="9c6bb73b-8430-49e8-85a1-b3d85d6669cb" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" unit="EURO" description="Cost in EUR" id="f0ac8533-0914-4776-811e-9db8c0880e8d" physicalQuantity="COST"/>
          </installationCosts>
          <investmentCosts xsi:type="esdl:SingleValue" id="6d4bfe1c-3c57-4134-a06d-160722908207" value="1800000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="572d594f-98fd-4bbe-bd61-8fa9cd5ccced" description="Cost in EUR/MW" perUnit="WATT" perMultiplier="MEGA" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="c1019862-6be2-4a24-aba0-409d28b841b3" value="6.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="c05b676c-30bd-4302-bf9e-f6a6e8b4ce6e" description="Cost in EUR/MWh" perUnit="WATTHOUR" perMultiplier="MEGA" unit="EURO" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <fixedMaintenanceCosts xsi:type="esdl:SingleValue" id="cfe25a6d-8fe9-4de6-9a24-aef75b77da8d" value="15300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="b326553c-4634-440a-9cdd-1c8ec3a7e52c" description="Cost in EUR/MW" perUnit="WATT" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </fixedMaintenanceCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Kleurenbuurt" id="404684bc-3874-42bf-a0d9-5f184591b57a">
        <geometry xsi:type="esdl:Point" lat="52.04172687268464" lon="4.313077926635743" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e415e84f-84b0-4133-925f-50c1903f9eb8" id="59a43508-4328-4e0a-94e1-fd6036172036" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="10.88" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="f3f7d77b-b199-4527-87de-0ec14c1647eb" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Kleurenbuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="0ab74980-9ef0-433d-bf30-ec3cf137ad7c" connectedTo="3a78bae6-bc9c-45c1-bd5d-6c722675944f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Artiestbuurt_2" id="7cfa7e7b-534a-4740-8fd8-f81c294bc24c">
        <geometry xsi:type="esdl:Point" lat="52.038644953041" lon="4.316178560256959" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="48b9c98f-4bc7-44fc-baee-1e9681c16c1f" id="9da9e597-f6fb-4831-b124-6de0044f5a3e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="93f35859-1168-465d-b4f3-a2b1d8cc4ceb" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="http://10.30.2.1" port="8086" field="Artiestbuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="6db24e9f-83c6-40be-a58a-faaa0e155b00" connectedTo="8d924dd2-37dc-4d06-9bb4-a028abc8d29c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Artiestenbuurt_1" id="a905fe2c-9611-4832-980d-51a0a8c00ae9">
        <geometry xsi:type="esdl:Point" lat="52.037722645629316" lon="4.318581819534303" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="eb08503b-b0a8-4713-b941-b626531b19e5" id="bb93e16a-1131-425f-b438-0a1afd02b5b7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="8.76" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="601276f8-c4f0-4ad4-b211-c7b85530bf89" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="http://10.30.2.1" port="8086" field="Artiestbuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="3b1b483c-7d9e-4ffd-aa58-e16edef4cb44" connectedTo="4f760ceb-53f0-4867-bf62-bbecaf12c768" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer1" id="70837549-df4b-4009-840e-8bb40b6aa7fc">
        <geometry xsi:type="esdl:Point" lat="52.03756425102864" lon="4.317970275878907" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="57581db4-cc0f-4260-8538-bf2599cdece2" connectedTo="47b7daf3-d849-4176-8825-84e4f1b46b58" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="2f3babb2-f6e5-469e-a3b5-374d087ef970" id="d0832916-11c6-4f8f-a601-0fb90b1f5764" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Muziekbuurt_2" id="6c5208f4-a292-41b9-974d-f8432d363d0e">
        <geometry xsi:type="esdl:Point" lat="52.03491765821515" lon="4.319719076156617" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="89bfd689-d399-42f9-a0c7-145a09816032" id="8c696b24-1afb-4af2-ab02-4c911a43f9a2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="b53b972f-cf48-4f3e-99ae-9bae9ee8bb51" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Muziekbuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="9076d363-fa38-4470-a566-1c692a87bae7" connectedTo="7a9cd346-22a0-467f-8eae-b72541b0bef4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Muziekbuurt_3" id="ba67bf8f-47f6-4884-8165-b85f6a49b7fa">
        <geometry xsi:type="esdl:Point" lat="52.033208179577564" lon="4.3213941156864175" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="27ad4ddb-4499-4264-b6d2-2c21b71e68c3" id="504787dd-d2c9-44ed-afb3-88071b846d9b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="1ef0ad6e-7d2a-44d4-973c-ef0b990de334" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Muziekbuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="60efe325-8f39-4540-b7b2-cfa8d605a9db" connectedTo="cb5c01ac-0415-418b-a92f-e4d7b001cd10" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Muziekbuurt_1" id="59b7a07e-01ed-4937-8989-23395852dd95">
        <geometry xsi:type="esdl:Point" lat="52.03182866846768" lon="4.324289560317994" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="6bfc5f80-e091-40d1-bc29-9d71281d5251" id="0148e3f8-6850-4ab4-a3d3-3e49ebd7e137" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="17944386-a137-4cac-965e-e8b5a399bb3c" connectedTo="129982a4-6b66-43c8-ba16-1b2ee1caf049" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8cee" id="8ceef208-2d8a-402b-84e9-0ce3cb280e71">
        <geometry xsi:type="esdl:Point" lat="52.04164900734322" lon="4.312788248062135" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e6e03317-762d-45a5-9481-5e52464e2e95 5e043cfa-f07b-4f73-a9d2-e76a21840101" id="57dd5638-d1ba-4ed9-8efe-f1da25ea2f52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d9e086ab-5ad3-4235-98ad-0f109001a4d4" connectedTo="6058d754-7518-47ba-8ed7-fe056fdcae0e d9094d66-9b87-4773-bbb0-0d4473d07636 19c66535-7622-4e61-8056-ad6c9e150172" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_5163" id="516346a3-f37f-4382-92bc-213da6b5b607">
        <geometry xsi:type="esdl:Point" lat="52.03879819320885" lon="4.316698908805848" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="8e2c060b-efa1-4e27-afe6-48cdc7ae2a1d 03764218-27f8-4596-a02c-ec0365d0890f" id="b36fcc81-7a66-4b77-9108-48ee1ed0fda6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ba4ce072-68d5-4603-aa4d-7ecf059f5020" connectedTo="07e7713b-2ef3-4117-bb46-00cf80079901 c57ca1fe-b04f-41c6-abd1-1dd286045e4f 6afb228c-79ec-408d-a489-5549a2b01a4d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_828a" id="828af701-3d67-4318-ad08-42697c7c0733">
        <geometry xsi:type="esdl:Point" lat="52.03766978308298" lon="4.3182599544525155" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="74b1d111-dfb0-4780-a4d8-7740855d92cb 630b935b-252e-489c-a6a6-d4d56bf1668e" id="0306fc7e-d183-4db6-86a6-f523ab77e17e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="39f0b02a-1dbf-42e7-99a6-143e5868754a" connectedTo="c0ce56e8-260b-4dea-a690-86d623612ee9 767d00ad-0379-4618-ad42-322481c41a4a d929c625-08e9-4d06-bf5e-03cd9be0d981" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1514" id="15141c6e-dad7-44a7-847d-d0e2b0fcc2f8">
        <geometry xsi:type="esdl:Point" lat="52.03473314368742" lon="4.319311380386353" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b58b1d37-d79b-4e73-a9ba-3977125cbc31 16cfc909-805a-4d30-bcbf-a0e703a6fef6" id="2d464b69-7340-4d13-b04e-8d642145a6fe" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="72c1d1a9-8f49-4a26-b861-ec067835ef38" connectedTo="d4a42b33-e4ca-4743-a240-9b3b0847ff60 c8146f79-e5f6-430a-97ce-253f14b77e21 4c1cd0f9-9981-455f-bf4b-a6207be5c83e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_217a" id="217ae474-eee0-4911-9bd0-9643fcd698be">
        <geometry xsi:type="esdl:Point" lat="52.03287207234976" lon="4.320539832115174" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e4b21d04-196b-460f-a4ff-b3b7de3c9ccb 27795ecb-87b0-47e2-b93e-1dc3964d0352" id="b2547ccb-93f2-4c84-839d-0e34211aeece" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4a82116b-fede-476f-9135-24d51f45f1b3" connectedTo="c1318085-1ced-470d-b53e-6ac138f43364 ac87490f-88d5-43a5-89d8-300f719ebfec 075fd612-bea2-4d9c-9731-dd32cda939a1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1_ret" diameter="DN400" length="355.4" name="Pipe1" id="Pipe1" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0443961581157" lon="4.3101945519447336"/>
          <point xsi:type="esdl:Point" lat="52.04365837648511" lon="4.3111950159072885"/>
          <point xsi:type="esdl:Point" lat="52.04164900734322" lon="4.312788248062135"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c709f60d-55ac-4791-8b4e-e8cd6733fc33" id="f55f315e-169c-46c9-84f7-a7f305dd7165" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e6e03317-762d-45a5-9481-5e52464e2e95" connectedTo="57dd5638-d1ba-4ed9-8efe-f1da25ea2f52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="def08568-97c0-42b7-bde4-542b8617d24f">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2_ret" diameter="DN400" length="495.8" name="Pipe2" id="Pipe2" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04164900734322" lon="4.312788248062135"/>
          <point xsi:type="esdl:Point" lat="52.03909226576622" lon="4.314950108528138"/>
          <point xsi:type="esdl:Point" lat="52.03947499237764" lon="4.316108822822572"/>
          <point xsi:type="esdl:Point" lat="52.03879819320885" lon="4.316698908805848"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d9e086ab-5ad3-4235-98ad-0f109001a4d4" id="6058d754-7518-47ba-8ed7-fe056fdcae0e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="8e2c060b-efa1-4e27-afe6-48cdc7ae2a1d" connectedTo="b36fcc81-7a66-4b77-9108-48ee1ed0fda6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2bbc120b-1217-4f7a-ad06-919f825edfce">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3_ret" diameter="DN400" length="170.5" name="Pipe3" id="Pipe3" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03879819320885" lon="4.316698908805848"/>
          <point xsi:type="esdl:Point" lat="52.038547864509624" lon="4.317444562911988"/>
          <point xsi:type="esdl:Point" lat="52.03766978308298" lon="4.3182599544525155"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba4ce072-68d5-4603-aa4d-7ecf059f5020" id="07e7713b-2ef3-4117-bb46-00cf80079901" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="74b1d111-dfb0-4780-a4d8-7740855d92cb" connectedTo="0306fc7e-d183-4db6-86a6-f523ab77e17e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c0479248-dfe2-4c09-a9e7-5ce7d3a92db0">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4_ret" diameter="DN400" length="26.1" name="Pipe4" id="Pipe4" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.044602369454225" lon="4.310012161731721"/>
          <point xsi:type="esdl:Point" lat="52.0443961581157" lon="4.3101945519447336"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="6882fee0-bcea-4c96-9030-d8b70894d27a" id="0410dd85-26b3-4ceb-bbb9-a8a738a1a020" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c3e7a4d0-a8ce-4688-af7f-3ef98bcc7f06" connectedTo="d44b1d62-d2be-482a-9c1b-29c7a5073711" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="f93441fc-464f-43a5-9998-e147aabd10cd">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe5_ret" diameter="DN400" length="34.3" name="Pipe5" id="Pipe5" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04469805119214" lon="4.3102964758872995"/>
          <point xsi:type="esdl:Point" lat="52.0443961581157" lon="4.3101945519447336"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="bd8a6c48-a664-4f32-bf2e-68046a3bfeb4" id="cc9f9cec-d565-422a-b645-29c24f85678d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9df74fea-c3d1-4733-8cbe-02f9ee031511" connectedTo="d44b1d62-d2be-482a-9c1b-29c7a5073711" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="07d9b1a5-f1e4-4b5d-86ce-fa5507105e11">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Presidentenbuurt" id="6f01df0b-7c29-467b-93e8-691f3ef43835">
        <geometry xsi:type="esdl:Point" lat="52.03904506381387" lon="4.30440902709961" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d2065857-7f90-4645-8e5b-af7b096dc84c" id="49eb6e79-0f68-44c0-a9ca-3398028e5b5e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.5" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="601949fb-26ab-411e-9db1-28e02541c1cc" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Presidentenbuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="6194bccd-c26a-469b-88fd-6d9d5453065a" connectedTo="0efab627-79b2-432a-b65e-2ea0b7de34f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_DeStrijp_2" id="7fc734dc-8ab1-416d-8076-2b7c89508063">
        <geometry xsi:type="esdl:Point" lat="52.036880614888744" lon="4.298100471496583" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b7c5f7da-761d-4d69-90c8-cc18e551c6b5" id="856149d8-c74a-496e-a6fd-b9ed82e843af" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="4.07" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="8050c84c-63fb-4953-af82-735fb1770edd" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="De Strijp">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="824b1653-94c7-408a-a1a6-9aab266bfa89" connectedTo="01ae5f1b-6a10-430c-aefb-6b981fc85911" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_DeStrijp_3" id="15759457-af7f-4c6e-852b-17f6cc935e77">
        <geometry xsi:type="esdl:Point" lat="52.036880614888744" lon="4.296705722808839" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="589c1244-bdbf-42a0-a6cc-b99c18916900" id="414973d8-31e5-4237-9ebf-b319c6ec404b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="4.07" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="cc4fa425-6a51-415c-bb86-deabd5ee1d64" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="De Strijp">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="327389ca-879f-499b-9a03-97a00ea571f4" connectedTo="997ff413-e4c0-4cad-a81e-b1474abdb91f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Ministerbuurt_1" id="f60821f7-86e2-4ad3-aad2-3faef4a7488c">
        <geometry xsi:type="esdl:Point" lat="52.035745557102025" lon="4.303786754608155" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="00f6e2a9-bd19-43ec-965b-fae9fcc33bdb" id="70dc27c6-6a93-4801-b83e-140394796527" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="4.83" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="ae8e24dd-cf5f-4fc7-a304-264ebde69618" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="MinisterBuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="6cf7348f-8aac-4f1e-9e31-6ac8295f5706" connectedTo="2f4e69b7-96f9-4eb7-ac5d-97a6255dbb7c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Ministerbuurt_2" id="4c0f0eff-1d3c-4e82-8093-bcf49ac3ea9a">
        <geometry xsi:type="esdl:Point" lat="52.03474245879195" lon="4.309816360473634" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f9aae992-24b9-4fb8-adbb-c8a0846c999d" id="66455895-fbde-4174-a263-86f9586c2fc8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="7.36" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="567b5341-8f86-4c8b-bd93-40308e8fd1db" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="MinisterBuurt">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="bbe6e10e-2805-44ce-9cba-279dac4e168b" connectedTo="c8cd7c63-3a3d-4760-9b18-333d9a5f0e64" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_DeStrijp_1" id="39099775-f29c-4757-9409-4ade3408e07f">
        <geometry xsi:type="esdl:Point" lat="52.03030743866227" lon="4.299387931823731" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d1611125-9f5a-4f97-9786-1e9fd114730a" id="fd0f353b-455a-4f90-b8c2-b5643121a8e3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="4.07" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="651428d2-f28e-4873-bb37-02070c5e1d2a" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="De Strijp">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="35e0390a-790b-4f1f-8a9e-e3cfea3fe38b" connectedTo="c9db5954-0973-4a45-a040-008ca98e05d0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Stervoorde_2" id="92679c73-bdd4-4e61-ab3f-423a8a095563">
        <geometry xsi:type="esdl:Point" lat="52.03098063967937" lon="4.310073852539063" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="2dff1dbb-1a51-4e39-9555-7e16d38d7da1" id="ec81162a-89f8-474f-82e4-760db081a5a1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="6.02" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="a6520538-51ba-4bd4-b8b2-b67c1750505f" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Stervoorde">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="1417f744-930d-478b-bafd-d9d358b83e8e" connectedTo="4e5ce568-bbc1-484f-be68-e5a46a61b93b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Stervoorde_1" id="4c5e140c-ef2a-4c8e-a058-c5e6edfeabce">
        <geometry xsi:type="esdl:Point" lat="52.03211581843917" lon="4.3117690086364755" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="43b110a3-9963-4073-a743-de6aeb5a0d82" id="e6383ee7-aa2a-4c42-97c7-d7c7cdd7937b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="4.05" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="a288b889-0c3e-4592-993a-6006fd634c36" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Stervoorde">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="4b7d7685-af5d-4ccd-9f6c-d66a7a447a0e" connectedTo="8fa483ed-2b40-4410-bcd6-301b097ec23e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Stervoorde_3" id="4e4c1ec6-ea8c-42ac-bd65-8f41df0b51bc">
        <geometry xsi:type="esdl:Point" lat="52.029753030214444" lon="4.314708709716798" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="2218f77f-a63c-425a-8900-179dfdb53706" id="d937e02c-ce60-4352-90a2-ad6420f3147e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="10.5" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="864fc107-1b7b-44e8-b07b-b113f9b11056" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Stervoorde">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="016ba65e-e955-4fd2-bdfd-8ebecb31702f" connectedTo="265e54cf-0576-4d80-b6ac-9bdb881575d5" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Stervoorde_4" id="6125ea38-b2c2-47be-a8d4-2b9cc9f2c300">
        <geometry xsi:type="esdl:Point" lat="52.027627734152475" lon="4.3096661567688" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e1a13ea4-f7eb-4fb4-afe8-283e388d19b1" id="fc31e841-4b59-467b-a070-ba95631e977b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="2.98" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="4a6c3895-cc73-4178-9512-3dbbaea90023" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Stervoorde">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="922d3ae9-e869-4ee0-bfc9-a00f7fdbf283" connectedTo="e578607d-8bb3-40ec-b689-d991e1bf4300" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Stervoorde_5" id="7c52a9d7-622c-4741-9ce1-a823698b86b9">
        <geometry xsi:type="esdl:Point" lat="52.02601719409004" lon="4.312541484832765" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b299f0e5-4a02-431b-adcc-27e0967c5f5a" id="412c281b-228b-41f3-bdef-8bc997fd8e6b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.79" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="cbd88eb4-831b-4256-99d1-7bc82653dbb6" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Stervoorde">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="940d3006-7e6e-4a7c-814f-d130d992564c" connectedTo="c6164983-07e3-400b-b0db-936daaface70" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Hoekpolder" id="f26f4a87-a8e8-454d-b041-77b0de3c546f">
        <geometry xsi:type="esdl:Point" lat="52.025145894142554" lon="4.315352439880372" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba89be56-23e3-4048-ae31-ded09be9d50b" id="056aef7b-b808-46c6-baa1-010a0f4bf11a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.5" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="f32555ca-ee0b-437c-828c-29a4a66c31c8" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Hoekpolder">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="b3941ae3-a8c5-4a33-8369-68dfa6fa3fd1" connectedTo="d7d4264b-01c0-43ba-8864-9b01f445a9aa" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_20e0" id="20e06866-ab41-4ed5-beed-e8980c3f8ebd">
        <geometry xsi:type="esdl:Point" lat="52.02605679822976" lon="4.314301013946534" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7b52d17c-6205-4141-aa5e-8334b834c2d7 6a434cd0-16d6-4138-ba03-c8d397f4e0dd" id="8312aa67-8f5f-4eb0-a8ad-07dc0c7f3649" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e70641e4-96be-4c1b-9c04-b9257adda5d7" connectedTo="d21ba8f2-b54b-4c84-b013-67b1a7feb953 9c69135d-a0e5-4550-aa97-eb77b4d32306 6ad1c8fb-c41a-46a1-a2ac-44404baeef81" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_72e3" id="72e34ad8-feef-4604-a1aa-6c8e6fc0c7bc">
        <geometry xsi:type="esdl:Point" lat="52.028010558913955" lon="4.312477111816407" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d46d9aa4-b8a3-41f7-a7a3-79175568753c" id="ff808ae2-2e13-49e1-9054-da89bef0db88" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="460ad9bd-ae51-47e2-b9a7-e08bc4c86fb9" connectedTo="a40eada5-6329-4846-bce0-65cdfb1203c1 ba5d2033-f0b4-414e-b39c-0e72e122b747" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_aae6" id="aae6769c-d4f6-40ab-b830-a574fdebeba0">
        <geometry xsi:type="esdl:Point" lat="52.02896100622209" lon="4.313313961029054" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="eb254b90-29b7-4b12-a4f3-daf2c9f65026" id="621b064d-fbc6-451f-8a99-8d109330d3ab" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="bda0d0ea-4371-4e95-b38b-c7b0002e2d73" connectedTo="9ee69b64-1da1-4651-9f39-362cb02bd308 7178dbd9-04d5-46cc-83f7-42a255a66eed" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_839c" id="839c3742-38eb-43ac-b784-a5808b31159c">
        <geometry xsi:type="esdl:Point" lat="52.0312182376185" lon="4.311296939849854" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="2cfcc85a-c0dd-484d-984f-9a9a40436d78 008df1e4-53f0-4122-ad6b-901857d7a052" id="96fb6c73-14d6-4e0f-b18f-87b1f146a419" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ba5f246f-3f6a-4b80-ba1e-92b9c818e6d9" connectedTo="5c23cf16-c177-46e8-bd07-f6525567b205 bb7a2d77-d43b-46fc-b78c-9ed1a6d4de6d 5479c538-0675-4039-8872-73449b96d60c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_b466" id="b466938b-00ff-4d1c-ae5c-17667a640e8d">
        <geometry xsi:type="esdl:Point" lat="52.031799027265784" lon="4.310760498046876" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="957a1d73-aa94-4ad7-93c9-e11689c8a828 39b5089e-627f-48b2-a419-6c0a3f24d00a" id="d0436b36-2700-440b-a0f9-869730642d19" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4af9fa7e-41ee-4e79-97c4-e0dd2b1419d5" connectedTo="4e05723e-da3c-46eb-a06f-b5fba5f48e77 a17aca11-124a-4092-b206-b0b2c6d19666 267e82e0-ee82-4fe5-b249-50a115de72f6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_45b2" id="45b2b17b-2e1d-478b-b8e4-73a523c07180">
        <geometry xsi:type="esdl:Point" lat="52.03260420043414" lon="4.310138225555421" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="98cb1464-4bf2-48f5-9100-00faea463328" id="fc4fc28b-3d67-4ea5-b18a-78f06b4a485e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9b977589-8e44-475a-a0ed-f4420d88e233" connectedTo="dc272572-24e8-4732-9760-c72cd468ce81 25717110-9674-418f-a506-055fe48c695f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7975" id="7975feaf-7c8d-44b2-adc6-c756698428a7">
        <geometry xsi:type="esdl:Point" lat="52.03438608950049" lon="4.308958053588868" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="621e16d5-311b-4c95-87ca-4f220bbbcf1e 0e1a3e0d-166b-4639-8723-3a1b9f416fec" id="d90ca402-cd8d-429e-a04c-d5e817f99181" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="59de07d5-eb72-42f5-af24-0974da4122c9" connectedTo="170363ab-2815-4f4a-9d3b-0e9b223ba6cc 74b08f4f-d766-437b-a861-1af16ca2527d 0ba07947-d938-4f48-9ebc-a38f4a70a15d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_26ed" id="26ed9660-eca1-4483-bb1b-14f2d83626b0">
        <geometry xsi:type="esdl:Point" lat="52.03623389944942" lon="4.305782318115235" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3462ee9c-1d29-4970-a03b-019dc46e0f3b" id="46b318b8-4c66-4f46-9995-34803863a16b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7fb2c2ec-9dff-4709-b221-9a6bcc1b205b" connectedTo="5f744428-c765-473a-a226-9f64d27ac881 c4747870-0077-46b4-803f-8d8111839843" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_75c2" id="75c21861-e461-423e-ad15-09b9200da342">
        <geometry xsi:type="esdl:Point" lat="52.03721056814132" lon="4.305396080017091" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3648f104-f01b-4bf3-a8c2-433043926a3a" id="47fad4e7-aff2-4858-8e0b-b827e3cd4986" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="32f70da1-9ba0-4948-88dd-d3a76a439e75" connectedTo="cef325c2-be23-452e-a81c-4e1496e4e7d4 89801693-28b8-4a00-8635-a38a020bfc24" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_50da" id="50daf054-b881-4e62-adfc-4d2750a2a1c3">
        <geometry xsi:type="esdl:Point" lat="52.0372369642963" lon="4.297864437103272" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f0e6c142-ec72-41ad-8d2b-68b3b3303dd4 08fee2ea-5e6e-4032-9062-a5208ced4a3c" id="a7a34f2e-4ac0-40b4-9d15-ef4d784b5e1f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="91e38e17-04b1-4174-a200-899ca54224be" connectedTo="129b7254-de08-4555-ba45-ffaeaac817db 4d2b3f47-812e-488b-b176-a3abf9a2f46a daf67778-22d9-4e57-9467-3477c15d08f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a9fd" id="a9fdfa02-e51d-41ef-bc2b-2e59f2933df4">
        <geometry xsi:type="esdl:Point" lat="52.03907950206215" lon="4.304720163345338" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="caad53c5-4aa5-47ff-98c6-bdd1da60c36b b21a2b68-30a7-478b-a633-5a398f2a1834" id="125e86b9-1b17-45cb-b4ae-eddbe3a4809d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a572f6a9-36cb-498c-b44d-b092d7288aa3" connectedTo="47d38527-b3a2-4cb9-8336-023ad666841f 3f1c3554-4d86-42b3-ac2d-af6267f84a26 4ebbb86f-e759-459d-bf89-6d816892becd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe6_ret" diameter="DN400" length="826.7" name="Pipe6" id="Pipe6" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0443961581157" lon="4.3101945519447336"/>
          <point xsi:type="esdl:Point" lat="52.0429386442008" lon="4.305803775787354"/>
          <point xsi:type="esdl:Point" lat="52.04064241331808" lon="4.303615093231202"/>
          <point xsi:type="esdl:Point" lat="52.03907950206215" lon="4.304720163345338"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c709f60d-55ac-4791-8b4e-e8cd6733fc33" id="ec75d694-eb55-4409-880c-6c342f3abbfe" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="caad53c5-4aa5-47ff-98c6-bdd1da60c36b" connectedTo="125e86b9-1b17-45cb-b4ae-eddbe3a4809d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3e3ef2d7-1c21-44fa-b191-5fe640d6892e">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe7_ret" diameter="DN400" length="212.9" name="Pipe7" id="Pipe7" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03907950206215" lon="4.304720163345338"/>
          <point xsi:type="esdl:Point" lat="52.03721056814132" lon="4.305396080017091"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a572f6a9-36cb-498c-b44d-b092d7288aa3" id="47d38527-b3a2-4cb9-8336-023ad666841f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="3648f104-f01b-4bf3-a8c2-433043926a3a" connectedTo="47fad4e7-aff2-4858-8e0b-b827e3cd4986" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="d3a85ffd-efde-488b-b855-3fa05404a597">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe8_ret" diameter="DN400" length="111.8" name="Pipe8" id="Pipe8" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03721056814132" lon="4.305396080017091"/>
          <point xsi:type="esdl:Point" lat="52.03623389944942" lon="4.305782318115235"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="32f70da1-9ba0-4948-88dd-d3a76a439e75" id="cef325c2-be23-452e-a81c-4e1496e4e7d4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="3462ee9c-1d29-4970-a03b-019dc46e0f3b" connectedTo="46b318b8-4c66-4f46-9995-34803863a16b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="8939ef13-c7ba-492d-a1e7-ff558437dd9a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe9_ret" diameter="DN400" length="387.2" name="Pipe9" id="Pipe9" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03623389944942" lon="4.305782318115235"/>
          <point xsi:type="esdl:Point" lat="52.034336566309356" lon="4.307069778442384"/>
          <point xsi:type="esdl:Point" lat="52.03470613481827" lon="4.308700561523438"/>
          <point xsi:type="esdl:Point" lat="52.03438608950049" lon="4.308958053588868"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7fb2c2ec-9dff-4709-b221-9a6bcc1b205b" id="5f744428-c765-473a-a226-9f64d27ac881" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="621e16d5-311b-4c95-87ca-4f220bbbcf1e" connectedTo="d90ca402-cd8d-429e-a04c-d5e817f99181" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="816b4563-f0b8-4d05-aa12-74ea573c366a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe10_ret" diameter="DN400" length="214.0" name="Pipe10" id="Pipe10" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03438608950049" lon="4.308958053588868"/>
          <point xsi:type="esdl:Point" lat="52.03260420043414" lon="4.310138225555421"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="59de07d5-eb72-42f5-af24-0974da4122c9" id="170363ab-2815-4f4a-9d3b-0e9b223ba6cc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="98cb1464-4bf2-48f5-9100-00faea463328" connectedTo="fc4fc28b-3d67-4ea5-b18a-78f06b4a485e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="29fbf25c-b0c3-441b-bb36-45b77584edf3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe11_ret" diameter="DN400" length="99.1" name="Pipe11" id="Pipe11" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03260420043414" lon="4.310138225555421"/>
          <point xsi:type="esdl:Point" lat="52.031799027265784" lon="4.310760498046876"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9b977589-8e44-475a-a0ed-f4420d88e233" id="dc272572-24e8-4732-9760-c72cd468ce81" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="957a1d73-aa94-4ad7-93c9-e11689c8a828" connectedTo="d0436b36-2700-440b-a0f9-869730642d19" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="d7bd5cc1-ea96-4e94-97ee-b3b1104c54ad">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe12_ret" diameter="DN400" length="74.3" name="Pipe12" id="Pipe12" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.031799027265784" lon="4.310760498046876"/>
          <point xsi:type="esdl:Point" lat="52.0312182376185" lon="4.311296939849854"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4af9fa7e-41ee-4e79-97c4-e0dd2b1419d5" id="4e05723e-da3c-46eb-a06f-b5fba5f48e77" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2cfcc85a-c0dd-484d-984f-9a9a40436d78" connectedTo="96fb6c73-14d6-4e0f-b18f-87b1f146a419" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2204b61b-dd02-4977-bdde-228190566d18">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe13_ret" diameter="DN400" length="286.4" name="Pipe13" id="Pipe13" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0312182376185" lon="4.311296939849854"/>
          <point xsi:type="esdl:Point" lat="52.02896100622209" lon="4.313313961029054"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba5f246f-3f6a-4b80-ba1e-92b9c818e6d9" id="5c23cf16-c177-46e8-bd07-f6525567b205" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="eb254b90-29b7-4b12-a4f3-daf2c9f65026" connectedTo="621b064d-fbc6-451f-8a99-8d109330d3ab" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="5e252139-9554-4f1a-8a9d-30a29586a3fd">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe15_ret" diameter="DN200" length="142.5" name="Pipe15" id="Pipe15" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02896100622209" lon="4.313313961029054"/>
          <point xsi:type="esdl:Point" lat="52.02861392516555" lon="4.3134427070617685"/>
          <point xsi:type="esdl:Point" lat="52.02832351059946" lon="4.313335418701173"/>
          <point xsi:type="esdl:Point" lat="52.028165101859365" lon="4.313099384307862"/>
          <point xsi:type="esdl:Point" lat="52.028010558913955" lon="4.312477111816407"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="bda0d0ea-4371-4e95-b38b-c7b0002e2d73" id="9ee69b64-1da1-4651-9f39-362cb02bd308" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d46d9aa4-b8a3-41f7-a7a3-79175568753c" connectedTo="ff808ae2-2e13-49e1-9054-da89bef0db88" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="11fdcd62-d6c9-486e-939e-a51193021f93">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe16_ret" diameter="DN200" length="250.5" name="Pipe16" id="Pipe16" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.028010558913955" lon="4.312477111816407"/>
          <point xsi:type="esdl:Point" lat="52.02605679822976" lon="4.314301013946534"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="460ad9bd-ae51-47e2-b9a7-e08bc4c86fb9" id="a40eada5-6329-4846-bce0-65cdfb1203c1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7b52d17c-6205-4141-aa5e-8334b834c2d7" connectedTo="8312aa67-8f5f-4eb0-a8ad-07dc0c7f3649" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="6cd35c91-bece-4eab-b04f-001f5f0000c9">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe17_ret" diameter="DN200" length="164.5" name="Pipe17" id="Pipe17" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02605679822976" lon="4.314301013946534"/>
          <point xsi:type="esdl:Point" lat="52.02560377928665" lon="4.312777519226075"/>
          <point xsi:type="esdl:Point" lat="52.02601719409004" lon="4.312541484832765"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e70641e4-96be-4c1b-9c04-b9257adda5d7" id="d21ba8f2-b54b-4c84-b013-67b1a7feb953" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b299f0e5-4a02-431b-adcc-27e0967c5f5a" connectedTo="412c281b-228b-41f3-bdef-8bc997fd8e6b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="fbd97b94-bbe8-405b-9cf0-5e79123c7f30">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe19_ret" diameter="DN200" length="70.8" name="Pipe19" id="Pipe19" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03438608950049" lon="4.308958053588868"/>
          <point xsi:type="esdl:Point" lat="52.03474245879195" lon="4.309816360473634"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="59de07d5-eb72-42f5-af24-0974da4122c9" id="74b08f4f-d766-437b-a861-1af16ca2527d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f9aae992-24b9-4fb8-adbb-c8a0846c999d" connectedTo="66455895-fbde-4174-a263-86f9586c2fc8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4b0da46a-893c-4746-8017-796453417db9">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe21_ret" diameter="DN200" length="571.4" name="Pipe21" id="Pipe21" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03721056814132" lon="4.305396080017091"/>
          <point xsi:type="esdl:Point" lat="52.036938447097874" lon="4.303593635559083"/>
          <point xsi:type="esdl:Point" lat="52.036938447097874" lon="4.3023061752319345"/>
          <point xsi:type="esdl:Point" lat="52.0373607862784" lon="4.300847053527833"/>
          <point xsi:type="esdl:Point" lat="52.03783591308721" lon="4.300374984741212"/>
          <point xsi:type="esdl:Point" lat="52.0372369642963" lon="4.297864437103272"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="32f70da1-9ba0-4948-88dd-d3a76a439e75" id="89801693-28b8-4a00-8635-a38a020bfc24" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f0e6c142-ec72-41ad-8d2b-68b3b3303dd4" connectedTo="a7a34f2e-4ac0-40b4-9d15-ef4d784b5e1f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="f4033bcc-ceb8-4909-b054-9f4c023a08f5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe22_ret" diameter="DN200" length="88.6" name="Pipe22" id="Pipe22" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0372369642963" lon="4.297864437103272"/>
          <point xsi:type="esdl:Point" lat="52.036880614888744" lon="4.296705722808839"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="91e38e17-04b1-4174-a200-899ca54224be" id="129b7254-de08-4555-ba45-ffaeaac817db" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="589c1244-bdbf-42a0-a6cc-b99c18916900" connectedTo="414973d8-31e5-4237-9ebf-b319c6ec404b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="43b30ec4-c3af-44e7-b7d6-8a187c2debe5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe24_ret" diameter="DN200" length="69.4" name="Pipe24" id="Pipe24" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03287207234976" lon="4.320539832115174"/>
          <point xsi:type="esdl:Point" lat="52.033208179577564" lon="4.3213941156864175"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4a82116b-fede-476f-9135-24d51f45f1b3" id="ac87490f-88d5-43a5-89d8-300f719ebfec" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="27ad4ddb-4499-4264-b6d2-2c21b71e68c3" connectedTo="504787dd-d2c9-44ed-afb3-88071b846d9b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2e6036e6-23c1-42d8-bd99-6e18c4cdfa7d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe25_ret" diameter="DN300" length="410.3" name="Pipe25" id="Pipe25" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03766978308298" lon="4.3182599544525155"/>
          <point xsi:type="esdl:Point" lat="52.0364477896304" lon="4.319493770599366"/>
          <point xsi:type="esdl:Point" lat="52.035827464917006" lon="4.3182599544525155"/>
          <point xsi:type="esdl:Point" lat="52.03473314368742" lon="4.319311380386353"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="39f0b02a-1dbf-42e7-99a6-143e5868754a" id="c0ce56e8-260b-4dea-a690-86d623612ee9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b58b1d37-d79b-4e73-a9ba-3977125cbc31" connectedTo="2d464b69-7340-4d13-b04e-8d642145a6fe" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="56dc2574-68f0-470f-8fe2-262dd840453b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe26_ret" diameter="DN300" length="255.8" name="Pipe26" id="Pipe26" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03473314368742" lon="4.319311380386353"/>
          <point xsi:type="esdl:Point" lat="52.03360346408251" lon="4.320352077484132"/>
          <point xsi:type="esdl:Point" lat="52.03346487283161" lon="4.319922924041749"/>
          <point xsi:type="esdl:Point" lat="52.03287207234976" lon="4.320539832115174"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="72c1d1a9-8f49-4a26-b861-ec067835ef38" id="d4a42b33-e4ca-4743-a240-9b3b0847ff60" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e4b21d04-196b-460f-a4ff-b3b7de3c9ccb" connectedTo="b2547ccb-93f2-4c84-839d-0e34211aeece" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4963d088-f700-450b-be62-deb8d0a46d86">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe27_ret" diameter="DN150" length="34.6" name="Pipe27" id="Pipe27" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03473314368742" lon="4.319311380386353"/>
          <point xsi:type="esdl:Point" lat="52.03491765821515" lon="4.319719076156617"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="72c1d1a9-8f49-4a26-b861-ec067835ef38" id="c8146f79-e5f6-430a-97ce-253f14b77e21" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="89bfd689-d399-42f9-a0c7-145a09816032" connectedTo="8c696b24-1afb-4af2-ab02-4c911a43f9a2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="cfe1dc4e-9f41-401a-ad50-c745013e06ff">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe28_ret" diameter="DN150" length="22.8" name="Pipe28" id="Pipe28" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03766978308298" lon="4.3182599544525155"/>
          <point xsi:type="esdl:Point" lat="52.037722645629316" lon="4.318581819534303"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="39f0b02a-1dbf-42e7-99a6-143e5868754a" id="767d00ad-0379-4618-ad42-322481c41a4a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="eb08503b-b0a8-4713-b941-b626531b19e5" connectedTo="bb93e16a-1131-425f-b438-0a1afd02b5b7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c2f75df9-8e03-4655-81b9-fd4baa405c49">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe29_ret" diameter="DN400" length="23.0" name="Pipe29" id="Pipe29" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03756425102864" lon="4.317970275878907"/>
          <point xsi:type="esdl:Point" lat="52.03766978308298" lon="4.3182599544525155"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="57581db4-cc0f-4260-8538-bf2599cdece2" id="47b7daf3-d849-4176-8825-84e4f1b46b58" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="630b935b-252e-489c-a6a6-d4d56bf1668e" connectedTo="0306fc7e-d183-4db6-86a6-f523ab77e17e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="4dc63595-f9f7-4924-8093-ad9cf754d32b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe30_ret" diameter="DN300" length="39.5" name="Pipe30" id="Pipe30" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03879819320885" lon="4.316698908805848"/>
          <point xsi:type="esdl:Point" lat="52.038644953041" lon="4.316178560256959"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba4ce072-68d5-4603-aa4d-7ecf059f5020" id="c57ca1fe-b04f-41c6-abd1-1dd286045e4f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="48b9c98f-4bc7-44fc-baee-1e9681c16c1f" connectedTo="9da9e597-f6fb-4831-b124-6de0044f5a3e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="0aaaffe8-f85e-46fb-8f3f-d556cfc46e56">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe31_ret" diameter="DN150" length="21.6" name="Pipe31" id="Pipe31" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04164900734322" lon="4.312788248062135"/>
          <point xsi:type="esdl:Point" lat="52.04172687268464" lon="4.313077926635743"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d9e086ab-5ad3-4235-98ad-0f109001a4d4" id="d9094d66-9b87-4773-bbb0-0d4473d07636" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e415e84f-84b0-4133-925f-50c1903f9eb8" connectedTo="59a43508-4328-4e0a-94e1-fd6036172036" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="a3b3b0cc-c9c0-4448-909a-ed4a251108e1">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe32_ret" diameter="DN300" length="77.5" name="Pipe32" id="Pipe32" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.031799027265784" lon="4.310760498046876"/>
          <point xsi:type="esdl:Point" lat="52.03211581843917" lon="4.3117690086364755"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4af9fa7e-41ee-4e79-97c4-e0dd2b1419d5" id="a17aca11-124a-4092-b206-b0b2c6d19666" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="43b110a3-9963-4073-a743-de6aeb5a0d82" connectedTo="e6383ee7-aa2a-4c42-97c7-d7c7cdd7937b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="5f43418e-9e19-4fef-ac64-80e94fc29b8a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe33_ret" diameter="DN300" length="87.7" name="Pipe33" id="Pipe33" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0312182376185" lon="4.311296939849854"/>
          <point xsi:type="esdl:Point" lat="52.03098063967937" lon="4.310073852539063"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba5f246f-3f6a-4b80-ba1e-92b9c818e6d9" id="bb7a2d77-d43b-46fc-b78c-9ed1a6d4de6d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2dff1dbb-1a51-4e39-9555-7e16d38d7da1" connectedTo="ec81162a-89f8-474f-82e4-760db081a5a1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="59e145b3-f803-4ebd-825b-33bd0d2ecf34">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe35_ret" diameter="DN150" length="124.2" name="Pipe35" id="Pipe35" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02605679822976" lon="4.314301013946534"/>
          <point xsi:type="esdl:Point" lat="52.025145894142554" lon="4.315352439880372"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e70641e4-96be-4c1b-9c04-b9257adda5d7" id="9c69135d-a0e5-4550-aa97-eb77b4d32306" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ba89be56-23e3-4048-ae31-ded09be9d50b" connectedTo="056aef7b-b808-46c6-baa1-010a0f4bf11a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="27fa6084-42c0-41ab-a35e-92238451267a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe36_ret" diameter="DN150" length="42.8" name="Pipe36" id="Pipe36" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0372369642963" lon="4.297864437103272"/>
          <point xsi:type="esdl:Point" lat="52.036880614888744" lon="4.298100471496583"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="91e38e17-04b1-4174-a200-899ca54224be" id="4d2b3f47-812e-488b-b176-a3abf9a2f46a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b7c5f7da-761d-4d69-90c8-cc18e551c6b5" connectedTo="856149d8-c74a-496e-a6fd-b9ed82e843af" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="ffff3613-fd49-4aef-847b-f6bfee4d37c2">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe37_ret" diameter="DN150" length="21.6" name="Pipe37" id="Pipe37" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03907950206215" lon="4.304720163345338"/>
          <point xsi:type="esdl:Point" lat="52.03904506381387" lon="4.30440902709961"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a572f6a9-36cb-498c-b44d-b092d7288aa3" id="3f1c3554-4d86-42b3-ac2d-af6267f84a26" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d2065857-7f90-4645-8e5b-af7b096dc84c" connectedTo="49eb6e79-0f68-44c0-a9ca-3398028e5b5e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="db7a5d61-49b2-40ec-975e-8dff9f1088d2">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_HuisTeLande" id="0a70ec45-35ef-4d7d-a180-58eabd74ae5f">
        <geometry xsi:type="esdl:Point" lat="52.04313255137357" lon="4.319053888320924" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="878fc721-fcd7-45e7-a85e-de03b6c34cf0" id="dcb6b734-6182-4704-bf62-97c6bc0a6cea" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="5.84" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="38ad48bb-6bb1-43a1-b78b-984cb7e1a123" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Huis Te Lande">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="3fc2139b-301d-4c12-8ec7-6a263075334c" connectedTo="dc7d95d4-a8b1-4500-a48e-1f83b876b49a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Stationskwartier" id="5c1560f1-7f7b-4d98-826d-54543e15c87e">
        <geometry xsi:type="esdl:Point" lat="52.03957592470637" lon="4.321918487548829" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="285dee2d-f174-4a51-8272-7f97c4606e6d" id="799464bf-4173-450b-b2e4-0c143c0e4394" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="4.81" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="540d94fb-bd7e-4b89-b6b4-d3b26da029c2" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Stationskwartier">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="0ecf3ec6-0c44-4231-a553-ca5173d09dfa" connectedTo="d229c40e-ebfe-499f-94b2-f6d215b9d794" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_TeWerve_1" id="ca11a5d4-8c1f-4524-b41f-7c3e768ed1b6">
        <geometry xsi:type="esdl:Point" lat="52.04157528352519" lon="4.324407577514649" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e2ef19c1-e80c-4a2a-b13d-62bd6c2ad5ed" id="39af40d7-19ef-4aed-ba38-fd5d034b4cd7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="28d9df50-da78-47a0-b2f8-272bd64644c4" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Te Werve">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="b2a780bf-6ceb-4c2f-8242-42b3e8d6e1be" connectedTo="89fa68ff-861e-4c57-a53b-c44cf7536af7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_TeWerve_2" id="d94b2641-6b91-4c7d-a3e3-9ec0e7cd1469">
        <geometry xsi:type="esdl:Point" lat="52.0429015415213" lon="4.326370954513551" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="eef6b3f9-b113-4b26-801b-0845350e6dbe" id="c480062c-4cba-4b85-a992-db4f6136ffb4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="834bdeee-16be-4893-af7d-9460fbb39853" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Te Werve">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="8696c727-7e34-4217-b1c6-2b3638bbbb8f" connectedTo="dd39438a-9e0a-4d27-a533-fa940877ff36" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_TeWerve_3" id="704fb7af-842f-4823-9ab8-b5a3402e2c58">
        <geometry xsi:type="esdl:Point" lat="52.04429374016349" lon="4.328430891036988" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="240f2f27-935f-4a2b-99b0-6fb92c41b68e" id="eb42ca4e-4b23-4096-bdb4-75f5f6e4d767" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="d248ebbe-61fb-4444-8d4c-b37b115da7af" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Te Werve">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="e4f770db-2bcb-444c-9b76-7e894a06ab9b" connectedTo="359cd445-80bc-484c-b4db-e2c988cc6176" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_TeWerve_4" id="5eb81d9e-6c44-408d-9b1a-6ea8135e94c9">
        <geometry xsi:type="esdl:Point" lat="52.04642484175937" lon="4.330179691314698" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e481b04a-fffe-4181-913d-17c120558ed9" id="612572ec-f6f1-43e9-a5d3-d783cef32723" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="d0fad1cf-b411-4289-8895-635de4ff449f" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Te Werve">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="483a6cd3-01b3-405f-924b-995cad603382" connectedTo="3741698a-602c-4d08-9a22-f5b42d8eb53a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Rembrandtkwartier_1" id="68e28dbe-8780-4cad-8a21-039732ea6b04">
        <geometry xsi:type="esdl:Point" lat="52.04959161616739" lon="4.334771633148194" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9b7e34bb-1cc6-4f25-891d-cca582c3783c" id="6bc62478-cdab-4c7c-bcb4-c236c475230a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="24b145e5-c510-4eff-9e7f-34b0f940b4f0" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Rembrandtkwartier">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="c735b1aa-e309-4945-8886-195af354f0ac" connectedTo="c617234b-aa4d-420e-a4eb-615713be15d3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_fea2" id="fea26cb1-d772-493b-836c-20b9e3362c4a">
        <geometry xsi:type="esdl:Point" lat="52.04850966015021" lon="4.337850809097291" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="8cf61346-445d-44e1-bd5f-02b51b9af83c" id="f470c680-16ff-449c-9c6d-58e59dd3b478" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="9.02" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="baa1cc77-6753-4837-86f6-d183830f192c" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Rembrandtkwartier">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="827b6304-b33d-4a73-83ec-2111e6a3c837" connectedTo="08705ddf-393d-4827-aba5-e45902bd65bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_2bbd" id="2bbdbeb9-abd4-4389-bb5d-4f10c5b7a0d7">
        <geometry xsi:type="esdl:Point" lat="52.0511881131729" lon="4.337314367294312" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="be3c8912-b660-4c3b-8a34-c929386c2107" id="34ea2553-c243-4d3d-b607-9aecd7fdf690" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="3f74d3ec-f91c-4960-9927-087a710541ed" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Rembrandtkwartier">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="a0c2f54d-b8af-44b3-9107-1ec0571d61ab" connectedTo="3240dcc5-5056-409d-8d83-c8e3989139b4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_Rembrandtkwartier_2" id="39540a79-e5fc-4e6c-aee4-75357b5fba2f">
        <geometry xsi:type="esdl:Point" lat="52.05197974342533" lon="4.332336187362672" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5802017c-c2da-48c5-88df-b23fc4090720" id="02fddd61-e3fb-4ed9-aa92-08694de98345" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="3.0" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="1330bedf-7b9b-40d3-95f1-e6ce8724d5ba" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Rembrandtkwartier">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="df8b4206-fbd9-4b20-b477-13129693dd70" connectedTo="9f295aaf-a774-4a72-afcc-64d131904643" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:HeatingDemand" name="HeatingDemand_WelGelegen" id="54fbc660-8fa1-4215-a7ba-dd01b04a988f">
        <geometry xsi:type="esdl:Point" lat="52.054987810473264" lon="4.3325185775756845" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="78aa70aa-e167-4cc8-93e9-d4c1ca8d52df" id="3e8bbc50-eb90-4c5d-9035-f77f38b544c4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75">
          <profile xsi:type="esdl:InfluxDBProfile" endDate="2019-12-31T22:00:00.000000+0000" multiplier="9.02" startDate="2018-12-31T23:00:00.000000+0000" filters="" id="33aee60b-3e0c-4dc9-a968-48cd10537af5" database="energy_profiles" measurement="Warmteprofielen_Rijswijk_normalized" host="wu-profiles.esdl-beta.hesi.energy" port="443" field="Rembrandtkwartier">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port xsi:type="esdl:OutPort" name="Out" id="8b9f02d3-cbb2-4584-b8f7-16e181318b58" connectedTo="3b2d411f-d681-4539-a42f-4e6bc5df4f41" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a7f5" id="a7f51bba-1d31-4707-a1ec-839046c04441">
        <geometry xsi:type="esdl:Point" lat="52.042894338333426" lon="4.318453073501588" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ddf5f2e9-be69-42e8-b140-e2bed714c18a f039081a-6763-4917-bd96-9ec83518efaa" id="034e5cc7-68f3-4c88-8be9-31902d03d441" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="679d18da-f447-41dd-916f-e5d3b29d1184" connectedTo="610382e7-e72b-4f11-b1af-6c51ec4611d7 14d9687b-c97e-4d19-ad3e-b0776928f0bd e3603d73-e225-49bc-aef9-dc6b2c67a3f1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1b08" id="1b080109-62f7-491c-adfd-386c1616cac2">
        <geometry xsi:type="esdl:Point" lat="52.03995144351459" lon="4.321660995483399" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ced5dcbb-5dcb-47cd-816b-ad8a1bf3e5c8 b3330cdc-ce06-4264-9d30-b5fc3d2e657c" id="eede8573-c3f5-4156-baea-f8ef53a002a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4cac6a54-6b27-456e-8d17-f5910fb4744c" connectedTo="dd138595-545f-4546-ad23-be3b0b811ebd f219ef2b-e6b8-404d-ad1f-610c680fae0a c72bf1cd-6be8-47ab-9288-d4017b440783" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_ae77" id="ae779c26-d5c1-4ff3-b405-bed65e4651b6">
        <geometry xsi:type="esdl:Point" lat="52.04102040978031" lon="4.323420524597169" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7cf79d2a-59dc-4bd6-9d01-69932c059e64 c56dbff6-43d0-424e-82af-647304a719ef" id="0f1d6205-a978-46a7-8c51-3aa7e22d6ac3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="3f8246ac-5bf4-4dff-b00d-f5e2e956c2b9" connectedTo="7461f449-90e0-4338-b378-a435b9702ace 00fcc715-3f27-4296-ac99-519c7eaa8a2e 5900a9f1-f801-437d-aabd-7d762c8135d1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a659" id="a659c876-1c54-4376-a15e-aff2fca735e3">
        <geometry xsi:type="esdl:Point" lat="52.04235988080374" lon="4.327293634414674" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9bd8f8d0-fc4f-4c9c-bb21-304fb4b7b0c2 7f7fd672-8ea1-45cb-baf0-ac9821ea412f" id="5de6a1ad-fd67-4b43-8fdf-70d74d6fd0bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7fc22ee8-6e17-4a43-857c-d5d35bf111cb" connectedTo="b5504e49-bf61-4362-817f-87b7ac726de9 5c60b4cd-ab2c-4ec2-a694-672ea9d24418 d6506a36-4356-4ac8-8964-0afab3b35e21" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_d25d" id="d25d304d-3642-4c67-9929-752bd0ef3497">
        <geometry xsi:type="esdl:Point" lat="52.04362013462266" lon="4.32937502861023" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a96941c6-8936-44f9-b823-e7bc445b3dec 7611d55e-8f95-482e-9014-c57e8f3e53dd" id="599d0c0b-c7e2-4314-bc5c-da112c9da4c1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b85c3b55-c290-4987-93b3-6c03bde6e8b4" connectedTo="9a8757d1-ad73-4684-9296-60a54013b89f c73d289c-dda5-4d21-b03f-dc19eae8aaf1 55a260e8-8191-460a-91aa-b04cdb701a77" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a80f" id="a80ffc0c-5c63-4122-9284-6582c2c81a09">
        <geometry xsi:type="esdl:Point" lat="52.04556653011963" lon="4.3322181701660165" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="05fdb753-212b-4c7c-bc6b-bd1536f19337" id="45013b48-72c7-41fc-9bfe-edd02d2e380a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c527ea3d-4e8b-40e6-883e-4c60cbcfbe67" connectedTo="61d57ac4-ea0f-456c-9272-ae2e9b8118a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_4e62" id="4e621b56-23a9-400a-a334-3b84b5734e79">
        <geometry xsi:type="esdl:Point" lat="52.04658258087284" lon="4.330555200576783" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="37c19345-8501-4de1-a138-999a8e0880f6 4880facf-4871-4773-a1ac-0e48df8eca65" id="90cc9947-a05b-41c5-8b9e-e678bf033c61" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="aae35ef1-f784-489b-9e36-f0b903ca948d" connectedTo="1502fc4c-0c2e-4889-b09b-777549e2739d 44631aef-3734-44e2-b2e3-738189062c62 e2b27087-2c20-4082-9c83-6ea731ca7d21" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_31ec" id="31ec673d-7e80-4994-8de9-0c9269e1248c">
        <geometry xsi:type="esdl:Point" lat="52.050039619503906" lon="4.334267377853394" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="93835c29-ca52-4ed0-beaa-75161e840249 a82e0ed6-fc80-4319-820c-7de68a337cd3" id="29b4f6b2-45e0-452d-82ed-0a1507fa40f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a20b67c3-c88a-49a1-8428-01424efffc51" connectedTo="cc22d0e4-4d26-48ee-ab92-1ca8b5571536 50a51359-11eb-42a0-b56c-3abedae163e1 f5b9c2e0-4e88-463c-89dd-f3197e9eae02" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1085" id="1085e540-3ee2-4724-a8bb-d747f38cef3d">
        <geometry xsi:type="esdl:Point" lat="52.0507784938338" lon="4.336756467819215" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ea278421-bea3-4098-9add-271bdc3bae8d a84b4b4b-316c-44e9-8aaa-8af5b97c4506" id="49caaf6b-e676-4e65-9ce5-027c2406c9b7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d906ef7e-682b-484e-82ac-584a0115c91e" connectedTo="5f245a7a-bb9c-4c6d-af8c-05a27a76f17c 566725ca-a91e-4c6c-b84e-b3809e241f82 edd5a51a-c5a8-4e2a-bc61-f8b0d031aa11" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_cf81" id="cf81608a-f000-45e8-aa70-6068585ed016">
        <geometry xsi:type="esdl:Point" lat="52.05278394828437" lon="4.334599971771241" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ead81158-fba7-4d8e-a6b0-d19ea45dd8fb" id="88ddaaa8-e23c-4135-92b3-a379f53f1a86" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9a8102ad-8172-41f1-9939-ff995a85d39e" connectedTo="cf93b38e-49d4-41e0-8f2e-cae0e61e47bb 2e30acdb-3a7d-4c27-8917-127592c0c3a3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_6d6e" id="6d6e3237-3971-480d-b128-847a9dc8a58f">
        <geometry xsi:type="esdl:Point" lat="52.051490968223646" lon="4.3359625339508066" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="00d04a71-fa13-4bcc-919e-93a0ded8c13b" id="a26aefdd-b3f6-4f32-a185-ed07a8a47fe6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="022586f7-6c01-4622-ad59-cdd682be2e9b" connectedTo="74f1cfd7-08c7-416e-a9d9-c77fb32e881d b58cd7f3-5157-4dae-b01c-030fa56958bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe38_ret" diameter="DN400" length="785.2" name="Pipe38" id="Pipe38" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0443961581157" lon="4.3101945519447336"/>
          <point xsi:type="esdl:Point" lat="52.04530641967745" lon="4.313077926635743"/>
          <point xsi:type="esdl:Point" lat="52.0446994137633" lon="4.31440830230713"/>
          <point xsi:type="esdl:Point" lat="52.04427714391706" lon="4.314751625061036"/>
          <point xsi:type="esdl:Point" lat="52.04488415556644" lon="4.31640386581421"/>
          <point xsi:type="esdl:Point" lat="52.0447917847603" lon="4.316725730895997"/>
          <point xsi:type="esdl:Point" lat="52.042894338333426" lon="4.318453073501588"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c709f60d-55ac-4791-8b4e-e8cd6733fc33" id="cea9d279-8464-4774-b382-34e7b3e69c5a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ddf5f2e9-be69-42e8-b140-e2bed714c18a" connectedTo="034e5cc7-68f3-4c88-8be9-31902d03d441" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c45789a4-72b8-45f7-bb23-e0391fcb2619">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe39_ret" diameter="DN400" length="397.3" name="Pipe39" id="Pipe39" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.042894338333426" lon="4.318453073501588"/>
          <point xsi:type="esdl:Point" lat="52.041387127570374" lon="4.3196439743042"/>
          <point xsi:type="esdl:Point" lat="52.03995144351459" lon="4.321660995483399"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="679d18da-f447-41dd-916f-e5d3b29d1184" id="610382e7-e72b-4f11-b1af-6c51ec4611d7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ced5dcbb-5dcb-47cd-816b-ad8a1bf3e5c8" connectedTo="eede8573-c3f5-4156-baea-f8ef53a002a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="ad9c8c29-7359-40d6-bf4a-cc0a9eb3d704">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe40_ret" diameter="DN400" length="169.1" name="Pipe40" id="Pipe40" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03995144351459" lon="4.321660995483399"/>
          <point xsi:type="esdl:Point" lat="52.04102040978031" lon="4.323420524597169"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4cac6a54-6b27-456e-8d17-f5910fb4744c" id="dd138595-545f-4546-ad23-be3b0b811ebd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7cf79d2a-59dc-4bd6-9d01-69932c059e64" connectedTo="0f1d6205-a978-46a7-8c51-3aa7e22d6ac3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="be97e60c-f70f-4bf8-a0eb-5e43589f278d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe41_ret" diameter="DN400" length="381.2" name="Pipe41" id="Pipe41" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04102040978031" lon="4.323420524597169"/>
          <point xsi:type="esdl:Point" lat="52.0404633386421" lon="4.324536323547364"/>
          <point xsi:type="esdl:Point" lat="52.04235988080374" lon="4.327293634414674"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3f8246ac-5bf4-4dff-b00d-f5e2e956c2b9" id="7461f449-90e0-4338-b378-a435b9702ace" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9bd8f8d0-fc4f-4c9c-bb21-304fb4b7b0c2" connectedTo="5de6a1ad-fd67-4b43-8fdf-70d74d6fd0bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b487cdc6-72b9-4aae-986f-4232eaca1607">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe42_ret" diameter="DN400" length="200.5" name="Pipe42" id="Pipe42" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04235988080374" lon="4.327293634414674"/>
          <point xsi:type="esdl:Point" lat="52.04306309588502" lon="4.328269958496095"/>
          <point xsi:type="esdl:Point" lat="52.04362013462266" lon="4.32937502861023"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7fc22ee8-6e17-4a43-857c-d5d35bf111cb" id="b5504e49-bf61-4362-817f-87b7ac726de9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a96941c6-8936-44f9-b823-e7bc445b3dec" connectedTo="599d0c0b-c7e2-4314-bc5c-da112c9da4c1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="e216f2ac-9e4e-49a3-8150-302634fec259">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe43_ret" diameter="DN400" length="290.9" name="Pipe43" id="Pipe43" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04362013462266" lon="4.32937502861023"/>
          <point xsi:type="esdl:Point" lat="52.04556653011963" lon="4.3322181701660165"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b85c3b55-c290-4987-93b3-6c03bde6e8b4" id="9a8757d1-ad73-4684-9296-60a54013b89f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="05fdb753-212b-4c7c-bc6b-bd1536f19337" connectedTo="45013b48-72c7-41fc-9bfe-edd02d2e380a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="726183ce-1b2a-4a20-af24-d5162dcfa6c3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe45_ret" diameter="DN300" length="160.3" name="Pipe45" id="Pipe45" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04556653011963" lon="4.3322181701660165"/>
          <point xsi:type="esdl:Point" lat="52.04658258087284" lon="4.330555200576783"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c527ea3d-4e8b-40e6-883e-4c60cbcfbe67" id="61d57ac4-ea0f-456c-9272-ae2e9b8118a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="37c19345-8501-4de1-a138-999a8e0880f6" connectedTo="90cc9947-a05b-41c5-8b9e-e678bf033c61" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="62916fdb-adc2-458d-85f0-6bd270df322c">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe46_ret" diameter="DN300" length="537.9" name="Pipe46" id="Pipe46" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04658258087284" lon="4.330555200576783"/>
          <point xsi:type="esdl:Point" lat="52.0472499812397" lon="4.329664707183839"/>
          <point xsi:type="esdl:Point" lat="52.050039619503906" lon="4.334267377853394"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="aae35ef1-f784-489b-9e36-f0b903ca948d" id="1502fc4c-0c2e-4889-b09b-777549e2739d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="93835c29-ca52-4ed0-beaa-75161e840249" connectedTo="29b4f6b2-45e0-452d-82ed-0a1507fa40f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="77327827-0d29-488b-ba05-ea97e8b31ea9">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe47_ret" diameter="DN300" length="198.7" name="Pipe47" id="Pipe47" outerDiameter="0.5" innerDiameter="0.3127">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.050039619503906" lon="4.334267377853394"/>
          <point xsi:type="esdl:Point" lat="52.051490968223646" lon="4.3359625339508066"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="2668ac25-44a2-4305-bbbf-b10e6c7dd3c3"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.08245">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="4da97fd3-43ae-4cdd-bf85-5c00c0950dee"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0056">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="47a862f8-97be-4f97-aa8c-38c940278d33"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a20b67c3-c88a-49a1-8428-01424efffc51" id="cc22d0e4-4d26-48ee-ab92-1ca8b5571536" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="00d04a71-fa13-4bcc-919e-93a0ded8c13b" connectedTo="a26aefdd-b3f6-4f32-a185-ed07a8a47fe6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="bec06fed-e5a4-414c-b965-94e025501d4d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1962.1" id="7ffb929c-9640-436c-907a-556e790a6c7d">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="b76702e2-0572-4647-bb32-4a94640d714c" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe48_ret" diameter="DN200" length="96.0" name="Pipe48" id="Pipe48" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.051490968223646" lon="4.3359625339508066"/>
          <point xsi:type="esdl:Point" lat="52.0507784938338" lon="4.336756467819215"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="022586f7-6c01-4622-ad59-cdd682be2e9b" id="74f1cfd7-08c7-416e-a9d9-c77fb32e881d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ea278421-bea3-4098-9add-271bdc3bae8d" connectedTo="49caaf6b-e676-4e65-9ce5-027c2406c9b7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="6f1ed545-6ab3-4e01-9f2f-c30981620af7">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe50_ret" diameter="DN200" length="171.3" name="Pipe50" id="Pipe50" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.051490968223646" lon="4.3359625339508066"/>
          <point xsi:type="esdl:Point" lat="52.05278394828437" lon="4.334599971771241"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="022586f7-6c01-4622-ad59-cdd682be2e9b" id="b58cd7f3-5157-4dae-b01c-030fa56958bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="ead81158-fba7-4d8e-a6b0-d19ea45dd8fb" connectedTo="88ddaaa8-e23c-4135-92b3-a379f53f1a86" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="91b117c7-3243-43d0-88f4-e6466f9347d8">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe53_ret" diameter="DN200" length="60.6" name="Pipe53" id="Pipe53" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.050039619503906" lon="4.334267377853394"/>
          <point xsi:type="esdl:Point" lat="52.04959161616739" lon="4.334771633148194"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a20b67c3-c88a-49a1-8428-01424efffc51" id="50a51359-11eb-42a0-b56c-3abedae163e1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9b7e34bb-1cc6-4f25-891d-cca582c3783c" connectedTo="6bc62478-cdab-4c7c-bcb4-c236c475230a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="bcb3bfb8-6807-414b-98a4-a21ec6e4e59b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe54_ret" diameter="DN150" length="31.1" name="Pipe54" id="Pipe54" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04658258087284" lon="4.330555200576783"/>
          <point xsi:type="esdl:Point" lat="52.04642484175937" lon="4.330179691314698"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="aae35ef1-f784-489b-9e36-f0b903ca948d" id="44631aef-3734-44e2-b2e3-738189062c62" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e481b04a-fffe-4181-913d-17c120558ed9" connectedTo="612572ec-f6f1-43e9-a5d3-d783cef32723" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="01afe7ef-b586-4d69-98d7-75f5f58d661b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe55_ret" diameter="DN150" length="116.3" name="Pipe55" id="Pipe55" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04362013462266" lon="4.32937502861023"/>
          <point xsi:type="esdl:Point" lat="52.04417673408878" lon="4.328253865242005"/>
          <point xsi:type="esdl:Point" lat="52.04429374016349" lon="4.328430891036988"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b85c3b55-c290-4987-93b3-6c03bde6e8b4" id="c73d289c-dda5-4d21-b03f-dc19eae8aaf1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="240f2f27-935f-4a2b-99b0-6fb92c41b68e" connectedTo="eb42ca4e-4b23-4096-bdb4-75f5f6e4d767" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="6b65ac5a-5f72-4ab4-b774-3e46dfdbe94a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe56_ret" diameter="DN150" length="87.2" name="Pipe56" id="Pipe56" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04235988080374" lon="4.327293634414674"/>
          <point xsi:type="esdl:Point" lat="52.0429015415213" lon="4.326370954513551"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7fc22ee8-6e17-4a43-857c-d5d35bf111cb" id="5c60b4cd-ab2c-4ec2-a694-672ea9d24418" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="eef6b3f9-b113-4b26-801b-0845350e6dbe" connectedTo="c480062c-4cba-4b85-a992-db4f6136ffb4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="e9d86ef5-e230-459d-a961-d8e215d5e346">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe57_ret" diameter="DN150" length="91.5" name="Pipe57" id="Pipe57" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04102040978031" lon="4.323420524597169"/>
          <point xsi:type="esdl:Point" lat="52.04157528352519" lon="4.324407577514649"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3f8246ac-5bf4-4dff-b00d-f5e2e956c2b9" id="00fcc715-3f27-4296-ac99-519c7eaa8a2e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e2ef19c1-e80c-4a2a-b13d-62bd6c2ad5ed" connectedTo="39af40d7-19ef-4aed-ba38-fd5d034b4cd7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b8b054f1-dcac-4f32-b16d-70a572c82774">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe58_ret" diameter="DN150" length="45.3" name="Pipe58" id="Pipe58" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03995144351459" lon="4.321660995483399"/>
          <point xsi:type="esdl:Point" lat="52.03957592470637" lon="4.321918487548829"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4cac6a54-6b27-456e-8d17-f5910fb4744c" id="f219ef2b-e6b8-404d-ad1f-610c680fae0a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="285dee2d-f174-4a51-8272-7f97c4606e6d" connectedTo="799464bf-4173-450b-b2e4-0c143c0e4394" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="08743cc2-3cc9-4b7d-a25d-f8dfadacf57b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe59_ret" diameter="DN150" length="48.9" name="Pipe59" id="Pipe59" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.042894338333426" lon="4.318453073501588"/>
          <point xsi:type="esdl:Point" lat="52.04313255137357" lon="4.319053888320924"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="679d18da-f447-41dd-916f-e5d3b29d1184" id="14d9687b-c97e-4d19-ad3e-b0776928f0bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="878fc721-fcd7-45e7-a85e-de03b6c34cf0" connectedTo="dcb6b734-6182-4704-bf62-97c6bc0a6cea" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="72d9e8da-a018-4cc4-8b7e-2f1aaac5468b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe61_ret" diameter="DN250" length="59.4" name="Pipe61" id="Pipe61" outerDiameter="0.4" innerDiameter="0.263">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0507784938338" lon="4.336756467819215"/>
          <point xsi:type="esdl:Point" lat="52.0511881131729" lon="4.337314367294312"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.005">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="faac539b-4b7c-43f8-abcd-f08fa2652b7b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0587">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="d23b4eeb-a419-4c16-bc7e-280a76116f04"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0048">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="a2b91e8d-471d-4276-a8f6-4efb01054b4e"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d906ef7e-682b-484e-82ac-584a0115c91e" id="566725ca-a91e-4c6c-b84e-b3809e241f82" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="be3c8912-b660-4c3b-8a34-c929386c2107" connectedTo="34ea2553-c243-4d3d-b607-9aecd7fdf690" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="0804dd0e-b96d-42d0-86b9-ad2b389cd657">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1630.7" id="8dad1cd1-95ff-4c6e-b644-a1d38cfe4b1f">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="eca952d5-fbd9-4efd-a72c-2eb773445f25" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage1" id="451ed395-074a-41f3-853d-86fa5a924287">
        <geometry xsi:type="esdl:Point" lat="52.03733897302974" lon="4.317283630371095" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="eb32f4c3-c492-439d-89e8-142ea6882c67" id="6f024397-cf6a-4232-bdb3-e9e4285fed59" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5c089df4-9cb1-4249-8e2b-b46994220f62" connectedTo="95416ff8-70ce-48d9-ae41-cdf234f4408f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe63_ret" diameter="DN400" length="76.2" name="Pipe63" id="Pipe63" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03766978308298" lon="4.3182599544525155"/>
          <point xsi:type="esdl:Point" lat="52.03733897302974" lon="4.317283630371095"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="39f0b02a-1dbf-42e7-99a6-143e5868754a" id="d929c625-08e9-4d06-bf5e-03cd9be0d981" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="eb32f4c3-c492-439d-89e8-142ea6882c67" connectedTo="6f024397-cf6a-4232-bdb3-e9e4285fed59" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="9db631b4-0a8b-4fa6-a670-84e33b9bd158">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage2" id="577ead9d-020c-4eea-a10a-cc28a1f7b371">
        <geometry xsi:type="esdl:Point" lat="52.03936100685403" lon="4.305393397808076"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e9411099-f51f-42bc-a28c-34003785ef6e" id="e83e4b67-d962-4158-8cfd-b0d4e6a0dc59" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d3308201-f313-4047-8c84-e7d1f3ba6f41" connectedTo="f55dc1fe-acc4-42ec-9167-b959b5e3613b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage3" id="c63443fd-d008-434a-b3fc-a38f136dfd6d">
        <geometry xsi:type="esdl:Point" lat="52.03761704929123" lon="4.296324849128724"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b0f3d110-2bce-4152-913f-8acdd8250960" id="385cab49-0287-4090-bc88-c2675edcec8e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9b499471-ef6a-496d-9d5a-33b3e80441fa" connectedTo="cc812c98-eca1-4ac7-aa76-2387306e2ba3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage4" id="799de5fe-df07-42d7-815d-48dbb8797395">
        <geometry xsi:type="esdl:Point" lat="52.03418833265955" lon="4.307541847229005"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="6a4f3680-ed3c-4394-bb22-2a641a718b4b" id="9ee4d724-c7b5-46b4-8668-ae642b5eab4e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b00171ec-afff-420e-96f2-c3ddc27514b3" connectedTo="942ae075-61b0-4789-9cbe-b639210a4fc6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage5" id="2ada53b1-2ef7-42ab-9186-28fbdb00cf61">
        <geometry xsi:type="esdl:Point" lat="52.03169005495035" lon="4.312176704406739" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="6a9e967d-e812-4544-ab3a-b83c2a2c5836" id="70f55bb2-f1e7-4f4e-a2c1-52f858ba600d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="1a33dc29-d2c1-46da-a214-ac34db563a56" connectedTo="e23c6609-5456-44e7-bbef-213af17d50f2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage6" id="121ae7bf-44fa-4386-9b96-1530923716c1">
        <geometry xsi:type="esdl:Point" lat="52.03047551813576" lon="4.310331344604493" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="38b8ae60-446f-493a-bca2-3add528e4f63" id="c9150c2a-504b-413b-a3aa-d67cc04bd4e1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f6a0a68a-1b36-4692-8513-01fac18923d1" connectedTo="bb90655a-7cef-4708-9d83-189901ee2090" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage7" id="a58221b4-1ee9-45df-8daf-d44fa7743c62">
        <geometry xsi:type="esdl:Point" lat="52.03013227349181" lon="4.299452304840089" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="30fa16c4-09e1-4cda-a7d9-675fe8ba27c3" id="95a55199-8a02-406d-be28-3e1488c42d2f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2b1d2659-1961-47b0-a271-e4f4167722db" connectedTo="8c592f69-5d36-4512-b5da-67a6c3174d00" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage8" id="69163299-91a8-44a8-8630-62d12d2d04b2">
        <geometry xsi:type="esdl:Point" lat="52.02803314315194" lon="4.309612512588502" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="0b813381-3866-47ba-a1f9-af659164d3b2" id="91bb0e50-7375-4a9a-9134-92a0a405c56c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="06c7f257-3a41-4f5e-9574-64727cca9a3f" connectedTo="bc6002db-b6fb-4e63-9acb-13aa1367da4c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage9" id="5f24c38c-f939-4a6a-9b5d-986b9f611161">
        <geometry xsi:type="esdl:Point" lat="52.02953818999695" lon="4.314976930618287" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="fb359a59-4718-4c10-b657-5f6c6e588d54" id="d4959b19-f76f-41b4-879c-5dc3115e0663" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b8c47d9e-9c4c-4404-80a4-54d139c8ea00" connectedTo="3d47c270-f866-46af-b581-26fb8d9811a4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer2" id="2dd4cc9f-9160-40b9-a399-7f41a85199dd">
        <geometry xsi:type="esdl:Point" lat="52.0280595479191" lon="4.3102455139160165" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b9cc1676-2ae4-4c8f-8f2a-b5c30ddc2da1" connectedTo="16982a7c-6dd9-4d28-9625-c237d10ba7ba" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="0f8ad5ee-4492-4e07-bb07-e51f8289b390" id="983b16dc-f1ed-47a3-8eef-7116cec4cb5d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer3" id="cfb887f5-b4d4-4293-a8aa-a2f63ca34b2a">
        <geometry xsi:type="esdl:Point" lat="52.03010586994853" lon="4.314386844635011" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="875796b4-2494-412c-973d-4b701d1f8ce4" connectedTo="2a2133bd-d390-4437-93be-a5395093432f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="cec7a770-b845-44f2-a048-1dd075cf9bc2" id="af926335-28b7-49d3-bb8a-102f5349ec67" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer4" id="1c848227-ad3e-4363-be7d-32f83aef9121">
        <geometry xsi:type="esdl:Point" lat="52.030759353062656" lon="4.3096661567688" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7e829ee4-d06c-4989-949a-80b2c4b526ff" connectedTo="52fef504-660b-4412-a6c2-7e55e0ff40d4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="87141bc9-eff2-49db-be90-11d0e53ec3a6" id="c57e0353-939f-42ff-b6c0-6be7af99e5fc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer5" id="ba10e815-ff26-4ad0-adc9-f56db58ed3c8">
        <geometry xsi:type="esdl:Point" lat="52.03015207613905" lon="4.299956560134889" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a1059252-2185-489a-b446-609114b8bf11" connectedTo="c304fb04-593d-4bef-8e35-4f72f9d3251a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d6caa807-0d17-4fd0-a983-5c1036159a78" id="49295fb9-e5b6-4217-82a3-1ccd76940098" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer6" id="302257e5-0f5a-40ae-a0c4-607954e73ba6">
        <geometry xsi:type="esdl:Point" lat="52.034128930054116" lon="4.309451580047608"/>
        <port xsi:type="esdl:OutPort" name="Out" id="baa97c90-387b-4e15-8ef4-a8323e864b89" connectedTo="1c232335-88f9-48e6-9391-7a9c62d24b97" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="42ed67c8-1ad9-4f54-a04e-50ae8c419bb2" id="9124da2f-9c41-4217-9ba2-db523680605a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer7" id="cc6869cf-df28-48f5-b6cf-567328d16d6b">
        <geometry xsi:type="esdl:Point" lat="52.03538296830313" lon="4.304199814796449"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5909bc15-6be7-4fbe-962d-bde77cd80556" connectedTo="07af7b95-0734-4c10-85ed-2137da51b226" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="1adbcc35-d2b4-4115-b635-ed074a43dc7f" id="ed240007-6411-4983-b2ae-10eeb9d4c735" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer8" id="19e16a55-bbfc-4704-8dd2-f7a6e555959f">
        <geometry xsi:type="esdl:Point" lat="52.03785794057299" lon="4.2971616983413705"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b780bbf5-40ad-4116-b54c-627af1bf664d" connectedTo="716a7cf3-1326-4eb0-8f45-3d40bdecfdbb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d080f436-4bd8-4cc7-bc75-c38ce217ac88" id="1a6bbd6f-acb3-4ec3-93f0-690c1ed4cc71" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer9" id="f4634bd3-56a7-4b70-9a97-5dd9d2534cc3">
        <geometry xsi:type="esdl:Point" lat="52.03965468459409" lon="4.304840862751008"/>
        <port xsi:type="esdl:OutPort" name="Out" id="04b35e4e-fb03-42d5-b923-8a5e0269caa4" connectedTo="11750784-fe1d-417c-946b-26e499eae225" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="2af09cda-994d-426f-8b34-9377f3427944" id="88741a34-59fa-455e-828d-a0144a9178bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage10" id="7c6a7002-bb33-4ba4-8058-9b0ca80e1139">
        <geometry xsi:type="esdl:Point" lat="52.03549847005698" lon="4.303100109100343"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e9dc6a73-d926-4e52-8308-e35953601f6a" id="784481ed-cfb1-4565-b4c8-182b95daf6e1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="87c171ef-0902-49cd-a82c-c44d687aa5d4" connectedTo="876f3a14-19ac-44e6-a746-d509979e459b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe65_ret" diameter="DN400" length="38.3" name="Pipe65" id="Pipe65" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03907950206215" lon="4.304720163345338"/>
          <point xsi:type="esdl:Point" lat="52.03925376450173" lon="4.3052029609680185"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a572f6a9-36cb-498c-b44d-b092d7288aa3" id="4ebbb86f-e759-459d-bf89-6d816892becd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e9411099-f51f-42bc-a28c-34003785ef6e" connectedTo="e83e4b67-d962-4158-8cfd-b0d4e6a0dc59" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b75ca449-69af-46e1-a012-d39bfe11561d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe66_ret" diameter="DN400" length="57.9" name="Pipe66" id="Pipe66" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03957714065919" lon="4.304966926574708"/>
          <point xsi:type="esdl:Point" lat="52.03907950206215" lon="4.304720163345338"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="04b35e4e-fb03-42d5-b923-8a5e0269caa4" id="11750784-fe1d-417c-946b-26e499eae225" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b21a2b68-30a7-478b-a633-5a398f2a1834" connectedTo="125e86b9-1b17-45cb-b4ae-eddbe3a4809d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="01fbf590-84aa-49d7-b27c-d53f8326bf29">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe68_ret" diameter="DN400" length="88.2" name="Pipe68" id="Pipe68" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0372369642963" lon="4.297864437103272"/>
          <point xsi:type="esdl:Point" lat="52.03757745060015" lon="4.296700358390809"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="91e38e17-04b1-4174-a200-899ca54224be" id="daf67778-22d9-4e57-9467-3477c15d08f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b0f3d110-2bce-4152-913f-8acdd8250960" connectedTo="385cab49-0287-4090-bc88-c2675edcec8e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="44bdd65b-da82-44cb-b9c6-9f84421a8a9a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe67_ret" diameter="DN400" length="55.3" name="Pipe67" id="Pipe67" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03765004817369" lon="4.29741382598877"/>
          <point xsi:type="esdl:Point" lat="52.0372369642963" lon="4.297864437103272"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b780bbf5-40ad-4116-b54c-627af1bf664d" id="716a7cf3-1326-4eb0-8f45-3d40bdecfdbb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="08fee2ea-5e6e-4032-9062-a5208ced4a3c" connectedTo="a7a34f2e-4ac0-40b4-9d15-ef4d784b5e1f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2b01912e-7c94-46ed-a65a-e74531fec962">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe20_a_ret" diameter="DN200" length="115.61" name="Pipe20_a" id="Pipe20_a" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03623389944942" lon="4.305782318115235"/>
          <point xsi:type="esdl:Point" lat="52.03585157356818" lon="4.304210543632508"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7fb2c2ec-9dff-4709-b221-9a6bcc1b205b" id="c4747870-0077-46b4-803f-8d8111839843" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4dfb863d-e5c8-4acb-8085-ce515352a87b" connectedTo="29e66021-7a5c-4d0a-8aea-0bc3f98033fd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="ea2dfb8b-c135-49ba-8153-ed52def11b0a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe20_b_ret" diameter="DN200" length="31.29" name="Pipe20_b" id="Pipe20_b" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03585157356818" lon="4.304210543632508"/>
          <point xsi:type="esdl:Point" lat="52.035745557102025" lon="4.303786754608155"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="afb35818-0d1a-49e2-ac7b-18794b38fecd" id="ca1a3430-182c-40a6-a952-5765d848daf4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="00f6e2a9-bd19-43ec-965b-fae9fcc33bdb" connectedTo="70dc27c6-6a93-4801-b83e-140394796527" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="ea2dfb8b-c135-49ba-8153-ed52def11b0a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_824f" id="824fc703-5685-49d1-8f23-1e1568b11f2c">
        <geometry xsi:type="esdl:Point" lat="52.03585157356818" lon="4.304210543632508"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4dfb863d-e5c8-4acb-8085-ce515352a87b 8763a794-0f1b-46ce-b379-8071ab326346" id="29e66021-7a5c-4d0a-8aea-0bc3f98033fd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="afb35818-0d1a-49e2-ac7b-18794b38fecd" connectedTo="ca1a3430-182c-40a6-a952-5765d848daf4 afae16a7-ca05-4abf-8824-329fd192cad0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe70_ret" diameter="DN400" length="73.0" name="Pipe70" id="Pipe70" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03585157356818" lon="4.304210543632508"/>
          <point xsi:type="esdl:Point" lat="52.03550507014818" lon="4.3033039569854745"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="afb35818-0d1a-49e2-ac7b-18794b38fecd" id="afae16a7-ca05-4abf-8824-329fd192cad0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e9dc6a73-d926-4e52-8308-e35953601f6a" connectedTo="784481ed-cfb1-4565-b4c8-182b95daf6e1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="00ceaa5e-38cc-4103-a97e-961412665d06">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe69_ret" diameter="DN400" length="38.9" name="Pipe69" id="Pipe69" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03550672017082" lon="4.304301738739015"/>
          <point xsi:type="esdl:Point" lat="52.03585157356818" lon="4.304210543632508"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5909bc15-6be7-4fbe-962d-bde77cd80556" id="07af7b95-0734-4c10-85ed-2137da51b226" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="8763a794-0f1b-46ce-b379-8071ab326346" connectedTo="29e66021-7a5c-4d0a-8aea-0bc3f98033fd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="01bcbae8-dc80-4130-9c1a-f496dd711b36">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe71_ret" diameter="DN400" length="43.1" name="Pipe71" id="Pipe71" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0342411349092" lon="4.309542775154115"/>
          <point xsi:type="esdl:Point" lat="52.03438608950049" lon="4.308958053588868"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="baa97c90-387b-4e15-8ef4-a8323e864b89" id="1c232335-88f9-48e6-9391-7a9c62d24b97" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="0e1a3e0d-166b-4639-8723-3a1b9f416fec" connectedTo="d90ca402-cd8d-429e-a04c-d5e817f99181" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="a9efd229-3f7b-477c-aa93-e35383f3a9e4">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe72_ret" diameter="DN400" length="74.9" name="Pipe72" id="Pipe72" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03438608950049" lon="4.308958053588868"/>
          <point xsi:type="esdl:Point" lat="52.034211433651436" lon="4.307901263237"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="59de07d5-eb72-42f5-af24-0974da4122c9" id="0ba07947-d938-4f48-9ebc-a38f4a70a15d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="6a4f3680-ed3c-4394-bb22-2a641a718b4b" connectedTo="9ee4d724-c7b5-46b4-8668-ae642b5eab4e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="d72b45d3-54c2-4921-92c6-5a9d221c1832">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe14_a_ret" diameter="DN200" length="721.48" name="Pipe14_a" id="Pipe14_a" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03260420043414" lon="4.310138225555421"/>
          <point xsi:type="esdl:Point" lat="52.03077239626749" lon="4.303561449050904"/>
          <point xsi:type="esdl:Point" lat="52.03040290891157" lon="4.300289154052735"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9b977589-8e44-475a-a0ed-f4420d88e233" id="25717110-9674-418f-a506-055fe48c695f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d1efeaaa-918a-461b-bbfe-9175bb377b7b" connectedTo="017e5ea5-8c44-49c5-84b6-c2c692894357" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="02a46818-1ec5-4ab4-9570-e664c50e6cc5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe14_b_ret" diameter="DN200" length="62.56" name="Pipe14_b" id="Pipe14_b" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03040290891157" lon="4.300289154052735"/>
          <point xsi:type="esdl:Point" lat="52.03030743866227" lon="4.299387931823731"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3f54f5e4-895e-4f71-b94a-bc861965ca6b" id="1ece7e67-6fa4-4043-b78d-2f8a66a1ffe7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d1611125-9f5a-4f97-9786-1e9fd114730a" connectedTo="fd0f353b-455a-4f90-b8c2-b5643121a8e3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="02a46818-1ec5-4ab4-9570-e664c50e6cc5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_c553" id="c5538e17-2a9b-4914-996f-7f64a4f79958">
        <geometry xsi:type="esdl:Point" lat="52.03040290891157" lon="4.300289154052735"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d1efeaaa-918a-461b-bbfe-9175bb377b7b 4a2aea50-5ff6-42d8-8e2f-1fd6c28757d7" id="017e5ea5-8c44-49c5-84b6-c2c692894357" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="3f54f5e4-895e-4f71-b94a-bc861965ca6b" connectedTo="1ece7e67-6fa4-4043-b78d-2f8a66a1ffe7 c701aa19-086a-4011-870f-0434af7cef89" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe74_ret" diameter="DN400" length="64.7" name="Pipe74" id="Pipe74" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03040290891157" lon="4.300289154052735"/>
          <point xsi:type="esdl:Point" lat="52.03013227349181" lon="4.299452304840089"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3f54f5e4-895e-4f71-b94a-bc861965ca6b" id="c701aa19-086a-4011-870f-0434af7cef89" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="30fa16c4-09e1-4cda-a7d9-675fe8ba27c3" connectedTo="95a55199-8a02-406d-be28-3e1488c42d2f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c0abd746-e68f-4f64-be0d-e7c2c973a503">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe73_ret" diameter="DN400" length="36.0" name="Pipe73" id="Pipe73" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03015207613905" lon="4.299956560134889"/>
          <point xsi:type="esdl:Point" lat="52.03040290891157" lon="4.300289154052735"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a1059252-2185-489a-b446-609114b8bf11" id="c304fb04-593d-4bef-8e35-4f72f9d3251a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4a2aea50-5ff6-42d8-8e2f-1fd6c28757d7" connectedTo="017e5ea5-8c44-49c5-84b6-c2c692894357" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="044fd2be-aa16-4f7b-92cc-d96cc6ea3aa2">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer10" id="b8ec1d77-e9e2-4cce-9599-9a07d2d36373">
        <geometry xsi:type="esdl:Point" lat="52.03154484032849" lon="4.31166708469391" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="048442b5-b987-405f-91de-042ad374e432" connectedTo="57df0793-8665-4d90-af15-f4b9447204f5" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7cc1c74d-db40-45ab-ad50-5ed0b697f795" id="1de51e8b-def4-4443-8ce4-b9eb14dc8e8a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe75_ret_ret" diameter="DN400" length="97.6" name="Pipe75" id="Pipe75_ret" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.031799027265784" lon="4.310760498046876"/>
          <point xsi:type="esdl:Point" lat="52.03169005495035" lon="4.312176704406739"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4af9fa7e-41ee-4e79-97c4-e0dd2b1419d5" id="267e82e0-ee82-4fe5-b249-50a115de72f6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="6a9e967d-e812-4544-ab3a-b83c2a2c5836" connectedTo="70f55bb2-f1e7-4f4e-a2c1-52f858ba600d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="9c1f014c-d507-4d3e-baf5-d78512df87d2">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe76_ret" diameter="DN400" length="68.2" name="Pipe76" id="Pipe76" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03154484032849" lon="4.31166708469391"/>
          <point xsi:type="esdl:Point" lat="52.031799027265784" lon="4.310760498046876"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="048442b5-b987-405f-91de-042ad374e432" id="57df0793-8665-4d90-af15-f4b9447204f5" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="39b5089e-627f-48b2-a419-6c0a3f24d00a" connectedTo="d0436b36-2700-440b-a0f9-869730642d19" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="feb16ff9-1c02-435c-b11a-dbf474c83c8f">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe77_ret" diameter="DN400" length="105.8" name="Pipe77" id="Pipe77" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0312182376185" lon="4.311296939849854"/>
          <point xsi:type="esdl:Point" lat="52.03047551813576" lon="4.310331344604493"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba5f246f-3f6a-4b80-ba1e-92b9c818e6d9" id="5479c538-0675-4039-8872-73449b96d60c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="38b8ae60-446f-493a-bca2-3add528e4f63" connectedTo="c9150c2a-504b-413b-a3aa-d67cc04bd4e1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="d232a6a2-d6bb-471e-a05e-5d7d813789ec">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe78_ret" diameter="DN400" length="122.7" name="Pipe78" id="Pipe78" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.030759353062656" lon="4.3096661567688"/>
          <point xsi:type="esdl:Point" lat="52.0312182376185" lon="4.311296939849854"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7e829ee4-d06c-4989-949a-80b2c4b526ff" id="52fef504-660b-4412-a6c2-7e55e0ff40d4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="008df1e4-53f0-4122-ad6b-901857d7a052" connectedTo="96fb6c73-14d6-4e0f-b18f-87b1f146a419" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="e33ca196-f70b-44a9-973d-3fbbdcd7d8f0">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage11" id="3dd97e5f-35a6-42a8-92a8-e78bc0f85d82">
        <geometry xsi:type="esdl:Point" lat="52.02621777802815" lon="4.314944744110108" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4ea44fca-2441-4be7-99fb-95dc2f02bd95" id="6ab1a455-f072-411e-8395-0d3796fcfcd7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="fb2de451-5690-47c3-9447-4bcb65f5b683" connectedTo="36be5ca1-5f82-40c1-9845-43f86ea6ec37" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer11" id="bae40abf-8612-4b71-a847-829073d89ac4">
        <geometry xsi:type="esdl:Point" lat="52.02650824140003" lon="4.3146550655365" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5851a97b-a795-469f-8ef9-100e8fadd8d9" connectedTo="f1449c57-1d0e-407b-804a-1552351713a9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="da69fa88-7382-4cd7-9252-e7569aa71709" id="f4cb60f0-0db8-44ca-9e7e-f3952e867f4b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe60_a_ret" diameter="DN250" length="120.04" name="Pipe60_a" id="Pipe60_a" outerDiameter="0.4" innerDiameter="0.263">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02896100622209" lon="4.313313961029054"/>
          <point xsi:type="esdl:Point" lat="52.02924683664851" lon="4.3142902851104745"/>
          <point xsi:type="esdl:Point" lat="52.02945144288258" lon="4.314107894897462"/>
          <point xsi:type="esdl:Point" lat="52.02958344640772" lon="4.314236640930177"/>
          <point xsi:type="esdl:Point" lat="52.02957779580883" lon="4.314279556274415"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.005">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="faac539b-4b7c-43f8-abcd-f08fa2652b7b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0587">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="d23b4eeb-a419-4c16-bc7e-280a76116f04"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0048">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="a2b91e8d-471d-4276-a8f6-4efb01054b4e"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="bda0d0ea-4371-4e95-b38b-c7b0002e2d73" id="7178dbd9-04d5-46cc-83f7-42a255a66eed" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a468aff1-0ff1-49a5-83eb-52542719efce" connectedTo="5a4152c3-c720-478b-8f1b-41d9a9f37397" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="d0cf8bf4-c920-40a5-ab34-ab8672a744d8">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1630.7" id="8dad1cd1-95ff-4c6e-b644-a1d38cfe4b1f">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="eca952d5-fbd9-4efd-a72c-2eb773445f25" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe60_b_ret" diameter="DN250" length="35.24" name="Pipe60_b" id="Pipe60_b" outerDiameter="0.4" innerDiameter="0.263">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02957779580883" lon="4.314279556274415"/>
          <point xsi:type="esdl:Point" lat="52.029753030214444" lon="4.314708709716798"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.005">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="faac539b-4b7c-43f8-abcd-f08fa2652b7b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0587">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="d23b4eeb-a419-4c16-bc7e-280a76116f04"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0048">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="a2b91e8d-471d-4276-a8f6-4efb01054b4e"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5918b541-ae92-4799-968c-cfea926b766d" id="82343f0c-b845-446a-80fe-3bb9ce2c0989" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2218f77f-a63c-425a-8900-179dfdb53706" connectedTo="d937e02c-ce60-4352-90a2-ad6420f3147e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="d0cf8bf4-c920-40a5-ab34-ab8672a744d8">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1630.7" id="8dad1cd1-95ff-4c6e-b644-a1d38cfe4b1f">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="eca952d5-fbd9-4efd-a72c-2eb773445f25" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_4bae" id="4bae0ca9-ead6-4b3e-b29e-ddcd81293db6">
        <geometry xsi:type="esdl:Point" lat="52.02957779580883" lon="4.314279556274415"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a468aff1-0ff1-49a5-83eb-52542719efce c788bfe6-2428-44e4-b09a-960613f8da9c" id="5a4152c3-c720-478b-8f1b-41d9a9f37397" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5918b541-ae92-4799-968c-cfea926b766d" connectedTo="82343f0c-b845-446a-80fe-3bb9ce2c0989 3d185b3b-05a7-4483-85da-f28aaed9f8d0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe34_a_ret" diameter="DN150" length="165.61" name="Pipe34_a" id="Pipe34_a" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.028010558913955" lon="4.312477111816407"/>
          <point xsi:type="esdl:Point" lat="52.02740281078387" lon="4.310722947120667"/>
          <point xsi:type="esdl:Point" lat="52.02749184198936" lon="4.310342073440553"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="460ad9bd-ae51-47e2-b9a7-e08bc4c86fb9" id="ba5d2033-f0b4-414e-b39c-0e72e122b747" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="8272bbb1-5648-4bdf-9355-4aa5918b70e9" connectedTo="82c301a9-3a78-46c6-9ccc-eddd1d8f7eea" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b33c5685-8c2c-4c89-9041-28615d7b6cd3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe34_b_ret" diameter="DN150" length="48.65" name="Pipe34_b" id="Pipe34_b" outerDiameter="0.25" innerDiameter="0.1603">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02749184198936" lon="4.310342073440553"/>
          <point xsi:type="esdl:Point" lat="52.027627734152475" lon="4.3096661567688"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.004">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="fa85538e-ebfa-4bce-8386-04980e793e18"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.03725">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="3bafa031-f40f-42fc-b409-e35fffe5f457"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0036">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="893337e3-58f1-4fb4-8c25-68d71b11fb71"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5e82e025-ebf3-4252-816b-ebe9b272bddb" id="4a55e313-5c67-4414-b33e-9341364a3c1c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e1a13ea4-f7eb-4fb4-afe8-283e388d19b1" connectedTo="fc31e841-4b59-467b-a070-ba95631e977b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b33c5685-8c2c-4c89-9041-28615d7b6cd3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1126.4" id="234f7d03-69a7-429a-b44f-f57c522345f1">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="a5eefd42-c8e8-4dd6-b0dc-53ed92ead8a5" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1817" id="18175367-cea8-4443-9219-b767d7b8cfee">
        <geometry xsi:type="esdl:Point" lat="52.02749184198936" lon="4.310342073440553"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="8272bbb1-5648-4bdf-9355-4aa5918b70e9 79485ed3-8bb1-4fc2-a45a-c22c5c69db97" id="82c301a9-3a78-46c6-9ccc-eddd1d8f7eea" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5e82e025-ebf3-4252-816b-ebe9b272bddb" connectedTo="4a55e313-5c67-4414-b33e-9341364a3c1c 2c356e71-d7ce-4e62-a453-3ef7675a62df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe82_ret" diameter="DN400" length="78.2" name="Pipe82" id="Pipe82" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02749184198936" lon="4.310342073440553"/>
          <point xsi:type="esdl:Point" lat="52.02803314315194" lon="4.309612512588502"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5e82e025-ebf3-4252-816b-ebe9b272bddb" id="2c356e71-d7ce-4e62-a453-3ef7675a62df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="0b813381-3866-47ba-a1f9-af659164d3b2" connectedTo="91bb0e50-7375-4a9a-9134-92a0a405c56c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="e3dceb26-0c1b-4d85-9e8c-52e6c26db478">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe81_ret" diameter="DN400" length="63.5" name="Pipe81" id="Pipe81" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0280595479191" lon="4.3102455139160165"/>
          <point xsi:type="esdl:Point" lat="52.02749184198936" lon="4.310342073440553"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b9cc1676-2ae4-4c8f-8f2a-b5c30ddc2da1" id="16982a7c-6dd9-4d28-9625-c237d10ba7ba" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="79485ed3-8bb1-4fc2-a45a-c22c5c69db97" connectedTo="82c301a9-3a78-46c6-9ccc-eddd1d8f7eea" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="f08d1569-902e-4b6f-a148-3ea56a7785bb">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe79_ret" diameter="DN400" length="47.9" name="Pipe79" id="Pipe79" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02957779580883" lon="4.314279556274415"/>
          <point xsi:type="esdl:Point" lat="52.02953818999695" lon="4.314976930618287"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5918b541-ae92-4799-968c-cfea926b766d" id="3d185b3b-05a7-4483-85da-f28aaed9f8d0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="fb359a59-4718-4c10-b657-5f6c6e588d54" connectedTo="d4959b19-f76f-41b4-879c-5dc3115e0663" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3df780c4-19b1-4eb4-b5b5-9de9eda28d41">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe80_ret" diameter="DN400" length="59.2" name="Pipe80" id="Pipe80" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03010586994853" lon="4.314386844635011"/>
          <point xsi:type="esdl:Point" lat="52.02957779580883" lon="4.314279556274415"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="875796b4-2494-412c-973d-4b701d1f8ce4" id="2a2133bd-d390-4437-93be-a5395093432f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c788bfe6-2428-44e4-b09a-960613f8da9c" connectedTo="5a4152c3-c720-478b-8f1b-41d9a9f37397" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="20ebe898-ee82-45b1-928e-885dd9c320a8">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe83_ret" diameter="DN400" length="47.5" name="Pipe83" id="Pipe83" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02605679822976" lon="4.314301013946534"/>
          <point xsi:type="esdl:Point" lat="52.02621777802815" lon="4.314944744110108"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e70641e4-96be-4c1b-9c04-b9257adda5d7" id="6ad1c8fb-c41a-46a1-a2ac-44404baeef81" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4ea44fca-2441-4be7-99fb-95dc2f02bd95" connectedTo="6ab1a455-f072-411e-8395-0d3796fcfcd7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b35b23b2-3746-4930-a9f1-911bb7a602d6">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe84_ret" diameter="DN400" length="55.7" name="Pipe84" id="Pipe84" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.02650824140003" lon="4.3146550655365"/>
          <point xsi:type="esdl:Point" lat="52.02605679822976" lon="4.314301013946534"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5851a97b-a795-469f-8ef9-100e8fadd8d9" id="f1449c57-1d0e-407b-804a-1552351713a9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="6a434cd0-16d6-4138-ba03-c8d397f4e0dd" connectedTo="8312aa67-8f5f-4eb0-a8ad-07dc0c7f3649" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="cab30504-99a1-47c7-82e0-c852cde7fb80">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage12" id="7f518e5f-efcd-41c9-bfdf-936bc682332d">
        <geometry xsi:type="esdl:Point" lat="52.041576741265814" lon="4.312326908111573"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="34a530ba-1ff9-4515-b958-f0eed2a2ebc9" id="fa7a0ff7-fbc2-47fa-b24a-8a5b0eff235d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b7acc708-f7d2-4dd5-95ae-d5265c59fb13" connectedTo="4d317d5f-fdd9-47b8-b5e7-7789e8aa16eb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage13" id="a163a84e-52c6-4ac3-b877-aba6bd9f6108">
        <geometry xsi:type="esdl:Point" lat="52.03900298138126" lon="4.317176342010499" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9e6e3962-48a6-44b6-b4ed-4aa291015e64" id="3ae7bb41-1706-4b92-9b1d-69bf45707787" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d1a8eb75-ec20-49bb-b581-e2f6e48be4a6" connectedTo="1ab46cfb-031a-49f5-a1ea-31dbe10ba17e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage14" id="280562a7-2d09-4e93-820d-43aafd2461e5">
        <geometry xsi:type="esdl:Point" lat="52.03460744879936" lon="4.318914413452149" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="fd9d1bb3-97a4-449b-ac31-188a28a653a3" id="6d24513f-b153-4241-a5e6-813ce7a33ccf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="acbbb125-fd6a-454e-a645-70ac2e831883" connectedTo="e5a5d161-9a32-4d74-8881-1f4f6848c815" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage15" id="ac049de7-891c-4c2a-ab4b-9b1d1c8eb162">
        <geometry xsi:type="esdl:Point" lat="52.03273294610615" lon="4.320030212402345" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f119a0bf-e439-43dd-8a5c-29d84e45be5d" id="4213513b-398a-4350-bb50-38b447403f0a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="079a3f77-803c-487a-b141-a845644b0d5b" connectedTo="ccf83cae-ce9a-4ab7-89f6-180ae3884858" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage16" id="853710ce-eb55-486e-bc37-78547cdd840a">
        <geometry xsi:type="esdl:Point" lat="52.03137322244006" lon="4.32464361190796" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="41716cca-e7a3-4f30-96c1-7a1db967a797" id="e9a48e2c-bace-41f7-ad38-e84678dfe345" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="13874a79-3fbe-418c-a19d-2c0f98228bee" connectedTo="5b07d92e-c441-49eb-a123-e81d02a2a358" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer12" id="94cae62b-ba21-49e3-bce5-e39d59549f5f">
        <geometry xsi:type="esdl:Point" lat="52.03217850067201" lon="4.323420524597169" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9fc6944d-648c-4294-9784-156f8156dd5f" connectedTo="625c9e58-0600-4305-9486-f4b31b6a001e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b47dfdc3-f363-4220-82d6-a9e73bb84f13" id="74f2d395-2b58-4f92-8253-5996a7e1eada" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer13" id="6ffee8fb-e57c-4db4-8a3c-5f5a823df922">
        <geometry xsi:type="esdl:Point" lat="52.03238971879093" lon="4.32039499282837" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="0e5966f7-a348-49b8-9c57-5c79356bd22f" connectedTo="9972a975-06e2-4452-89c0-2b428b97380c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="79fffb1a-d928-48ab-99f2-25e0f5fe9af0" id="d9838289-335e-4070-a7fa-12423f2c62de" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer14" id="11f14150-e21c-4ef9-8bb9-8ac66ca15e62">
        <geometry xsi:type="esdl:Point" lat="52.03433023856414" lon="4.319128990173341" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7ad6ccee-4f96-44bb-aba0-7a6d719bd538" connectedTo="d7f08ab0-2d0e-4ee4-bf34-7e30a6625122" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c44e43f9-9508-4a85-8d9c-c23f9ffd7083" id="b25eaf74-e967-4ca2-b8dc-0f5e2df6f719" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer15" id="44c23c96-2733-4ab3-8dbf-8a3e757f2d05">
        <geometry xsi:type="esdl:Point" lat="52.039451750181925" lon="4.316875934600831" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7f48a3f7-45d9-4fef-bef3-ed371083a026" connectedTo="01bccc70-3665-4c0b-9494-a89ee374d1b1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a2e00dd2-85d1-4ad5-a8f7-c2258fb36c47" id="88e26467-fe6f-48f2-ae81-a0fe0a44f184" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer16" id="1836c3d1-4b8b-48cb-96d4-20e0e7b3d7a0">
        <geometry xsi:type="esdl:Point" lat="52.04131277271331" lon="4.312541484832765" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a62b4690-0b57-4cbe-bce8-2642b14b2ab7" connectedTo="d332e890-d425-4cbb-b7d8-d6b51922d397" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e1bf78da-811b-4b3f-a3e1-6681a3dd7a77" id="ca759595-c15b-44fe-9e33-2e2ccfd2d459" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe85_ret" diameter="DN400" length="32.6" name="Pipe85" id="Pipe85" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04164900734322" lon="4.312788248062135"/>
          <point xsi:type="esdl:Point" lat="52.041576741265814" lon="4.312326908111573"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d9e086ab-5ad3-4235-98ad-0f109001a4d4" id="19c66535-7622-4e61-8056-ad6c9e150172" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="34a530ba-1ff9-4515-b958-f0eed2a2ebc9" connectedTo="fa7a0ff7-fbc2-47fa-b24a-8a5b0eff235d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="a45e7aef-4039-4b1a-a35d-d592ac79154e">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe86_ret" diameter="DN400" length="41.0" name="Pipe86" id="Pipe86" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04131277271331" lon="4.312541484832765"/>
          <point xsi:type="esdl:Point" lat="52.04164900734322" lon="4.312788248062135"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a62b4690-0b57-4cbe-bce8-2642b14b2ab7" id="d332e890-d425-4cbb-b7d8-d6b51922d397" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5e043cfa-f07b-4f73-a9d2-e76a21840101" connectedTo="57dd5638-d1ba-4ed9-8efe-f1da25ea2f52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="cebfec93-245c-46b8-b9d7-68f7ef772f7c">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe87_ret" diameter="DN400" length="39.8" name="Pipe87" id="Pipe87" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03879819320885" lon="4.316698908805848"/>
          <point xsi:type="esdl:Point" lat="52.03900298138126" lon="4.317176342010499"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="ba4ce072-68d5-4603-aa4d-7ecf059f5020" id="6afb228c-79ec-408d-a489-5549a2b01a4d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9e6e3962-48a6-44b6-b4ed-4aa291015e64" connectedTo="3ae7bb41-1706-4b92-9b1d-69bf45707787" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="a9684373-d066-4f20-9ab6-c65842ef7122">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe88_ret" diameter="DN400" length="73.7" name="Pipe88" id="Pipe88" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.039451750181925" lon="4.316875934600831"/>
          <point xsi:type="esdl:Point" lat="52.03879819320885" lon="4.316698908805848"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7f48a3f7-45d9-4fef-bef3-ed371083a026" id="01bccc70-3665-4c0b-9494-a89ee374d1b1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="03764218-27f8-4596-a02c-ec0365d0890f" connectedTo="b36fcc81-7a66-4b77-9108-48ee1ed0fda6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="f265e487-11a8-4c5d-8639-00c91205008b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe89_ret" diameter="DN400" length="30.5" name="Pipe89" id="Pipe89" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03473314368742" lon="4.319311380386353"/>
          <point xsi:type="esdl:Point" lat="52.03460744879936" lon="4.318914413452149"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="72c1d1a9-8f49-4a26-b861-ec067835ef38" id="4c1cd0f9-9981-455f-bf4b-a6207be5c83e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="fd9d1bb3-97a4-449b-ac31-188a28a653a3" connectedTo="6d24513f-b153-4241-a5e6-813ce7a33ccf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="44a0af7a-b2ce-4438-a911-0c0708e16816">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe90_ret" diameter="DN400" length="46.5" name="Pipe90" id="Pipe90" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03433023856414" lon="4.319128990173341"/>
          <point xsi:type="esdl:Point" lat="52.03473314368742" lon="4.319311380386353"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7ad6ccee-4f96-44bb-aba0-7a6d719bd538" id="d7f08ab0-2d0e-4ee4-bf34-7e30a6625122" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="16cfc909-805a-4d30-bcbf-a0e703a6fef6" connectedTo="2d464b69-7340-4d13-b04e-8d642145a6fe" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2ff731ab-5a28-4dbd-8ce3-be6264f06bc5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe91_ret" diameter="DN400" length="38.1" name="Pipe91" id="Pipe91" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03287207234976" lon="4.320539832115174"/>
          <point xsi:type="esdl:Point" lat="52.03273294610615" lon="4.320030212402345"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4a82116b-fede-476f-9135-24d51f45f1b3" id="075fd612-bea2-4d9c-9731-dd32cda939a1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f119a0bf-e439-43dd-8a5c-29d84e45be5d" connectedTo="4213513b-398a-4350-bb50-38b447403f0a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="73126892-d39f-4285-83db-bee0e475e591">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe92_ret" diameter="DN400" length="54.5" name="Pipe92" id="Pipe92" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03238971879093" lon="4.32039499282837"/>
          <point xsi:type="esdl:Point" lat="52.03287207234976" lon="4.320539832115174"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="0e5966f7-a348-49b8-9c57-5c79356bd22f" id="9972a975-06e2-4452-89c0-2b428b97380c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="27795ecb-87b0-47e2-b93e-1dc3964d0352" connectedTo="b2547ccb-93f2-4c84-839d-0e34211aeece" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="f23c6b40-dccb-46df-b836-5225a1215cd1">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe23_a_ret" diameter="DN200" length="376.75" name="Pipe23_a" id="Pipe23_a" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03287207234976" lon="4.320539832115174"/>
          <point xsi:type="esdl:Point" lat="52.03092116196779" lon="4.322347640991212"/>
          <point xsi:type="esdl:Point" lat="52.031610847033264" lon="4.3238282203674325"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4a82116b-fede-476f-9135-24d51f45f1b3" id="c1318085-1ced-470d-b53e-6ac138f43364" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="49ac8cea-691e-47b6-a7d3-e1b5de85f30c" connectedTo="1073ff1b-9db1-4005-9c55-62655c11858c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="9b4da3dd-45dc-45f4-b8e7-949f80c105bc">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe23_b_ret" diameter="DN200" length="39.78" name="Pipe23_b" id="Pipe23_b" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.031610847033264" lon="4.3238282203674325"/>
          <point xsi:type="esdl:Point" lat="52.03182866846768" lon="4.324289560317994"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e99fd6b4-da2d-47bd-8a96-78aa58d72ea8" id="2a283272-bf19-4813-a9ca-9febca262907" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="6bfc5f80-e091-40d1-bc29-9d71281d5251" connectedTo="0148e3f8-6850-4ab4-a3d3-3e49ebd7e137" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="9b4da3dd-45dc-45f4-b8e7-949f80c105bc">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_e143" id="e14383da-75cf-4815-9d48-18b724f3c9ec">
        <geometry xsi:type="esdl:Point" lat="52.031610847033264" lon="4.3238282203674325"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="49ac8cea-691e-47b6-a7d3-e1b5de85f30c be5b79db-2172-49c1-a4ba-ecb58ed60542" id="1073ff1b-9db1-4005-9c55-62655c11858c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e99fd6b4-da2d-47bd-8a96-78aa58d72ea8" connectedTo="2a283272-bf19-4813-a9ca-9febca262907 37097359-90c9-467f-a622-f6ced809dc3f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe93_ret" diameter="DN400" length="61.7" name="Pipe93" id="Pipe93" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.031610847033264" lon="4.3238282203674325"/>
          <point xsi:type="esdl:Point" lat="52.03137322244006" lon="4.32464361190796"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e99fd6b4-da2d-47bd-8a96-78aa58d72ea8" id="37097359-90c9-467f-a622-f6ced809dc3f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="41716cca-e7a3-4f30-96c1-7a1db967a797" connectedTo="e9a48e2c-bace-41f7-ad38-e84678dfe345" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="a88a01fb-9056-4421-844d-e9cf40d2ee60">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe94_ret" diameter="DN400" length="69.0" name="Pipe94" id="Pipe94" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03217850067201" lon="4.323420524597169"/>
          <point xsi:type="esdl:Point" lat="52.031610847033264" lon="4.3238282203674325"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9fc6944d-648c-4294-9784-156f8156dd5f" id="625c9e58-0600-4305-9486-f4b31b6a001e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="be5b79db-2172-49c1-a4ba-ecb58ed60542" connectedTo="1073ff1b-9db1-4005-9c55-62655c11858c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="da0a90e9-12c9-4d32-b320-6432a6b5d507">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage17" id="7147409b-d66b-43ea-8c8e-3487487be9c8">
        <geometry xsi:type="esdl:Point" lat="52.04279097652879" lon="4.317948818206788" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="97d14595-b7d3-4796-892a-948e12e451a9" id="e4043477-c159-4288-8613-e54f84d1c68e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="1135996f-b4d2-48b7-905a-82be2f2415dd" connectedTo="2fbbe982-4037-4169-a088-28bd455d9da0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage18" id="6b3dd637-08ce-4c2e-a673-854bcd3f885e">
        <geometry xsi:type="esdl:Point" lat="52.041708724957495" lon="4.32389259338379" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="be858226-e092-4972-be04-f3ec3a886717" id="0b22f7b5-8f8b-4809-895d-55565bf9b185" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="340abfbb-4247-49d0-ae09-deff4c6598ce" connectedTo="172d79dd-fa55-4205-9987-60ddc13ef5a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage19" id="cb5f1992-b0cd-4d80-938f-52997cb95093">
        <geometry xsi:type="esdl:Point" lat="52.042744783399556" lon="4.326049089431764" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e28202cd-8322-4828-811b-de02bc77ac79" id="94a73820-d066-4dd5-a305-9c48f9eeabfa" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="48bfbe22-4e4b-4046-9944-fce648d16f83" connectedTo="d1189e55-b3de-4a66-8a2a-4bb173c3b9bb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage20" id="5b43a607-df3f-4837-a344-830e0008eb68">
        <geometry xsi:type="esdl:Point" lat="52.04411735887327" lon="4.3278729915618905" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d77bfbc6-bb84-44e4-a3a3-4cc17b1b4415" id="79d805dc-a306-4152-9fd6-4630a8152f3c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7757e190-4910-46ca-a000-024abb383b4a" connectedTo="f676d4da-6721-43b5-9bfd-655d86de67d7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage21" id="d5bdc779-b83f-4184-99ae-e2e893dc49fa">
        <geometry xsi:type="esdl:Point" lat="52.04629491616592" lon="4.330501556396485" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c85b795b-7dea-44ca-8adf-91c298c776e4" id="076887b6-4a86-44d3-87bd-831a272c22d6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="96a3052d-6237-49d0-85ed-bd65235522d0" connectedTo="103cda9c-f400-4499-ab0d-fe99b25e5c9d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage22" id="7c83f708-5085-4b3f-9636-515f45096824">
        <geometry xsi:type="esdl:Point" lat="52.04933672008261" lon="4.334310293197633" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="aab5d7f2-b26d-4d00-bc54-f277bb1862b8" id="739dfd0a-f73b-44fa-9d58-3637e3ef8f43" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="1c6756d2-1797-4221-94d7-e74888cdf112" connectedTo="7dbbd7a6-e225-4778-b65f-a8795c732295" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage23" id="f3ad0a2c-8453-46fc-b840-3ae75ab9395b">
        <geometry xsi:type="esdl:Point" lat="52.04835360003072" lon="4.337303638458253" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="497ba0ee-5326-43ea-bb4f-ca703171460d" id="59f90be4-8cbe-4767-b96b-91146c041ba7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="bec0449f-b062-49f5-b4bd-ba9aadf2a76c" connectedTo="d78f6cd1-f89d-41d7-adc0-986f67d9ac60" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage24" id="f46ac1f7-cade-45e8-902f-f4ee07e40c5f">
        <geometry xsi:type="esdl:Point" lat="52.05092682149594" lon="4.337571859359742" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="55819fd4-65f2-4831-bb0e-f4cb519f50b7" id="fa579b86-df33-4338-9867-3304c70c15b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a863d129-71e2-416d-8c1a-058a380dcd80" connectedTo="b88490b8-5cd4-4563-9903-13efe321a07d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage25" id="a3e41f47-1bc0-4907-a106-249e9dfe6ffb">
        <geometry xsi:type="esdl:Point" lat="52.05176473598928" lon="4.332733154296876" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="048f78db-b39b-444a-b0d8-4fdfe17eb2c7" id="75ec7e5c-8392-4938-92b5-67edba84ee5d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="44ad7156-ef46-4550-974c-fe86d2b34d73" connectedTo="3c8b30af-bfd3-44f9-9653-ac48cac85a32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage26" id="8aecf9f0-0076-49ed-9083-a4927b3a9a7f">
        <geometry xsi:type="esdl:Point" lat="52.05513602841179" lon="4.333258867263795" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b6bbce6a-9863-4129-a4b2-6e17d75dedd3" id="e12c2ace-055a-4e68-a00f-a542f914c919" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="38a7d6c4-b647-47c5-97df-3bfa3cd35fa8" connectedTo="ebc2e9bc-2d17-4398-a182-9eb0026fef52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer17" id="2f22b7c6-b7ce-43e6-a1dd-e40c5107256a">
        <geometry xsi:type="esdl:Point" lat="52.054872139943235" lon="4.333677291870118" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7d25b81b-272d-4eb7-86d6-c8d68fbf7981" connectedTo="13b89cff-7987-4998-9766-ee3e9bc10b40" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="8410eb59-05a7-4f17-aca6-9cca0adcaf05" id="49c5ab59-21bf-4d65-9071-1c73a003cfa5" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer18" id="6893005c-2c61-45d6-93ca-bf73c729bb0d">
        <geometry xsi:type="esdl:Point" lat="52.0524113049362" lon="4.332432746887208" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c83f1338-1b0b-4fcb-ac64-cf19d41be72b" connectedTo="d8b610a7-c87e-4aa9-b7a6-edb55f3a8ac6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="32a75cd3-df73-4797-988c-af0b84e781c3" id="b059f601-3773-4380-bc23-34cc480bc769" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer19" id="37208c67-8730-4f71-8d9f-c9a0f9f56725">
        <geometry xsi:type="esdl:Point" lat="52.051395263818755" lon="4.336906671524049" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="2cb2ce2c-d1bd-4c17-8a73-9b8012f2262e" connectedTo="4477c893-8e56-40cd-a669-fd50ff57698d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="fc239a2f-c737-47ee-89c3-c69baed28e51" id="d4012c03-8dfd-4a9f-8674-9774f1dd612b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer20" id="b002a257-64f7-4eb9-b317-839b88035b72">
        <geometry xsi:type="esdl:Point" lat="52.048564741731425" lon="4.336574077606202" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="24d2b55c-4822-4524-9b2a-ebe99070fce1" connectedTo="6532545a-8a96-4a08-bef4-a1b9bd024812" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e238321b-26ad-4b91-8314-7e1624b36a94" id="9548a1f0-f3eb-4a95-a592-2356ab97130c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer21" id="7e9c28ea-1d36-4db7-8e26-247e59a8d5b7">
        <geometry xsi:type="esdl:Point" lat="52.04979198311488" lon="4.335168600082398" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e376e0fc-ed34-4e24-bbb4-37f69445f7a2" connectedTo="8a56e2ae-2090-4578-928a-c5beaf608a32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f2d22a19-aaeb-418e-8bcb-cdc789d3b956" id="44e1a886-5a4e-41c8-8921-4cb53eef3dc6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer22" id="b9f7e29d-062e-4b89-bc64-ecfccb9031eb">
        <geometry xsi:type="esdl:Point" lat="52.04679639917788" lon="4.331016540527345" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="289cccc1-0ce6-4d43-92f8-4cbe8246476e" connectedTo="87f1d839-d3a5-4660-a4d4-5280d7c4d5b0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3800b6d2-cfdb-4dd6-8cff-72156de139fe" id="95556d67-1ffd-4d8e-afb1-83c6e63e8f5f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer23" id="27824581-94aa-416c-8242-59c85dc0cad9">
        <geometry xsi:type="esdl:Point" lat="52.04389959730849" lon="4.3280982971191415" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7e0e66f1-effe-44c1-b08c-e4628e1511f7" connectedTo="4c9b284f-9a0b-4fe6-8d6c-34343d76e784" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="dc290931-0cdf-444e-9f3a-dc8b52ff4920" id="4ffa71aa-a584-41e1-97a1-61199da28891" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer24" id="3573c751-4116-4600-b261-060fc75f230f">
        <geometry xsi:type="esdl:Point" lat="52.04312092606362" lon="4.326682090759278" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="87aab61f-344d-4e1e-a560-b3a1725d2ba8" connectedTo="52349e83-008e-42eb-b009-8d8dee21bd7d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="60b1043e-22bd-4119-98b7-b282144c0967" id="f572e6a0-69de-40e5-ba67-cca9d0eb7452" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer25" id="9ad2ed86-e8bd-4328-a3c5-15293b4905a6">
        <geometry xsi:type="esdl:Point" lat="52.039682732367204" lon="4.322293996810914" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5b0da039-78a0-41a8-98ec-3ca85ff090d5" connectedTo="56900fea-7b65-4f37-9d17-aa99dbb72813" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="47411f3a-7951-4684-bc58-dbbc2f86147d" id="b95748d5-64e4-46fd-9dbc-0b3528f6e3ff" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer26" id="2ce79393-01d9-4fe1-8ea2-ed8fb8750403">
        <geometry xsi:type="esdl:Point" lat="52.04256001040516" lon="4.318131208419801" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="1241c02e-13e9-4f66-ac45-12344bb3bd47" connectedTo="9cf14369-e908-46c7-95e7-467c79c14198" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e14b3d48-c8d0-47ce-af2e-bcd5173687cd" id="9d08b748-1db7-42fd-ae46-4645ed9b83fd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatProducer" name="HeatProducer27" id="a28e3d9b-6643-4fc8-a427-26090071f814">
        <geometry xsi:type="esdl:Point" lat="52.041378764997596" lon="4.324654340744019" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="Out" id="6fea1bf7-5c9a-4d80-b0b3-1a83b1853b10" connectedTo="6cb7b677-a823-477b-ae45-6dc4b931e9be" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b65eee76-8d5c-46f5-aa26-6aa060435ec2" id="04f3033e-e363-4f74-bb59-c1de3f1af911" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="100000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MW" perUnit="WATT" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </investmentCosts>
          <variableOperationalAndMaintenanceCosts xsi:type="esdl:SingleValue" id="f9c3998f-dfcf-4fa1-860f-dfaaf91e7e9c" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/MWh" perUnit="WATTHOUR" id="d3ef5c4c-12b1-4851-af0f-04c95e210be9" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalAndMaintenanceCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300000.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe95_ret" diameter="DN400" length="36.4" name="Pipe95" id="Pipe95" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.042894338333426" lon="4.318453073501588"/>
          <point xsi:type="esdl:Point" lat="52.04279097652879" lon="4.317948818206788"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="679d18da-f447-41dd-916f-e5d3b29d1184" id="e3603d73-e225-49bc-aef9-dc6b2c67a3f1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="97d14595-b7d3-4796-892a-948e12e451a9" connectedTo="e4043477-c159-4288-8613-e54f84d1c68e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="842665c8-9133-4eaf-aa34-f58a156cb83d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe96_ret" diameter="DN400" length="43.2" name="Pipe96" id="Pipe96" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04256001040516" lon="4.318131208419801"/>
          <point xsi:type="esdl:Point" lat="52.042894338333426" lon="4.318453073501588"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="1241c02e-13e9-4f66-ac45-12344bb3bd47" id="9cf14369-e908-46c7-95e7-467c79c14198" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f039081a-6763-4917-bd96-9ec83518efaa" connectedTo="034e5cc7-68f3-4c88-8be9-31902d03d441" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="572a93a7-6911-4896-aff2-a50c20ee10cd">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe97_ret" diameter="DN400" length="52.6" name="Pipe97" id="Pipe97" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.039682732367204" lon="4.322293996810914"/>
          <point xsi:type="esdl:Point" lat="52.03995144351459" lon="4.321660995483399"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="5b0da039-78a0-41a8-98ec-3ca85ff090d5" id="56900fea-7b65-4f37-9d17-aa99dbb72813" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b3330cdc-ce06-4264-9d30-b5fc3d2e657c" connectedTo="eede8573-c3f5-4156-baea-f8ef53a002a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="af80498a-df27-42b0-90e0-401018a3dc08">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe98_ret" diameter="DN400" length="83.1" name="Pipe98" id="Pipe98" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04102040978031" lon="4.323420524597169"/>
          <point xsi:type="esdl:Point" lat="52.041708724957495" lon="4.32389259338379"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="3f8246ac-5bf4-4dff-b00d-f5e2e956c2b9" id="5900a9f1-f801-437d-aabd-7d762c8135d1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="be858226-e092-4972-be04-f3ec3a886717" connectedTo="0b22f7b5-8f8b-4809-895d-55565bf9b185" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="51723c88-4479-4774-b45c-d85c9543e4c1">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe99_ret" diameter="DN400" length="93.3" name="Pipe99" id="Pipe99" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.041378764997596" lon="4.324654340744019"/>
          <point xsi:type="esdl:Point" lat="52.04102040978031" lon="4.323420524597169"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="6fea1bf7-5c9a-4d80-b0b3-1a83b1853b10" id="6cb7b677-a823-477b-ae45-6dc4b931e9be" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c56dbff6-43d0-424e-82af-647304a719ef" connectedTo="0f1d6205-a978-46a7-8c51-3aa7e22d6ac3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="77009f21-f5a4-4962-ac95-fbf5a29641dd">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe100_ret" diameter="DN400" length="95.3" name="Pipe100" id="Pipe100" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04235988080374" lon="4.327293634414674"/>
          <point xsi:type="esdl:Point" lat="52.042744783399556" lon="4.326049089431764"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7fc22ee8-6e17-4a43-857c-d5d35bf111cb" id="d6506a36-4356-4ac8-8964-0afab3b35e21" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="e28202cd-8322-4828-811b-de02bc77ac79" connectedTo="94a73820-d066-4dd5-a305-9c48f9eeabfa" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="eefab07c-d1b2-4e40-9424-d645482e7d6b">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe101_ret" diameter="DN400" length="94.4" name="Pipe101" id="Pipe101" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04312092606362" lon="4.326682090759278"/>
          <point xsi:type="esdl:Point" lat="52.04235988080374" lon="4.327293634414674"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="87aab61f-344d-4e1e-a560-b3a1725d2ba8" id="52349e83-008e-42eb-b009-8d8dee21bd7d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7f7fd672-8ea1-45cb-baf0-ac9821ea412f" connectedTo="5de6a1ad-fd67-4b43-8fdf-70d74d6fd0bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2b34cb99-0ac1-4fb1-ad21-fc66e3f1a2fa">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe102_ret" diameter="DN400" length="116.7" name="Pipe102" id="Pipe102" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04362013462266" lon="4.32937502861023"/>
          <point xsi:type="esdl:Point" lat="52.04411735887327" lon="4.3278729915618905"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b85c3b55-c290-4987-93b3-6c03bde6e8b4" id="55a260e8-8191-460a-91aa-b04cdb701a77" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="d77bfbc6-bb84-44e4-a3a3-4cc17b1b4415" connectedTo="79d805dc-a306-4152-9fd6-4630a8152f3c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c49679bf-831c-45a0-ac90-9fc65043d88e">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe103_ret" diameter="DN400" length="92.7" name="Pipe103" id="Pipe103" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04389959730849" lon="4.3280982971191415"/>
          <point xsi:type="esdl:Point" lat="52.04362013462266" lon="4.32937502861023"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7e0e66f1-effe-44c1-b08c-e4628e1511f7" id="4c9b284f-9a0b-4fe6-8d6c-34343d76e784" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="7611d55e-8f95-482e-9014-c57e8f3e53dd" connectedTo="599d0c0b-c7e2-4314-bc5c-da112c9da4c1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="9eca3dd8-35d9-431a-b9ea-81e27ebaba65">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe104_ret" diameter="DN400" length="32.2" name="Pipe104" id="Pipe104" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04658258087284" lon="4.330555200576783"/>
          <point xsi:type="esdl:Point" lat="52.04629491616592" lon="4.330501556396485"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="aae35ef1-f784-489b-9e36-f0b903ca948d" id="e2b27087-2c20-4082-9c83-6ea731ca7d21" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c85b795b-7dea-44ca-8adf-91c298c776e4" connectedTo="076887b6-4a86-44d3-87bd-831a272c22d6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="e304d5ef-4205-49dc-a01f-e4ab0bf5f704">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe105_ret" diameter="DN400" length="39.5" name="Pipe105" id="Pipe105" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04679639917788" lon="4.331016540527345"/>
          <point xsi:type="esdl:Point" lat="52.04658258087284" lon="4.330555200576783"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="289cccc1-0ce6-4d43-92f8-4cbe8246476e" id="87f1d839-d3a5-4660-a4d4-5280d7c4d5b0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4880facf-4871-4773-a1ac-0e48df8eca65" connectedTo="90cc9947-a05b-41c5-8b9e-e678bf033c61" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="a8601187-2eaa-49fe-85ab-baef7ad13c93">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe106_ret" diameter="DN400" length="78.2" name="Pipe106" id="Pipe106" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.050039619503906" lon="4.334267377853394"/>
          <point xsi:type="esdl:Point" lat="52.04933672008261" lon="4.334310293197633"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="a20b67c3-c88a-49a1-8428-01424efffc51" id="f5b9c2e0-4e88-463c-89dd-f3197e9eae02" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="aab5d7f2-b26d-4d00-bc54-f277bb1862b8" connectedTo="739dfd0a-f73b-44fa-9d58-3637e3ef8f43" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3c5fa53e-cb66-4968-b70b-c7af866798f3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe107_ret" diameter="DN400" length="67.5" name="Pipe107" id="Pipe107" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04979198311488" lon="4.335168600082398"/>
          <point xsi:type="esdl:Point" lat="52.050039619503906" lon="4.334267377853394"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="e376e0fc-ed34-4e24-bbb4-37f69445f7a2" id="8a56e2ae-2090-4578-928a-c5beaf608a32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a82e0ed6-fc80-4319-820c-7de68a337cd3" connectedTo="29b4f6b2-45e0-452d-82ed-0a1507fa40f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="51f19a21-b32d-4cc7-a4af-8d6f8d8543d5">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe108_ret" diameter="DN400" length="58.1" name="Pipe108" id="Pipe108" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0507784938338" lon="4.336756467819215"/>
          <point xsi:type="esdl:Point" lat="52.05092682149594" lon="4.337571859359742"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d906ef7e-682b-484e-82ac-584a0115c91e" id="edd5a51a-c5a8-4e2a-bc61-f8b0d031aa11" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="55819fd4-65f2-4831-bb0e-f4cb519f50b7" connectedTo="fa579b86-df33-4338-9867-3304c70c15b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3c83394b-f6ab-4673-bd95-3edd005f6a67">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe109_ret" diameter="DN400" length="69.3" name="Pipe109" id="Pipe109" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.051395263818755" lon="4.336906671524049"/>
          <point xsi:type="esdl:Point" lat="52.0507784938338" lon="4.336756467819215"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="2cb2ce2c-d1bd-4c17-8a73-9b8012f2262e" id="4477c893-8e56-40cd-a669-fd50ff57698d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="a84b4b4b-316c-44e9-8aaa-8af5b97c4506" connectedTo="49caaf6b-e676-4e65-9ce5-027c2406c9b7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="8c77194c-c815-43a9-8cd5-4b4ab70513ea">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe49_a_ret" diameter="DN200" length="277.05" name="Pipe49_a" id="Pipe49_a" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0507784938338" lon="4.336756467819215"/>
          <point xsi:type="esdl:Point" lat="52.04933476113613" lon="4.338204860687257"/>
          <point xsi:type="esdl:Point" lat="52.04872309735226" lon="4.33737874031067"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="d906ef7e-682b-484e-82ac-584a0115c91e" id="5f245a7a-bb9c-4c6d-af8c-05a27a76f17c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f24c0824-3758-4db7-983d-6417cfd87c8b" connectedTo="b0b8f904-3e41-417b-8dff-7b98d9c823e8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="606cd7cf-cb7c-4852-9d18-225198069dd3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe49_b_ret" diameter="DN200" length="47.47" name="Pipe49_b" id="Pipe49_b" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04872309735226" lon="4.33737874031067"/>
          <point xsi:type="esdl:Point" lat="52.04866183648353" lon="4.337325096130372"/>
          <point xsi:type="esdl:Point" lat="52.04850966015021" lon="4.337850809097291"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b6818887-3419-4627-80f7-dfefeed63fbb" id="2de5f269-8e5e-41c8-9c2b-808bd66dcbac" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="8cf61346-445d-44e1-bd5f-02b51b9af83c" connectedTo="f470c680-16ff-449c-9c6d-58e59dd3b478" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="606cd7cf-cb7c-4852-9d18-225198069dd3">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_458a" id="458a8a43-c143-45fc-95db-54b2d2c6b15a">
        <geometry xsi:type="esdl:Point" lat="52.04872309735226" lon="4.33737874031067"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f24c0824-3758-4db7-983d-6417cfd87c8b b73b25b0-e282-4901-b880-3bf568c03247" id="b0b8f904-3e41-417b-8dff-7b98d9c823e8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b6818887-3419-4627-80f7-dfefeed63fbb" connectedTo="2de5f269-8e5e-41c8-9c2b-808bd66dcbac 6708f9ae-b61c-4480-ab5b-ddd3769c1d4a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe52_a_ret" diameter="DN200" length="151.23" name="Pipe52_a" id="Pipe52_a" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05278394828437" lon="4.334599971771241"/>
          <point xsi:type="esdl:Point" lat="52.052121009831716" lon="4.3326687812805185"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9a8102ad-8172-41f1-9939-ff995a85d39e" id="2e30acdb-3a7d-4c27-8917-127592c0c3a3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="0577fd68-0ca6-4e1f-a386-b1b277e11aa2" connectedTo="c7cd8fb2-241f-45db-b462-59ebbf9d4e52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c312ce41-0938-4eca-8af8-20f0413300f9">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe52_b_ret" diameter="DN200" length="27.64" name="Pipe52_b" id="Pipe52_b" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.052121009831716" lon="4.3326687812805185"/>
          <point xsi:type="esdl:Point" lat="52.05197974342533" lon="4.332336187362672"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="174cd399-bf91-45c2-b3f3-134704467f3f" id="8313b9e4-56f4-4381-89bb-85bdcaecce29" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="5802017c-c2da-48c5-88df-b23fc4090720" connectedTo="02fddd61-e3fb-4ed9-aa92-08694de98345" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="c312ce41-0938-4eca-8af8-20f0413300f9">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_0969" id="0969517d-7b44-4de8-ba13-8ac30a59638c">
        <geometry xsi:type="esdl:Point" lat="52.052121009831716" lon="4.3326687812805185"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="0577fd68-0ca6-4e1f-a386-b1b277e11aa2 f7a61718-b55d-4ec1-816e-f397f8a253c8" id="c7cd8fb2-241f-45db-b462-59ebbf9d4e52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="174cd399-bf91-45c2-b3f3-134704467f3f" connectedTo="8313b9e4-56f4-4381-89bb-85bdcaecce29 8e42092b-63cc-4545-b872-cae3c1e5a319" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe51_a_ret" diameter="DN200" length="249.1" name="Pipe51_a" id="Pipe51_a" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05278394828437" lon="4.334599971771241"/>
          <point xsi:type="esdl:Point" lat="52.05472700062111" lon="4.332786798477174"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9a8102ad-8172-41f1-9939-ff995a85d39e" id="cf93b38e-49d4-41e0-8f2e-cae0e61e47bb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="914edb28-1bd6-443b-a11e-ec2936417159" connectedTo="33170f9d-b99a-41ee-9670-d405967bf293" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="582a88fb-6f00-4922-8b11-33729c4e17a8">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe51_b_ret" diameter="DN200" length="34.31" name="Pipe51_b" id="Pipe51_b" outerDiameter="0.315" innerDiameter="0.2101">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05472700062111" lon="4.332786798477174"/>
          <point xsi:type="esdl:Point" lat="52.054987810473264" lon="4.3325185775756845"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0045">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="930aa5cf-b76e-4049-afa7-ea79445faf55"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.04385">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="f6bd7242-b1a3-4b24-9edd-ad58a830444b"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0041">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="81df81a9-ac8b-4c9d-8d71-dd2bbee92fa3"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="94808684-1fc7-46ef-91e9-15e142c01bc0" id="34179142-5154-4df5-b501-ed9af4cf3481" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="78aa70aa-e167-4cc8-93e9-d4c1ca8d52df" connectedTo="3e8bbc50-eb90-4c5d-9035-f77f38b544c4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="582a88fb-6f00-4922-8b11-33729c4e17a8">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="1355.3" id="b887af7c-612e-4899-b64a-01ad7ed76cf4">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="fa8f44a4-599f-45be-ad50-353b4c8c116f" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_4f81" id="4f817fa9-bd0d-46e3-a988-3cb32a862f4d">
        <geometry xsi:type="esdl:Point" lat="52.05472700062111" lon="4.332786798477174"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="914edb28-1bd6-443b-a11e-ec2936417159 4efde9b3-d804-4bf4-8032-99d8014754f9" id="33170f9d-b99a-41ee-9670-d405967bf293" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="94808684-1fc7-46ef-91e9-15e142c01bc0" connectedTo="34179142-5154-4df5-b501-ed9af4cf3481 fd884acd-a54e-400a-8ca9-4bed70d4910d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe110_ret" diameter="DN400" length="63.0" name="Pipe110" id="Pipe110" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.054872139943235" lon="4.333677291870118"/>
          <point xsi:type="esdl:Point" lat="52.05472700062111" lon="4.332786798477174"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="7d25b81b-272d-4eb7-86d6-c8d68fbf7981" id="13b89cff-7987-4998-9766-ee3e9bc10b40" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="4efde9b3-d804-4bf4-8032-99d8014754f9" connectedTo="33170f9d-b99a-41ee-9670-d405967bf293" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="f24be569-b0ca-479c-9113-e5aa80f5a830">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe111_ret" diameter="DN400" length="55.8" name="Pipe111" id="Pipe111" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.05472700062111" lon="4.332786798477174"/>
          <point xsi:type="esdl:Point" lat="52.05513602841179" lon="4.333258867263795"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="94808684-1fc7-46ef-91e9-15e142c01bc0" id="fd884acd-a54e-400a-8ca9-4bed70d4910d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b6bbce6a-9863-4129-a4b2-6e17d75dedd3" connectedTo="e12c2ace-055a-4e68-a00f-a542f914c919" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="2d3f8d20-7731-4cf1-9271-838226d3aa59">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe112_ret" diameter="DN400" length="39.9" name="Pipe112" id="Pipe112" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.052121009831716" lon="4.3326687812805185"/>
          <point xsi:type="esdl:Point" lat="52.05176473598928" lon="4.332733154296876"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="174cd399-bf91-45c2-b3f3-134704467f3f" id="8e42092b-63cc-4545-b872-cae3c1e5a319" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="048f78db-b39b-444a-b0d8-4fdfe17eb2c7" connectedTo="75ec7e5c-8392-4938-92b5-67edba84ee5d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="de9e70a5-4f34-44fc-a793-a7efe3252057">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe113_ret" diameter="DN400" length="36.1" name="Pipe113" id="Pipe113" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.0524113049362" lon="4.332432746887208"/>
          <point xsi:type="esdl:Point" lat="52.052121009831716" lon="4.3326687812805185"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="c83f1338-1b0b-4fcb-ac64-cf19d41be72b" id="d8b610a7-c87e-4aa9-b7a6-edb55f3a8ac6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="f7a61718-b55d-4ec1-816e-f397f8a253c8" connectedTo="c7cd8fb2-241f-45db-b462-59ebbf9d4e52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="ee408af4-6eb5-4e5d-923a-d5e3be14c6bb">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe114_ret" diameter="DN400" length="41.4" name="Pipe114" id="Pipe114" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04872309735226" lon="4.33737874031067"/>
          <point xsi:type="esdl:Point" lat="52.04835360003072" lon="4.337303638458253"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="b6818887-3419-4627-80f7-dfefeed63fbb" id="6708f9ae-b61c-4480-ab5b-ddd3769c1d4a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="497ba0ee-5326-43ea-bb4f-ca703171460d" connectedTo="59f90be4-8cbe-4767-b96b-91146c041ba7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="80bee82b-78f3-4119-adda-4cfa2e5b2c2d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe115_ret" diameter="DN400" length="57.8" name="Pipe115" id="Pipe115" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.048564741731425" lon="4.336574077606202"/>
          <point xsi:type="esdl:Point" lat="52.04872309735226" lon="4.33737874031067"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="24d2b55c-4822-4524-9b2a-ebe99070fce1" id="6532545a-8a96-4a08-bef4-a1b9bd024812" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="b73b25b0-e282-4901-b880-3bf568c03247" connectedTo="b0b8f904-3e41-417b-8dff-7b98d9c823e8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="b25bd430-d990-4bd2-b615-3138e85c2b9d">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:HeatStorage" name="HeatStorage27" id="be5b56ed-765b-4624-9c07-fc0a26f6a481">
        <geometry xsi:type="esdl:Point" lat="52.03982792055847" lon="4.321199655532838" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="9ba6a0f8-d8e5-4e0e-a4df-2a6f7db320f1" id="55524d39-e69e-4020-9543-64509c16bcff" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="20653a96-e8e6-40fd-9f5c-fc2d1bb4cc86" connectedTo="5671f07e-0cf8-4a47-8a05-32b05b01bd8b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <costInformation xsi:type="esdl:CostInformation" id="f6ab2a6d-41d5-4e10-b917-e9c443151624">
          <investmentCosts xsi:type="esdl:SingleValue" id="47fd41e1-9017-466e-b6a1-5ea7f776dfa5" value="690.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Cost in EUR/m3" perUnit="CUBIC_METRE" id="ff0a7278-91de-4ca4-82a7-25ae9d63e8e1" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="cc47852c-c7cc-44b3-abd5-5e14378489e8" value="300.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="164720ce-b344-482b-8fb2-9fe2938cfde1" unit="EURO" description="Cost in EUR" physicalQuantity="COST"/>
          </installationCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe116_ret" diameter="DN400" length="34.4" name="Pipe116" id="Pipe116" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.03995144351459" lon="4.321660995483399"/>
          <point xsi:type="esdl:Point" lat="52.03982792055847" lon="4.321199655532838"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="4cac6a54-6b27-456e-8d17-f5910fb4744c" id="c72bf1cd-6be8-47ab-9288-d4017b440783" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="9ba6a0f8-d8e5-4e0e-a4df-2a6f7db320f1" connectedTo="55524d39-e69e-4020-9543-64509c16bcff" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="95feefab-d0ae-4cb8-b973-21f6c7b28572">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:ATES" name="ATES" id="d868a6b0-c2c6-4266-8130-9c2e6a3facee" maxDischargeRate="1.0" maxChargeRate="1.0">
        <geometry xsi:type="esdl:Point" lat="52.044472868844785" lon="4.3098217248916635" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="In" connectedTo="f2dc36cf-a2ab-41e1-933a-dfe5a1d8c118" id="413a8ce6-4b2e-442c-acca-21b5d928130c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out" id="885f141c-5637-42e1-bf5b-2a42c628f75d" connectedTo="74d1b153-1824-441b-9173-5ebe9613e58c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <costInformation xsi:type="esdl:CostInformation" id="2b5e75d5-a91f-45fa-9ce5-347fe365a04b">
          <variableOperationalCosts xsi:type="esdl:SingleValue" id="463aa0ae-0748-4523-be52-91142442d3f5" value="0.685">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="a754b88e-46f6-4921-bc09-5fcefc560a43" description="Cost in EUR/MWh" perUnit="WATTHOUR" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </variableOperationalCosts>
          <installationCosts xsi:type="esdl:SingleValue" id="b949af06-2525-4103-a362-da3182a8a7cc" value="200.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="e6d88dd0-91c0-4681-8d37-ada0fd1d422c" description="Cost in EUR/MW" perUnit="WATT" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </installationCosts>
          <fixedOperationalCosts xsi:type="esdl:SingleValue" id="af8e95a1-1f2d-44a1-aff1-d495e6ed082c" value="2584.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="eb9e4398-d6ce-4650-8686-551531d7e112" description="Cost in EUR/MW" perUnit="WATT" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </fixedOperationalCosts>
          <fixedMaintenanceCosts xsi:type="esdl:SingleValue" id="e670bb8a-d45c-4603-98a9-225d5db58458" value="9945.0">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" id="1e4f7346-cbd2-4e8e-8b69-718b1f94b4c5" description="Cost in EUR/MW" perUnit="WATT" unit="EURO" perMultiplier="MEGA" physicalQuantity="COST"/>
          </fixedMaintenanceCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe117_ret" diameter="DN400" length="24.1" name="Pipe117" id="Pipe117" outerDiameter="0.56" innerDiameter="0.3938">
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lat="52.04446956946128" lon="4.309845194220544"/>
          <point xsi:type="esdl:Point" lat="52.044399045081015" lon="4.3101777881383905"/>
        </geometry>
        <material xsi:type="esdl:CompoundMatter" compoundType="LAYERED">
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0063">
            <matter xsi:type="esdl:Material" thermalConductivity="52.15" name="steel" id="74d84321-1767-4cec-b6e7-b90c06020400"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0711">
            <matter xsi:type="esdl:Material" thermalConductivity="0.027" name="PUR" id="95012cd9-7648-4df5-8c83-74dc73bc16ba"/>
          </component>
          <component xsi:type="esdl:CompoundMatterComponent" layerWidth="0.0057">
            <matter xsi:type="esdl:Material" thermalConductivity="0.4" name="HDPE" id="1392ee3f-34f6-4c8e-ab0e-635b9d7ec9be"/>
          </component>
        </material>
        <port xsi:type="esdl:InPort" name="In" connectedTo="885f141c-5637-42e1-bf5b-2a42c628f75d" id="74d1b153-1824-441b-9173-5ebe9613e58c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <port xsi:type="esdl:OutPort" name="Out" id="c668dd7b-24f6-4e05-97b1-5ba65a743f93" connectedTo="d44b1d62-d2be-482a-9c1b-29c7a5073711" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75"/>
        <dataSource xsi:type="esdl:DataSource" name="Logstor Product Catalogue Version 2020.03" attribution="https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf"/>
        <costInformation xsi:type="esdl:CostInformation" id="3929d3b9-cbd1-4ad7-a363-e36053c8ab8a">
          <investmentCosts xsi:type="esdl:SingleValue" name="Combined investment and installation costs" value="2840.6" id="2ab2214c-8d42-4b74-877f-7fb97f6eab86">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Costs in EUR/m" perUnit="METRE" id="9169bd50-197f-4d6b-aaac-b383a59c815d" unit="EURO" physicalQuantity="COST"/>
          </investmentCosts>
        </costInformation>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_8cee_ret" id="e2320284-e0d5-4b07-be58-7a2abdf6154c">
        <geometry xsi:type="esdl:Point" lat="52.041739007433215" lon="4.312164208605956" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="684abfb7-4218-4a1e-b6da-a822ff112397 1ef2a05a-afb5-445a-8563-9a14af9b9f13 4b08c6c4-14d3-4715-9f96-e04ab147c9c1" id="675d388d-ece7-48c3-bbc5-e4b73b9510f3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="aa141fc0-ce28-4d14-99be-bed98630e2e5" connectedTo="c07eaec4-b994-4ff1-ba68-fa2452a8e424 6d43e535-b95f-41ea-8489-7c85d74ff650" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_bdf0_ret" id="5ae6c05d-c6e6-4beb-b403-4c903d9fea46">
        <geometry xsi:type="esdl:Point" lat="52.0444861582057" lon="4.309578632886089" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="8fb322ac-0b28-4cf2-b01e-8728a9646a65 d01a9f0c-09dc-44db-8937-2f651ce3fb85 04e44cf5-08e8-45e2-a7ce-0c10dcfbe9ee" id="6df6baf1-2c39-49c3-ad57-1911dd437fdf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="d49a9b33-62b9-43ee-aa3c-19e087899111" connectedTo="e49b7e3c-9fec-4b5a-ac37-9b7266342b35 a6b22da3-d60a-4617-8df7-5ed7fb04ea05 d4c7ba6d-d591-4fe9-8436-b9aaa334e1db" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_5163_ret" id="49388423-ec24-4ea8-8c6b-4d69f800a6cc">
        <geometry xsi:type="esdl:Point" lat="52.038888193298845" lon="4.316066207877817" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="401eae5a-7174-4c37-b64d-33c87782ed60" connectedTo="5967e711-a8e1-4fe6-8ef3-fbe16296ddaf da149f20-9c37-46ca-b006-c2b63a1c6e1f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="d6a89042-7b6f-4d2a-9408-d36ea437ec1e 8cca0c12-c7b2-45ed-9d3d-8353a5c9bf52 8681bff7-98ab-42ff-9d8b-8eb4f8df2ee6" id="2ec8715c-c5b9-4782-918c-f1b35d426904" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_828a_ret" id="d64b5013-193b-4470-9d4f-604b46722cd6">
        <geometry xsi:type="esdl:Point" lat="52.03775978317298" lon="4.317623756910878" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="51a9b7c1-021f-41bc-8690-1870c52fc0d0" connectedTo="a351aea6-ee34-4b0f-b0ba-388def8927a4 cbe50ae5-e435-4273-863b-4cdb724d77a7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="e1c36f63-4988-4e3c-8519-708b6ae10e86 15ed4e8a-5d45-4272-a2a1-5a0eb66587c3 69832a5b-c670-4a1a-8435-76788055420e" id="63920326-77bb-4ce1-aa31-bba537b8ec57" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1514_ret" id="b648c052-7277-45f6-97a2-80ef64dd1811">
        <geometry xsi:type="esdl:Point" lat="52.034823143777416" lon="4.318665895348221" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="2dcc99cf-eeea-47fd-92d9-e35cbd77a878 7a9f9f48-98cd-4bb1-a658-66538e9825ec 55abdde6-75c5-4fd7-9ceb-a4f7fe531e03" id="28b38730-7167-476f-8962-7d09d9f9d417" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="6f78a5f5-edd6-4641-afce-1f57fdb304b3" connectedTo="7510d944-54c6-4d1b-98f4-2485da00c854 aa2d568a-0e68-41a6-9ba1-9adcf0d9dede" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_217a_ret" id="6aaff62b-3e8f-40b7-995a-46933b91321a">
        <geometry xsi:type="esdl:Point" lat="52.03296207243976" lon="4.319888316571063" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="2a669cbf-4402-415f-9580-6f7bd49dafb3 36c80210-eb59-4981-a9dd-6c707c321160 47825b4e-4273-4b51-8c55-f6200eb7253d" id="425cb5ab-fe83-4313-82bc-6ea70546c77e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="122ee1ba-078e-4d41-8525-426ff44fbef8" connectedTo="c5ce9a46-6831-4a59-b2fc-feb0284949ed 407c747e-5e0f-4a08-84aa-6ffb2a6eb7ca" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_20e0_ret" id="2ef869dc-8913-4306-8fd4-15d1fb0e5b52">
        <geometry xsi:type="esdl:Point" lat="52.02614679831976" lon="4.313626398166783" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="3dda78e8-4980-4ac2-b97b-2a9b02909037 14826ed9-07be-48aa-81a3-01fdc1bdf5da cd2edc0a-a86a-43aa-8758-2583a5d1b953" id="7ac9b957-d251-4415-afb0-42c89590fb47" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="cd825ba8-0050-4b8b-96e8-511f6b1e025c" connectedTo="8914c40c-a08d-4a85-a893-4a796d062bfa de53da75-834f-4750-8776-49bbb4f510f1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_72e3_ret" id="5f82d889-44cc-4d40-b9c1-7f174b8aaa71">
        <geometry xsi:type="esdl:Point" lat="52.02810055900395" lon="4.311809287259864" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="cb91c93d-5a0b-4ac4-b4f4-4afd183b2a38 047ee5bd-2099-442f-b98b-552a081d039a" id="9ba3845f-c482-40bf-ac11-a20b37ca25df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="e526b492-b974-44a8-835b-53e8ffdb0e1b" connectedTo="9ff5ba8a-a36a-4a52-9f06-9ffc9bec6631" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_aae6_ret" id="17ad72e7-48ad-4e18-9233-22b95f764a75">
        <geometry xsi:type="esdl:Point" lat="52.02905100631209" lon="4.31264939009626" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="f48d27c5-cb99-45ed-bdf6-ac8594944ba9 a62cd13e-03bb-4323-b96e-5ac83870b099" id="cb5b59bf-b5a2-4948-ba3b-cbba2aea383b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="4a07801a-c279-4e37-8c21-ecbeb8248b20" connectedTo="e1ae80f8-f763-4f93-bf88-9977684161be" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_839c_ret" id="b394a8e0-475b-4a49-92e5-65f3bd98dbde">
        <geometry xsi:type="esdl:Point" lat="52.0313082377085" lon="4.310639968040655" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="f240ed4d-a9c6-4b70-9573-be6bb9330487 ec61db12-3e07-4b5b-834a-4ccdb5b97837 5467a867-c193-4d61-925a-7c84200a8eaf" id="7995eac2-82c7-4456-8f8e-a4fa46c783a2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="dc94e82b-2d27-4593-bf45-313f72814379" connectedTo="8b26c9ef-4a71-438c-8de9-d5a6b012d71a 1308055c-ce56-406f-bad6-5df7eac7e0e3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_b466_ret" id="23ddf095-9825-4914-b676-6ab046e9900a">
        <geometry xsi:type="esdl:Point" lat="52.03188902735578" lon="4.3101054529409275" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="dc4fee0c-c199-4453-b418-0e8f41e07660 67c5700f-a5ee-47c2-98d6-7f32bc7a5131 4e9a2741-bea2-48df-bb6a-8ffe8f445ed1" id="4c456b3e-d1c9-4cd0-ae4b-045a21340288" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="8961937d-3b73-487a-9bf2-1d982082c021" connectedTo="86837f57-529b-4d93-9e60-bf79ed82e113 f947aeb7-1503-4665-be37-da2be8a53ade" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_45b2_ret" id="ae1ac7d6-9e80-451b-9985-68e03dd71e73">
        <geometry xsi:type="esdl:Point" lat="52.03269420052414" lon="4.309485832539113" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="854aced4-c758-41ff-901e-7aa792e21ef8 e746e3c3-052f-4ab8-a29f-9b55c5543037" id="69ae33a3-c00d-40d5-926e-101d62e1bf07" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="4ed408bd-1977-4877-ab15-76d87ccd82c3" connectedTo="9f9a0f19-a92f-4454-a0fa-b77d8fd23c0b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_7975_ret" id="a9bb124f-48f4-4c4d-a179-544f9a0a7cc0">
        <geometry xsi:type="esdl:Point" lat="52.034476089590484" lon="4.30831145262905" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="71f6c981-0890-48b6-b705-a2d362a406c7 49c8cc59-e36c-4124-bc81-8f04e8fae0bb 55b8618f-111f-468c-8a66-8e76708ccb42" id="3aa1e57b-bc38-4266-859d-12b7601b09f4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="1ba07bbd-526f-4889-88b3-a0318998e4c6" connectedTo="965cf137-cd1c-4853-a305-ac10699b6319 9f0962d5-9dd9-4c43-aed0-75858a8123db" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_26ed_ret" id="1b4a81af-93fa-416d-aa1e-fb7f1c6342a5">
        <geometry xsi:type="esdl:Point" lat="52.03632389953942" lon="4.30514161372713" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="46571e5a-5b14-4b89-b835-0f4cbb41a5ae 5cdfadba-1a7d-40f9-8bf5-53d24c3758f7" id="8886e16b-996d-4d5b-9ca5-e29a952b461b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="b8f1e41a-cf6e-4f28-b472-4ac55e0368d4" connectedTo="7b7cab95-a956-4b1d-a976-1d698c0aea2d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_75c2_ret" id="9dd8a853-19a5-4a55-b170-82d33ae38d95">
        <geometry xsi:type="esdl:Point" lat="52.037300568231316" lon="4.304758448179862" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="70f5d3ce-4561-408b-bc1b-166e5cbc9b27 6aac1028-d0bd-44f2-b6da-f1f25b3f3660" id="ddc43005-31bf-418f-aa1e-71cdff9333f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="a77ac0ae-8354-4ff1-a959-9341fb7968dd" connectedTo="d8070dfb-9865-4995-b6a8-754dea752851" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_50da_ret" id="39c75c57-dfa4-40ee-995a-d198842a76fb">
        <geometry xsi:type="esdl:Point" lat="52.037326964386295" lon="4.297226887889662" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="71ef5381-44a4-4202-a886-469a25243fed 081ba6d6-328f-4bb5-a902-59bba57f67c3 661f8eae-94da-4d94-b5d8-f755b6ddefdb" id="d98ff88b-bd53-49fd-bc14-f1b63143025e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="35f23a87-6c03-40f4-b304-72f5fb15ede4" connectedTo="042b91ff-9261-48a7-a282-5f0729ff05b7 c28bc980-554f-4015-8a62-17758724473e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a9fd_ret" id="4ff674a5-248d-4290-9769-7eab0c1cda55">
        <geometry xsi:type="esdl:Point" lat="52.03916950215215" lon="4.304088328009291" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="21f190ff-1dc5-461f-a318-51751e0e9aad db9d952f-564d-4eb1-bdee-69fb4016c8de 594c6107-0f60-47a1-bca5-0e0263529938" id="727b3d6e-3bcb-49ca-896e-f97fd2ebf0bc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="8bdc69f3-4303-4559-99b1-7156c78f0cd9" connectedTo="73f1847a-7603-4002-b4b9-d599289856dc d4a43e90-7106-4613-9a90-45a02df974d9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a7f5_ret" id="02f9d442-9700-4c67-96b1-6027e11d1725">
        <geometry xsi:type="esdl:Point" lat="52.04298433842342" lon="4.317832742116763" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="beadae75-573c-494e-bc90-c8c4a373f918 34b93f89-52de-43ec-8ac0-ec752af4fe02 c7ae6629-a8da-4508-be88-26992fb56543" id="e6aa2067-fbd6-4a16-8ca2-0151b8d389ee" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="77fbb03a-8497-4aa5-b458-be03c126ca8a" connectedTo="883f43eb-dc02-4d6c-839f-2bf302cf6efe 64654e0a-ad86-491f-b5a4-69e069d340e0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1b08_ret" id="7b917718-89f7-413d-8253-ed0a6e7290b2">
        <geometry xsi:type="esdl:Point" lat="52.040041443604586" lon="4.321031827826063" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="ee57f2f9-2091-474a-9b81-35fa33a7ec44 af547ebe-0668-4fd7-9130-108069f59e83 11aa30ed-b270-4ed2-a24e-495bb069d71e" id="10ca3dd3-92cf-468a-bdef-fcd982ce8859" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="7ab22d2a-a5ce-43d4-a8b1-e7d93d543c89" connectedTo="f6cf6d52-bdb8-43d2-a4f8-a44c1432a473 9bfc3f39-1745-45f5-a649-44cf83c54098" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_ae77_ret" id="c21b5759-1198-4763-abf0-1cdec468c9be">
        <geometry xsi:type="esdl:Point" lat="52.041110409870306" lon="4.322794596185427" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="37354449-62e8-4fee-bdcc-a01212b3c35c 540899e1-91f7-4c41-bc8e-de46647fafbc d8ee80c7-3262-4aea-bd4f-2fc526e26fbc" id="c693a131-31b6-4b39-8999-bd57adb76b7b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="5c25fefc-e271-4d05-8cff-502ef0abc52e" connectedTo="011f96ad-982d-4dca-8df6-b2300d488582 729da237-9bbb-40ed-b85c-cb89d7e5aece" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a659_ret" id="8ba72fe6-c13e-48cf-96c3-4dedbd3333ec">
        <geometry xsi:type="esdl:Point" lat="52.042449880893734" lon="4.326671717170104" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="0d555c98-67b1-45df-b7d1-3f16d94acf4c e4a2dda5-35f7-432d-9c7f-1e63203c1e11 4714cac6-209a-4ab0-a0ca-37bf76468f55" id="38adcf2e-fd5e-44d7-9889-e41635df0822" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="7ba5ab0e-cd6f-461b-a949-9ff998a32223" connectedTo="a941fc70-51d0-4827-b471-e5155303ef3f c5d770bd-55e6-47d8-b851-ad799add08d6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_d25d_ret" id="3246cbb6-020b-48af-a123-d92f540ae138">
        <geometry xsi:type="esdl:Point" lat="52.043710134712654" lon="4.328756837645373" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="e8b4ea44-530f-42df-b2b6-219ab1d8966c fd3b2cfd-cc86-42b1-8ae2-986bcb59fb05 e4eb47c8-4f12-4b82-97fa-33a2f4d0af5d" id="0328e4a0-699e-4011-8090-cec6925ac527" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="5ee06323-235b-487e-b10c-24e037f15bcb" connectedTo="802d10d1-16dc-4027-9fb4-31051c51bca8 ec7a8701-76d6-4834-adc4-dde7340fd71e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_a80f_ret" id="a0a9d4fe-09b8-4121-9c7a-1227885b9c6f">
        <geometry xsi:type="esdl:Point" lat="52.04565653020963" lon="4.331605645414471" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="50bc5dc9-9a62-42fd-9b93-4a1709592763" id="2bef5f34-48ca-4f50-b017-37d960eaa133" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="e4caae09-f582-4c21-9d8e-b553849c7b8c" connectedTo="c1dab76b-9766-43ac-ba0f-989a65fb84df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_4e62_ret" id="690293bd-a170-4526-b327-154af9cbad5d">
        <geometry xsi:type="esdl:Point" lat="52.046672580962834" lon="4.329945591703785" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="9dbe3822-4e81-45ed-ab86-f91121a37326 c24e71f0-7cd2-48e8-bbc7-19d9dfc2ea0a 8f7969c1-f2f5-4816-8990-93b23c48a47e" id="f108633c-9c56-48dd-9ac8-de2b8ce206cf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="c5ef7be7-3d9c-4e71-b574-921fe2b63dbf" connectedTo="c35f5bdb-d000-480a-828a-4dbb1cd909aa f7297b6d-23d6-43a7-8afa-f5e2f1bbfdfb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_31ec_ret" id="607f0e8b-ead0-49c6-969e-773cd02bd7af">
        <geometry xsi:type="esdl:Point" lat="52.0501296195939" lon="4.333667480856885" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="ab993898-58b6-4c27-bafc-2027e335082a c14daf6d-97aa-406f-8b79-c0792fd02850 b4e684d3-fee7-4771-ba3f-74f18cf24b60" id="5081d9f4-d1f6-444f-9222-104e4b0fe984" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="9878324e-69e9-46c6-8d3a-b5b6fe013323" connectedTo="18cefdb6-963b-483d-a43b-8e8245954714 4db670e9-6af7-46aa-9b51-32ec5ec041d1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1085_ret" id="724b425c-8306-48b8-9534-2e6a5700bc90">
        <geometry xsi:type="esdl:Point" lat="52.0508684939238" lon="4.33615860562574" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="767cda8d-cfe3-4915-9f45-c4e2b0841eeb" connectedTo="970c7c00-1b2f-498b-bd64-76de2c35909b 8aa495f9-bfcd-4d2f-9d76-5294443b96f8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="a6446b1d-173a-4611-85ff-ee77b2695f2a 10986dc7-33f8-4b27-ac49-c5944176c585 de415c15-2b8d-45fd-a173-bfca996554f9" id="7bd8ead4-d8f9-40d4-a32a-5b33300cd3b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_cf81_ret" id="3174d55e-5ba0-4d79-b450-123364c26d2a">
        <geometry xsi:type="esdl:Point" lat="52.05287394837437" lon="4.334007561882603" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="0ab77890-2484-4e2a-99ed-7d9d5e14d619" connectedTo="8c3b0d43-9139-4507-a5a9-7b5333a8b551" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="a0825eaa-aa90-4df6-b0e1-115903aaae38 f08f0ea0-e8f2-44b0-aee5-553ea8af18df" id="ca8dd588-f4c7-45c8-bae8-eba8c790c466" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_6d6e_ret" id="d9dd69be-1915-4d48-a3f3-5a25f9333b36">
        <geometry xsi:type="esdl:Point" lat="52.051580968313644" lon="4.335366620519717" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="16b84d9d-e197-4753-b52d-7789c15ef891 321220e8-746a-48d1-9a1c-883b61cc794f" id="065518f2-9952-4951-8361-9899fdcad833" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="fae4d4ab-51bf-4b3e-94a1-1552327ab6f2" connectedTo="f267bc10-aaa8-4a5a-94ab-346019df8341" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_824f_ret" id="6cf9a7bf-8ea1-4b55-9748-3adaa14e9283">
        <geometry xsi:type="esdl:Point" lat="52.03594157365818" lon="4.303568628219171" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="b18f8317-80ac-4b2a-b38c-835316e2c407 7257e79c-77d5-44ec-826a-61f7379d6f2c" id="22a8375d-0126-4995-8410-6b0ab83936c2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="0815f851-4817-4a36-93e3-285bdc11aea1" connectedTo="c0d1e5a8-071f-4e8a-aa71-7f7d9e6046b3 41d926b6-c155-4212-8b2d-ea38cfae2458" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_c553_ret" id="f5f20d62-7ff1-447c-b9f1-4824cb715f5b">
        <geometry xsi:type="esdl:Point" lat="52.030492909001566" lon="4.299629457903739" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="2b301f59-5ca4-48c5-90c8-66c3396b7423 fd313f8f-3702-4058-ba69-9ccbe7b1bdbb" id="4012b484-d65f-477f-ad84-e6df35ab3531" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="4c6bab42-d9aa-4ff5-9c22-5f604b34fb22" connectedTo="111e7606-f406-40b3-8a27-1c79abcc36e2 df17614f-df2b-4402-83a7-bc5433c316cc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_4bae_ret" id="d4990471-fb71-4fe4-a772-1e739466af58">
        <geometry xsi:type="esdl:Point" lat="52.029667795898824" lon="4.3136170795418565" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="645997b9-ff0f-4e08-949d-e499d42286de 6099e2c6-e819-4bd5-b774-3adfb9913673" id="ccbdf824-b95a-4116-8e48-55eb708f44c9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="a0d9f763-3754-4439-9509-003807bbdd65" connectedTo="32a5133e-bb0a-4bc2-9d41-907b310ebf6e b8f6e3cf-e0cb-412e-86fe-4018183ffb1a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_1817_ret" id="d65baff8-7eb9-405b-bc5c-3ac0d9ad7acb">
        <geometry xsi:type="esdl:Point" lat="52.02758184207936" lon="4.309672459453438" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="8bad9ab5-164f-4cf1-96fd-7f7316aaafd6 8ea470e0-13bf-4144-a699-943cd3d6bd3c" id="cfff919b-0f98-40fb-a5c9-d733fff20bb4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="fd235b70-a17f-48da-b6b4-90470317eabf" connectedTo="6ef65168-b48d-4edd-9d5c-d24ba52a9da1 3aa80042-a30a-409d-94e6-32a6d3cb2232" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_e143_ret" id="929268ab-162e-405e-b5b4-6adf86c3da00">
        <geometry xsi:type="esdl:Point" lat="52.03170084712326" lon="4.323172552258066" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="aee25d4b-808b-41c5-b6a6-46cfd0534f61 0355616b-1d86-4553-9f23-3b87ed3966ca" id="adb7329e-ec83-40e4-967a-c5c33ffe7550" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="4a3c0adb-12d0-4731-9dc5-021bf4f1faa4" connectedTo="757cd47f-b98d-4340-b5a0-af50c62ccb66 ebfa9843-4015-4256-a33c-13036255719d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_458a_ret" id="13ed58f3-ddd1-4907-849b-f59506974c23">
        <geometry xsi:type="esdl:Point" lat="52.048813097442256" lon="4.336775182356948" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="f5813dd3-9d4c-4aa8-9e00-a9ec870007b3" connectedTo="35d57051-f40b-4aa4-a9e2-ac4eb7c347f7 c1a2f0f9-6cc0-4483-9175-62f12eda11b4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="bbb87a16-7099-448a-843d-e7376bcee47e 0b2c48de-6dce-4a48-9f33-15445eb38689" id="0e642c28-26ce-496a-9b16-a83a037accf1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_0969_ret" id="2864e5e2-3f57-4a5c-b02d-2250e15dc5df">
        <geometry xsi:type="esdl:Point" lat="52.05221100992171" lon="4.332074580336848" CRS="WGS84"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="3472767c-618c-4c0f-b8a2-2891f8bb2440 78821a62-b0a9-47bd-b271-d58622c8d8c2" id="3f5063b1-c676-4b81-bdaa-bfdd919a10ca" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="ba9aa151-e244-4a01-bbb3-1c4ead3f8e8a" connectedTo="f98c1cb7-877d-4640-bbc4-b67d43106659 4bb8d3f2-2399-44fc-91e4-d3d10dc7efd6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Joint" name="Joint_4f81_ret" id="c5a65a12-75cf-4caa-ab4e-ce1477a68be3">
        <geometry xsi:type="esdl:Point" lat="52.054817000711104" lon="4.33219957498349" CRS="WGS84"/>
        <port xsi:type="esdl:OutPort" name="ret_port" id="ca7ee8ee-caa3-4ae4-ae4b-c1ab3604022e" connectedTo="442c90bc-79fa-41e4-85bb-abf96e05feee 6fa9faf7-af3b-4987-90a8-847da7591e45" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:InPort" name="ret_port" connectedTo="e703e2e0-2a3c-48fb-89ef-19d0fd0109ed d4082c49-e51f-452e-b22f-8a818d534c64" id="621ab362-a3a1-464c-ae55-ecda1ee72c32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe1" diameter="DN400" length="355.4" name="Pipe1_ret" id="Pipe1_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.041739007433215" lon="4.312164208605956" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04374837657511" lon="4.310576937301564" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0444861582057" lon="4.309578632886089" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="aa141fc0-ce28-4d14-99be-bed98630e2e5" id="c07eaec4-b994-4ff1-ba68-fa2452a8e424" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8fb322ac-0b28-4cf2-b01e-8728a9646a65" connectedTo="6df6baf1-2c39-49c3-ad57-1911dd437fdf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe2" diameter="DN400" length="495.8" name="Pipe2_ret" id="Pipe2_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.038888193298845" lon="4.316066207877817" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03956499246764" lon="4.315478200334688" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03918226585622" lon="4.3143183124088695" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.041739007433215" lon="4.312164208605956" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="401eae5a-7174-4c37-b64d-33c87782ed60" id="5967e711-a8e1-4fe6-8ef3-fbe16296ddaf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="684abfb7-4218-4a1e-b6da-a822ff112397" connectedTo="675d388d-ece7-48c3-bbc5-e4b73b9510f3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe3" diameter="DN400" length="170.5" name="Pipe3_ret" id="Pipe3_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03775978317298" lon="4.317623756910878" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03863786459962" lon="4.3168110896795175" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.038888193298845" lon="4.316066207877817" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="51a9b7c1-021f-41bc-8690-1870c52fc0d0" id="a351aea6-ee34-4b0f-b0ba-388def8927a4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d6a89042-7b6f-4d2a-9408-d36ea437ec1e" connectedTo="2ec8715c-c5b9-4782-918c-f1b35d426904" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe4" diameter="DN400" length="26.1" name="Pipe4_ret" id="Pipe4_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0444861582057" lon="4.309578632886089" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04469236954422" lon="4.309396843514948" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="d49a9b33-62b9-43ee-aa3c-19e087899111" id="e49b7e3c-9fec-4b5a-ac37-9b7266342b35" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d05ca38e-424c-4325-8e75-69379f90bf24" connectedTo="85c8013d-9a9b-4efc-87ba-cfd3d999adb6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe5" diameter="DN400" length="34.3" name="Pipe5_ret" id="Pipe5_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0444861582057" lon="4.309578632886089" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04478805128214" lon="4.309681436053361" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="d49a9b33-62b9-43ee-aa3c-19e087899111" id="a6b22da3-d60a-4617-8df7-5ed7fb04ea05" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="f7d48b4e-2348-4b24-9af9-98598f270be2" connectedTo="3aacb2a9-615b-4047-8b01-04f5f1f32379" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe6" diameter="DN400" length="826.7" name="Pipe6_ret" id="Pipe6_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03916950215215" lon="4.304088328009291" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04073241340808" lon="4.302988023292634" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0430286442908" lon="4.305183575497449" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0444861582057" lon="4.309578632886089" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="8bdc69f3-4303-4559-99b1-7156c78f0cd9" id="73f1847a-7603-4002-b4b9-d599289856dc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d01a9f0c-09dc-44db-8937-2f651ce3fb85" connectedTo="6df6baf1-2c39-49c3-ad57-1911dd437fdf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe7" diameter="DN400" length="212.9" name="Pipe7_ret" id="Pipe7_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.037300568231316" lon="4.304758448179862" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03916950215215" lon="4.304088328009291" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="a77ac0ae-8354-4ff1-a959-9341fb7968dd" id="d8070dfb-9865-4995-b6a8-754dea752851" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="21f190ff-1dc5-461f-a318-51751e0e9aad" connectedTo="727b3d6e-3bcb-49ca-896e-f97fd2ebf0bc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe8" diameter="DN400" length="111.8" name="Pipe8_ret" id="Pipe8_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03632389953942" lon="4.30514161372713" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037300568231316" lon="4.304758448179862" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="b8f1e41a-cf6e-4f28-b472-4ac55e0368d4" id="7b7cab95-a956-4b1d-a976-1d698c0aea2d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="70f5d3ce-4561-408b-bc1b-166e5cbc9b27" connectedTo="ddc43005-31bf-418f-aa1e-71cdff9333f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe9" diameter="DN400" length="387.2" name="Pipe9_ret" id="Pipe9_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.034476089590484" lon="4.30831145262905" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03479613490827" lon="4.308054989781725" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034426566399354" lon="4.306423017924168" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03632389953942" lon="4.30514161372713" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="1ba07bbd-526f-4889-88b3-a0318998e4c6" id="965cf137-cd1c-4853-a305-ac10699b6319" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="46571e5a-5b14-4b89-b835-0f4cbb41a5ae" connectedTo="8886e16b-996d-4d5b-9ca5-e29a952b461b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe10" diameter="DN400" length="214.0" name="Pipe10_ret" id="Pipe10_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03269420052414" lon="4.309485832539113" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034476089590484" lon="4.30831145262905" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4ed408bd-1977-4877-ab15-76d87ccd82c3" id="9f9a0f19-a92f-4454-a0fa-b77d8fd23c0b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="71f6c981-0890-48b6-b705-a2d362a406c7" connectedTo="3aa1e57b-bc38-4266-859d-12b7601b09f4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe11" diameter="DN400" length="99.1" name="Pipe11_ret" id="Pipe11_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03188902735578" lon="4.3101054529409275" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03269420052414" lon="4.309485832539113" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="8961937d-3b73-487a-9bf2-1d982082c021" id="86837f57-529b-4d93-9e60-bf79ed82e113" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="854aced4-c758-41ff-901e-7aa792e21ef8" connectedTo="69ae33a3-c00d-40d5-926e-101d62e1bf07" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe12" diameter="DN400" length="74.3" name="Pipe12_ret" id="Pipe12_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0313082377085" lon="4.310639968040655" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03188902735578" lon="4.3101054529409275" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="dc94e82b-2d27-4593-bf45-313f72814379" id="8b26c9ef-4a71-438c-8de9-d5a6b012d71a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="dc4fee0c-c199-4453-b418-0e8f41e07660" connectedTo="4c456b3e-d1c9-4cd0-ae4b-045a21340288" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe13" diameter="DN400" length="286.4" name="Pipe13_ret" id="Pipe13_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02905100631209" lon="4.31264939009626" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0313082377085" lon="4.310639968040655" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4a07801a-c279-4e37-8c21-ecbeb8248b20" id="e1ae80f8-f763-4f93-bf88-9977684161be" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="f240ed4d-a9c6-4b70-9573-be6bb9330487" connectedTo="7995eac2-82c7-4456-8f8e-a4fa46c783a2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe15" diameter="DN200" length="142.5" name="Pipe15_ret" id="Pipe15_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02810055900395" lon="4.311809287259864" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02825510194936" lon="4.312432090999568" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02841351068946" lon="4.312668669036986" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02870392525555" lon="4.3127769517329035" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02905100631209" lon="4.31264939009626" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="e526b492-b974-44a8-835b-53e8ffdb0e1b" id="9ff5ba8a-a36a-4a52-9f06-9ffc9bec6631" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="f48d27c5-cb99-45ed-bdf6-ac8594944ba9" connectedTo="cb5b59bf-b5a2-4948-ba3b-cbba2aea383b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe16" diameter="DN200" length="250.5" name="Pipe16_ret" id="Pipe16_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02614679831976" lon="4.313626398166783" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02810055900395" lon="4.311809287259864" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="cd825ba8-0050-4b8b-96e8-511f6b1e025c" id="8914c40c-a08d-4a85-a893-4a796d062bfa" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="cb91c93d-5a0b-4ac4-b4f4-4afd183b2a38" connectedTo="9ba3845f-c482-40bf-ac11-a20b37ca25df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe17" diameter="DN200" length="164.5" name="Pipe17_ret" id="Pipe17_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.026107194180035" lon="4.311866729934698" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.025693779376645" lon="4.312101308620519" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02614679831976" lon="4.313626398166783" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="940d3006-7e6e-4a7c-814f-d130d992564c" id="c6164983-07e3-400b-b0db-936daaface70" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="3dda78e8-4980-4ac2-b97b-2a9b02909037" connectedTo="7ac9b957-d251-4415-afb0-42c89590fb47" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe19" diameter="DN200" length="70.8" name="Pipe19_ret" id="Pipe19_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03483245888195" lon="4.3091709053332385" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034476089590484" lon="4.30831145262905" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="bbe6e10e-2805-44ce-9cba-279dac4e168b" id="c8cd7c63-3a3d-4760-9b18-333d9a5f0e64" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="49c8cc59-e36c-4124-bc81-8f04e8fae0bb" connectedTo="3aa1e57b-bc38-4266-859d-12b7601b09f4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe21" diameter="DN200" length="571.4" name="Pipe21_ret" id="Pipe21_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.037326964386295" lon="4.297226887889662" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037925913177205" lon="4.2997393044629915" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037450786368396" lon="4.300209891602603" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03702844718787" lon="4.301667690342178" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03702844718787" lon="4.302955150669327" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037300568231316" lon="4.304758448179862" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="35f23a87-6c03-40f4-b304-72f5fb15ede4" id="042b91ff-9261-48a7-a282-5f0729ff05b7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="6aac1028-d0bd-44f2-b6da-f1f25b3f3660" connectedTo="ddc43005-31bf-418f-aa1e-71cdff9333f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe22" diameter="DN200" length="88.6" name="Pipe22_ret" id="Pipe22_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03697061497874" lon="4.296067056324813" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037326964386295" lon="4.297226887889662" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="327389ca-879f-499b-9a03-97a00ea571f4" id="997ff413-e4c0-4cad-a81e-b1474abdb91f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="71ef5381-44a4-4202-a886-469a25243fed" connectedTo="d98ff88b-bd53-49fd-bc14-f1b63143025e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe24" diameter="DN200" length="69.4" name="Pipe24_ret" id="Pipe24_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03329817966756" lon="4.320743697734791" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03296207243976" lon="4.319888316571063" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="60efe325-8f39-4540-b7b2-cfa8d605a9db" id="cb5c01ac-0415-418b-a92f-e4d7b001cd10" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="2a669cbf-4402-415f-9580-6f7bd49dafb3" connectedTo="425cb5ab-fe83-4313-82bc-6ea70546c77e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe25" diameter="DN300" length="410.3" name="Pipe25_ret" id="Pipe25_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.034823143777416" lon="4.318665895348221" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.035917465007" lon="4.317617962517913" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0365377897204" lon="4.318853741681011" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03775978317298" lon="4.317623756910878" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="6f78a5f5-edd6-4641-afce-1f57fdb304b3" id="7510d944-54c6-4d1b-98f4-2485da00c854" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e1c36f63-4988-4e3c-8519-708b6ae10e86" connectedTo="63920326-77bb-4ce1-aa31-bba537b8ec57" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe26" diameter="DN300" length="255.8" name="Pipe26_ret" id="Pipe26_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03296207243976" lon="4.319888316571063" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.033554872921606" lon="4.319273341811292" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03369346417251" lon="4.319702945557949" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034823143777416" lon="4.318665895348221" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="122ee1ba-078e-4d41-8525-426ff44fbef8" id="c5ce9a46-6831-4a59-b2fc-feb0284949ed" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="2dcc99cf-eeea-47fd-92d9-e35cbd77a878" connectedTo="28b38730-7167-476f-8962-7d09d9f9d417" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe27" diameter="DN150" length="34.6" name="Pipe27_ret" id="Pipe27_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03500765830515" lon="4.319074182809998" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034823143777416" lon="4.318665895348221" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="9076d363-fa38-4470-a566-1c692a87bae7" id="7a9cd346-22a0-467f-8eae-b72541b0bef4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="7a9f9f48-98cd-4bb1-a658-66538e9825ec" connectedTo="28b38730-7167-476f-8962-7d09d9f9d417" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe28" diameter="DN150" length="22.8" name="Pipe28_ret" id="Pipe28_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03781264571931" lon="4.317945786679015" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03775978317298" lon="4.317623756910878" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="3b1b483c-7d9e-4ffd-aa58-e16edef4cb44" id="4f760ceb-53f0-4867-bf62-bbecaf12c768" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="15ed4e8a-5d45-4272-a2a1-5a0eb66587c3" connectedTo="63920326-77bb-4ce1-aa31-bba537b8ec57" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe29" diameter="DN400" length="23.0" name="Pipe29_ret" id="Pipe29_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03775978317298" lon="4.317623756910878" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03765425111864" lon="4.31733374930543" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="51a9b7c1-021f-41bc-8690-1870c52fc0d0" id="cbe50ae5-e435-4273-863b-4cdb724d77a7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="2f3babb2-f6e5-469e-a3b5-374d087ef970" connectedTo="d0832916-11c6-4f8f-a601-0fb90b1f5764" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe30" diameter="DN300" length="39.5" name="Pipe30_ret" id="Pipe30_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.038734953131" lon="4.3155453867867015" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.038888193298845" lon="4.316066207877817" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="6db24e9f-83c6-40be-a58a-faaa0e155b00" id="8d924dd2-37dc-4d06-9bb4-a028abc8d29c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8cca0c12-c7b2-45ed-9d3d-8353a5c9bf52" connectedTo="2ec8715c-c5b9-4782-918c-f1b35d426904" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe31" diameter="DN150" length="21.6" name="Pipe31_ret" id="Pipe31_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04181687277464" lon="4.312454120356929" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.041739007433215" lon="4.312164208605956" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="0ab74980-9ef0-433d-bf30-ec3cf137ad7c" id="3a78bae6-bc9c-45c1-bd5d-6c722675944f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="1ef2a05a-afb5-445a-8563-9a14af9b9f13" connectedTo="675d388d-ece7-48c3-bbc5-e4b73b9510f3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe32" diameter="DN300" length="77.5" name="Pipe32_ret" id="Pipe32_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03220581852917" lon="4.3111150096007185" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03188902735578" lon="4.3101054529409275" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4b7d7685-af5d-4ccd-9f6c-d66a7a447a0e" id="8fa483ed-2b40-4410-bcd6-301b097ec23e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="67c5700f-a5ee-47c2-98d6-7f32bc7a5131" connectedTo="4c456b3e-d1c9-4cd0-ae4b-045a21340288" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe33" diameter="DN300" length="87.7" name="Pipe33_ret" id="Pipe33_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03107063976937" lon="4.309416089191067" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0313082377085" lon="4.310639968040655" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="1417f744-930d-478b-bafd-d9d358b83e8e" id="4e5ce568-bbc1-484f-be68-e5a46a61b93b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="ec61db12-3e07-4b5b-834a-4ccdb5b97837" connectedTo="7995eac2-82c7-4456-8f8e-a4fa46c783a2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe35" diameter="DN150" length="124.2" name="Pipe35_ret" id="Pipe35_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02523589423255" lon="4.314674609495366" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02614679831976" lon="4.313626398166783" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="b3941ae3-a8c5-4a33-8369-68dfa6fa3fd1" id="d7d4264b-01c0-43ba-8864-9b01f445a9aa" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="14826ed9-07be-48aa-81a3-01fdc1bdf5da" connectedTo="7ac9b957-d251-4415-afb0-42c89590fb47" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe36" diameter="DN150" length="42.8" name="Pipe36_ret" id="Pipe36_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03697061497874" lon="4.297461805012557" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037326964386295" lon="4.297226887889662" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="824b1653-94c7-408a-a1a6-9aab266bfa89" id="01ae5f1b-6a10-430c-aefb-6b981fc85911" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="081ba6d6-328f-4bb5-a902-59bba57f67c3" connectedTo="d98ff88b-bd53-49fd-bc14-f1b63143025e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe37" diameter="DN150" length="21.6" name="Pipe37_ret" id="Pipe37_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03913506390387" lon="4.303777085926437" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03916950215215" lon="4.304088328009291" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="6194bccd-c26a-469b-88fd-6d9d5453065a" id="0efab627-79b2-432a-b65e-2ea0b7de34f0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="db9d952f-564d-4eb1-bdee-69fb4016c8de" connectedTo="727b3d6e-3bcb-49ca-896e-f97fd2ebf0bc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe38" diameter="DN400" length="785.2" name="Pipe38_ret" id="Pipe38_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04298433842342" lon="4.317832742116763" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0448817848503" lon="4.316110963527158" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04497415565644" lon="4.315789366707846" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.044367144007055" lon="4.314135358682189" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.044789413853294" lon="4.313793266435686" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04539641976745" lon="4.3124646508230375" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0444861582057" lon="4.309578632886089" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="77fbb03a-8497-4aa5-b458-be03c126ca8a" id="883f43eb-dc02-4d6c-839f-2bf302cf6efe" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="04e44cf5-08e8-45e2-a7ce-0c10dcfbe9ee" connectedTo="6df6baf1-2c39-49c3-ad57-1911dd437fdf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe39" diameter="DN400" length="397.3" name="Pipe39_ret" id="Pipe39_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.040041443604586" lon="4.321031827826063" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04147712766037" lon="4.3190191493082635" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04298433842342" lon="4.317832742116763" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="7ab22d2a-a5ce-43d4-a8b1-e7d93d543c89" id="f6cf6d52-bdb8-43d2-a4f8-a44c1432a473" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="beadae75-573c-494e-bc90-c8c4a373f918" connectedTo="e6aa2067-fbd6-4a16-8ca2-0151b8d389ee" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe40" diameter="DN400" length="169.1" name="Pipe40_ret" id="Pipe40_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.041110409870306" lon="4.322794596185427" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.040041443604586" lon="4.321031827826063" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="5c25fefc-e271-4d05-8cff-502ef0abc52e" id="011f96ad-982d-4dca-8df6-b2300d488582" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="ee57f2f9-2091-474a-9b81-35fa33a7ec44" connectedTo="10ca3dd3-92cf-468a-bdef-fcd982ce8859" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe41" diameter="DN400" length="381.2" name="Pipe41_ret" id="Pipe41_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.042449880893734" lon="4.326671717170104" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.040553338732096" lon="4.323908711327824" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.041110409870306" lon="4.322794596185427" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="7ba5ab0e-cd6f-461b-a949-9ff998a32223" id="a941fc70-51d0-4827-b471-e5155303ef3f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="37354449-62e8-4fee-bdcc-a01212b3c35c" connectedTo="c693a131-31b6-4b39-8999-bd57adb76b7b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe42" diameter="DN400" length="200.5" name="Pipe42_ret" id="Pipe42_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.043710134712654" lon="4.328756837645373" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.043153095975015" lon="4.327650126138734" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.042449880893734" lon="4.326671717170104" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="5ee06323-235b-487e-b10c-24e037f15bcb" id="802d10d1-16dc-4027-9fb4-31051c51bca8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="0d555c98-67b1-45df-b7d1-3f16d94acf4c" connectedTo="38adcf2e-fd5e-44d7-9889-e41635df0822" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe43" diameter="DN400" length="290.9" name="Pipe43_ret" id="Pipe43_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04565653020963" lon="4.331605645414471" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.043710134712654" lon="4.328756837645373" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="e4caae09-f582-4c21-9d8e-b553849c7b8c" id="c1dab76b-9766-43ac-ba0f-989a65fb84df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e8b4ea44-530f-42df-b2b6-219ab1d8966c" connectedTo="0328e4a0-699e-4011-8090-cec6925ac527" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe45" diameter="DN300" length="160.3" name="Pipe45_ret" id="Pipe45_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.046672580962834" lon="4.329945591703785" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04565653020963" lon="4.331605645414471" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="c5ef7be7-3d9c-4e71-b574-921fe2b63dbf" id="c35f5bdb-d000-480a-828a-4dbb1cd909aa" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="50bc5dc9-9a62-42fd-9b93-4a1709592763" connectedTo="2bef5f34-48ca-4f50-b017-37d960eaa133" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe46" diameter="DN300" length="537.9" name="Pipe46_ret" id="Pipe46_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0501296195939" lon="4.333667480856885" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0473399813297" lon="4.3290569982271725" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.046672580962834" lon="4.329945591703785" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="9878324e-69e9-46c6-8d3a-b5b6fe013323" id="18cefdb6-963b-483d-a43b-8e8245954714" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="9dbe3822-4e81-45ed-ab86-f91121a37326" connectedTo="f108633c-9c56-48dd-9ac8-de2b8ce206cf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe47" diameter="DN300" length="198.7" name="Pipe47_ret" id="Pipe47_ret" innerDiameter="0.3127" outerDiameter="0.5">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.051580968313644" lon="4.335366620519717" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0501296195939" lon="4.333667480856885" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="fae4d4ab-51bf-4b3e-94a1-1552327ab6f2" id="f267bc10-aaa8-4a5a-94ab-346019df8341" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="ab993898-58b6-4c27-bafc-2027e335082a" connectedTo="5081d9f4-d1f6-444f-9222-104e4b0fe984" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe48" diameter="DN200" length="96.0" name="Pipe48_ret" id="Pipe48_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0508684939238" lon="4.33615860562574" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.051580968313644" lon="4.335366620519717" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="767cda8d-cfe3-4915-9f45-c4e2b0841eeb" id="970c7c00-1b2f-498b-bd64-76de2c35909b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="16b84d9d-e197-4753-b52d-7789c15ef891" connectedTo="065518f2-9952-4951-8361-9899fdcad833" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe50" diameter="DN200" length="171.3" name="Pipe50_ret" id="Pipe50_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05287394837437" lon="4.334007561882603" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.051580968313644" lon="4.335366620519717" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="0ab77890-2484-4e2a-99ed-7d9d5e14d619" id="8c3b0d43-9139-4507-a5a9-7b5333a8b551" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="321220e8-746a-48d1-9a1c-883b61cc794f" connectedTo="065518f2-9952-4951-8361-9899fdcad833" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe53" diameter="DN200" length="60.6" name="Pipe53_ret" id="Pipe53_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04968161625739" lon="4.3341704954656874" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0501296195939" lon="4.333667480856885" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="c735b1aa-e309-4945-8886-195af354f0ac" id="c617234b-aa4d-420e-a4eb-615713be15d3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="c14daf6d-97aa-406f-8b79-c0792fd02850" connectedTo="5081d9f4-d1f6-444f-9222-104e4b0fe984" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe54" diameter="DN150" length="31.1" name="Pipe54_ret" id="Pipe54_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.046514841849365" lon="4.3295696316225" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.046672580962834" lon="4.329945591703785" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="483a6cd3-01b3-405f-924b-995cad603382" id="3741698a-602c-4d08-9a22-f5b42d8eb53a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="c24e71f0-7cd2-48e8-bbc7-19d9dfc2ea0a" connectedTo="f108633c-9c56-48dd-9ac8-de2b8ce206cf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe55" diameter="DN150" length="116.3" name="Pipe55_ret" id="Pipe55_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04438374025349" lon="4.327814673115141" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04426673417878" lon="4.327637305524668" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.043710134712654" lon="4.328756837645373" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="e4f770db-2bcb-444c-9b76-7e894a06ab9b" id="359cd445-80bc-484c-b4db-e2c988cc6176" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="fd3b2cfd-cc86-42b1-8ae2-986bcb59fb05" connectedTo="0328e4a0-699e-4011-8090-cec6925ac527" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe56" diameter="DN150" length="87.2" name="Pipe56_ret" id="Pipe56_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0429915416113" lon="4.325750644445816" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.042449880893734" lon="4.326671717170104" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="8696c727-7e34-4217-b1c6-2b3638bbbb8f" id="dd39438a-9e0a-4d27-a533-fa940877ff36" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e4a2dda5-35f7-432d-9c7f-1e63203c1e11" connectedTo="38adcf2e-fd5e-44d7-9889-e41635df0822" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe57" diameter="DN150" length="91.5" name="Pipe57_ret" id="Pipe57_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04166528361519" lon="4.3237833171192" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.041110409870306" lon="4.322794596185427" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="b2a780bf-6ceb-4c2f-8242-42b3e8d6e1be" id="89fa68ff-861e-4c57-a53b-c44cf7536af7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="540899e1-91f7-4c41-bc8e-de46647fafbc" connectedTo="c693a131-31b6-4b39-8999-bd57adb76b7b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe58" diameter="DN150" length="45.3" name="Pipe58_ret" id="Pipe58_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03966592479637" lon="4.321288173828406" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.040041443604586" lon="4.321031827826063" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="0ecf3ec6-0c44-4231-a553-ca5173d09dfa" id="d229c40e-ebfe-499f-94b2-f6d215b9d794" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="af547ebe-0668-4fd7-9130-108069f59e83" connectedTo="10ca3dd3-92cf-468a-bdef-fcd982ce8859" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe59" diameter="DN150" length="48.9" name="Pipe59_ret" id="Pipe59_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.043222551463565" lon="4.318434261109859" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04298433842342" lon="4.317832742116763" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="3fc2139b-301d-4c12-8ec7-6a263075334c" id="dc7d95d4-a8b1-4500-a48e-1f83b876b49a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="34b93f89-52de-43ec-8ac0-ec752af4fe02" connectedTo="e6aa2067-fbd6-4a16-8ca2-0151b8d389ee" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe61" diameter="DN250" length="59.4" name="Pipe61_ret" id="Pipe61_ret" innerDiameter="0.263" outerDiameter="0.4">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.051278113262896" lon="4.336717627084579" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0508684939238" lon="4.33615860562574" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="a0c2f54d-b8af-44b3-9107-1ec0571d61ab" id="3240dcc5-5056-409d-8d83-c8e3989139b4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="a6446b1d-173a-4611-85ff-ee77b2695f2a" connectedTo="7bd8ead4-d8f9-40d4-a32a-5b33300cd3b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe63" diameter="DN400" length="76.2" name="Pipe63_ret" id="Pipe63_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03742897311974" lon="4.316646400253598" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03775978317298" lon="4.317623756910878" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="5c089df4-9cb1-4249-8e2b-b46994220f62" id="95416ff8-70ce-48d9-ae41-cdf234f4408f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="69832a5b-c670-4a1a-8435-76788055420e" connectedTo="63920326-77bb-4ce1-aa31-bba537b8ec57" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe65" diameter="DN400" length="38.3" name="Pipe65_ret" id="Pipe65_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03934376459173" lon="4.304571660628333" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03916950215215" lon="4.304088328009291" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="d3308201-f313-4047-8c84-e7d1f3ba6f41" id="f55dc1fe-acc4-42ec-9167-b959b5e3613b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="594c6107-0f60-47a1-bca5-0e0263529938" connectedTo="727b3d6e-3bcb-49ca-896e-f97fd2ebf0bc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe66" diameter="DN400" length="57.9" name="Pipe66_ret" id="Pipe66_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03916950215215" lon="4.304088328009291" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03966714074919" lon="4.304336616572189" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="8bdc69f3-4303-4559-99b1-7156c78f0cd9" id="d4a43e90-7106-4613-9a90-45a02df974d9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="2af09cda-994d-426f-8b34-9377f3427944" connectedTo="88741a34-59fa-455e-828d-a0144a9178bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe68" diameter="DN400" length="88.2" name="Pipe68_ret" id="Pipe68_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03766745069015" lon="4.296063872990465" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037326964386295" lon="4.297226887889662" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="9b499471-ef6a-496d-9d5a-33b3e80441fa" id="cc812c98-eca1-4ac7-aa76-2387306e2ba3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="661f8eae-94da-4d94-b5d8-f755b6ddefdb" connectedTo="d98ff88b-bd53-49fd-bc14-f1b63143025e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe67" diameter="DN400" length="55.3" name="Pipe67_ret" id="Pipe67_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.037326964386295" lon="4.297226887889662" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.037740048263686" lon="4.296777566943284" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="35f23a87-6c03-40f4-b304-72f5fb15ede4" id="c28bc980-554f-4015-8a62-17758724473e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d080f436-4bd8-4cc7-bc75-c38ce217ac88" connectedTo="1a6bbd6f-acb3-4ec3-93f0-690c1ed4cc71" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe20_a" diameter="DN200" length="115.61" name="Pipe20_a_ret" id="Pipe20_a_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03594157365818" lon="4.303568628219171" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03632389953942" lon="4.30514161372713" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="0815f851-4817-4a36-93e3-285bdc11aea1" id="c0d1e5a8-071f-4e8a-aa71-7f7d9e6046b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="5cdfadba-1a7d-40f9-8bf5-53d24c3758f7" connectedTo="8886e16b-996d-4d5b-9ca5-e29a952b461b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe20_b" diameter="DN200" length="31.29" name="Pipe20_b_ret" id="Pipe20_b_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03583555719202" lon="4.30314450255764" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03594157365818" lon="4.303568628219171" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="6cf7348f-8aac-4f1e-9e31-6ac8295f5706" id="2f4e69b7-96f9-4eb7-ac5d-97a6255dbb7c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="b18f8317-80ac-4b2a-b38c-835316e2c407" connectedTo="22a8375d-0126-4995-8410-6b0ab83936c2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe70" diameter="DN400" length="73.0" name="Pipe70_ret" id="Pipe70_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03559507023818" lon="4.3026609399724665" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03594157365818" lon="4.303568628219171" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="87c171ef-0902-49cd-a82c-c44d687aa5d4" id="876f3a14-19ac-44e6-a746-d509979e459b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="7257e79c-77d5-44ec-826a-61f7379d6f2c" connectedTo="22a8375d-0126-4995-8410-6b0ab83936c2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe69" diameter="DN400" length="38.9" name="Pipe69_ret" id="Pipe69_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03594157365818" lon="4.303568628219171" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03559672026082" lon="4.303658726980881" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="0815f851-4817-4a36-93e3-285bdc11aea1" id="41d926b6-c155-4212-8b2d-ea38cfae2458" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="1adbcc35-d2b4-4115-b635-ed074a43dc7f" connectedTo="ed240007-6411-4983-b2ae-10eeb9d4c735" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe71" diameter="DN400" length="43.1" name="Pipe71_ret" id="Pipe71_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.034476089590484" lon="4.30831145262905" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0343311349992" lon="4.3088957069395235" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="1ba07bbd-526f-4889-88b3-a0318998e4c6" id="9f0962d5-9dd9-4c43-aed0-75858a8123db" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="42ed67c8-1ad9-4f54-a04e-50ae8c419bb2" connectedTo="9124da2f-9c41-4217-9ba2-db523680605a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe72" diameter="DN400" length="74.9" name="Pipe72_ret" id="Pipe72_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03430143374143" lon="4.307254099196654" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034476089590484" lon="4.30831145262905" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="b00171ec-afff-420e-96f2-c3ddc27514b3" id="942ae075-61b0-4789-9cbe-b639210a4fc6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="55b8618f-111f-468c-8a66-8e76708ccb42" connectedTo="3aa1e57b-bc38-4266-859d-12b7601b09f4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe14_a" diameter="DN200" length="721.48" name="Pipe14_a_ret" id="Pipe14_a_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.030492909001566" lon="4.299629457903739" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.030862396357485" lon="4.302902990355719" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03269420052414" lon="4.309485832539113" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4c6bab42-d9aa-4ff5-9c22-5f604b34fb22" id="111e7606-f406-40b3-8a27-1c79abcc36e2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e746e3c3-052f-4ab8-a29f-9b55c5543037" connectedTo="69ae33a3-c00d-40d5-926e-101d62e1bf07" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe14_b" diameter="DN200" length="62.56" name="Pipe14_b_ret" id="Pipe14_b_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03039743875227" lon="4.2987279151631785" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.030492909001566" lon="4.299629457903739" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="35e0390a-790b-4f1f-8a9e-e3cfea3fe38b" id="c9db5954-0973-4a45-a040-008ca98e05d0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="2b301f59-5ca4-48c5-90c8-66c3396b7423" connectedTo="4012b484-d65f-477f-ad84-e6df35ab3531" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe74" diameter="DN400" length="64.7" name="Pipe74_ret" id="Pipe74_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.030222273581806" lon="4.298791699290853" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.030492909001566" lon="4.299629457903739" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="2b1d2659-1961-47b0-a271-e4f4167722db" id="8c592f69-5d36-4512-b5da-67a6c3174d00" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="fd313f8f-3702-4058-ba69-9ccbe7b1bdbb" connectedTo="4012b484-d65f-477f-ad84-e6df35ab3531" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe73" diameter="DN400" length="36.0" name="Pipe73_ret" id="Pipe73_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.030492909001566" lon="4.299629457903739" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03024207622905" lon="4.29929602121397" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4c6bab42-d9aa-4ff5-9c22-5f604b34fb22" id="df17614f-df2b-4402-83a7-bc5433c316cc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d6caa807-0d17-4fd0-a983-5c1036159a78" connectedTo="49295fb9-e5b6-4217-82a3-1ccd76940098" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe75_ret" diameter="DN400" length="97.6" name="Pipe75_ret" id="Pipe75_ret_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03178005504035" lon="4.311521298676168" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03188902735578" lon="4.3101054529409275" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="1a33dc29-d2c1-46da-a214-ac34db563a56" id="e23c6609-5456-44e7-bbef-213af17d50f2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="4e9a2741-bea2-48df-bb6a-8ffe8f445ed1" connectedTo="4c456b3e-d1c9-4cd0-ae4b-045a21340288" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe76" diameter="DN400" length="68.2" name="Pipe76_ret" id="Pipe76_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03188902735578" lon="4.3101054529409275" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03163484041849" lon="4.31101119777152" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="8961937d-3b73-487a-9bf2-1d982082c021" id="f947aeb7-1503-4665-be37-da2be8a53ade" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="7cc1c74d-db40-45ab-ad50-5ed0b697f795" connectedTo="1de51e8b-def4-4443-8ce4-b9eb14dc8e8a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe77" diameter="DN400" length="105.8" name="Pipe77_ret" id="Pipe77_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03056551822576" lon="4.309671892005951" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0313082377085" lon="4.310639968040655" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="f6a0a68a-1b36-4692-8513-01fac18923d1" id="bb90655a-7cef-4708-9d83-189901ee2090" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="5467a867-c193-4d61-925a-7c84200a8eaf" connectedTo="7995eac2-82c7-4456-8f8e-a4fa46c783a2" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe78" diameter="DN400" length="122.7" name="Pipe78_ret" id="Pipe78_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0313082377085" lon="4.310639968040655" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.030849353152654" lon="4.30900765447107" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="dc94e82b-2d27-4593-bf45-313f72814379" id="1308055c-ce56-406f-bad6-5df7eac7e0e3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="87141bc9-eff2-49db-be90-11d0e53ec3a6" connectedTo="c57e0353-939f-42ff-b6c0-6be7af99e5fc" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe60_a" diameter="DN250" length="120.04" name="Pipe60_a_ret" id="Pipe60_a_ret" innerDiameter="0.263" outerDiameter="0.4">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.029667795898824" lon="4.3136170795418565" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02967344649772" lon="4.313574183321056" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02954144297258" lon="4.313444990252005" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02933683673851" lon="4.313626686339623" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02905100631209" lon="4.31264939009626" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="a0d9f763-3754-4439-9509-003807bbdd65" id="32a5133e-bb0a-4bc2-9d41-907b310ebf6e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="a62cd13e-03bb-4323-b96e-5ac83870b099" connectedTo="cb5b59bf-b5a2-4948-ba3b-cbba2aea383b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe60_b" diameter="DN250" length="35.24" name="Pipe60_b_ret" id="Pipe60_b_ret" innerDiameter="0.263" outerDiameter="0.4">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02984303030444" lon="4.314046825510479" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.029667795898824" lon="4.3136170795418565" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="016ba65e-e955-4fd2-bdfd-8ebecb31702f" id="265e54cf-0576-4d80-b6ac-9bdb881575d5" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="645997b9-ff0f-4e08-949d-e499d42286de" connectedTo="ccbdf824-b95a-4116-8e48-55eb708f44c9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe34_a" diameter="DN150" length="165.61" name="Pipe34_a_ret" id="Pipe34_a_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02758184207936" lon="4.309672459453438" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.027492810873866" lon="4.310053025017606" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02810055900395" lon="4.311809287259864" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="fd235b70-a17f-48da-b6b4-90470317eabf" id="6ef65168-b48d-4edd-9d5c-d24ba52a9da1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="047ee5bd-2099-442f-b98b-552a081d039a" connectedTo="9ba3845f-c482-40bf-ac11-a20b37ca25df" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe34_b" diameter="DN150" length="48.65" name="Pipe34_b_ret" id="Pipe34_b_ret" innerDiameter="0.1603" outerDiameter="0.25">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02771773424247" lon="4.30899701251604" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02758184207936" lon="4.309672459453438" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="922d3ae9-e869-4ee0-bfc9-a00f7fdbf283" id="e578607d-8bb3-40ec-b689-d991e1bf4300" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8bad9ab5-164f-4cf1-96fd-7f7316aaafd6" connectedTo="cfff919b-0f98-40fb-a5c9-d733fff20bb4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe82" diameter="DN400" length="78.2" name="Pipe82_ret" id="Pipe82_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.028123143241935" lon="4.308944765720045" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02758184207936" lon="4.309672459453438" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="06c7f257-3a41-4f5e-9574-64727cca9a3f" id="bc6002db-b6fb-4e63-9acb-13aa1367da4c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8ea470e0-13bf-4144-a699-943cd3d6bd3c" connectedTo="cfff919b-0f98-40fb-a5c9-d733fff20bb4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe81" diameter="DN400" length="63.5" name="Pipe81_ret" id="Pipe81_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02758184207936" lon="4.309672459453438" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0281495480091" lon="4.309577857854631" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="fd235b70-a17f-48da-b6b4-90470317eabf" id="3aa80042-a30a-409d-94e6-32a6d3cb2232" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="0f8ad5ee-4492-4e07-bb07-e51f8289b390" connectedTo="983b16dc-f1ed-47a3-8eef-7116cec4cb5d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe79" diameter="DN400" length="47.9" name="Pipe79_ret" id="Pipe79_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.029628190086946" lon="4.314314319815374" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.029667795898824" lon="4.3136170795418565" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="b8c47d9e-9c4c-4404-80a4-54d139c8ea00" id="3d47c270-f866-46af-b581-26fb8d9811a4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="6099e2c6-e819-4bd5-b774-3adfb9913673" connectedTo="ccbdf824-b95a-4116-8e48-55eb708f44c9" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe80" diameter="DN400" length="59.2" name="Pipe80_ret" id="Pipe80_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.029667795898824" lon="4.3136170795418565" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.030195870038526" lon="4.313726150226659" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="a0d9f763-3754-4439-9509-003807bbdd65" id="b8f6e3cf-e0cb-412e-86fe-4018183ffb1a" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="cec7a770-b845-44f2-a048-1dd075cf9bc2" connectedTo="af926335-28b7-49d3-bb8a-102f5349ec67" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe83" diameter="DN400" length="47.5" name="Pipe83_ret" id="Pipe83_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.026307778118145" lon="4.314270693206706" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02614679831976" lon="4.313626398166783" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="fb2de451-5690-47c3-9447-4bcb65f5b683" id="36be5ca1-5f82-40c1-9845-43f86ea6ec37" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="cd2edc0a-a86a-43aa-8758-2583a5d1b953" connectedTo="7ac9b957-d251-4415-afb0-42c89590fb47" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe84" diameter="DN400" length="55.7" name="Pipe84_ret" id="Pipe84_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.02614679831976" lon="4.313626398166783" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.02659824149003" lon="4.313982031432099" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="cd825ba8-0050-4b8b-96e8-511f6b1e025c" id="de53da75-834f-4750-8776-49bbb4f510f1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="da69fa88-7382-4cd7-9252-e7569aa71709" connectedTo="f4cb60f0-0db8-44ca-9e7e-f3952e867f4b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe85" diameter="DN400" length="32.6" name="Pipe85_ret" id="Pipe85_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04166674135581" lon="4.3117026520863035" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.041739007433215" lon="4.312164208605956" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="b7acc708-f7d2-4dd5-95ae-d5265c59fb13" id="4d317d5f-fdd9-47b8-b5e7-7789e8aa16eb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="4b08c6c4-14d3-4715-9f96-e04ab147c9c1" connectedTo="675d388d-ece7-48c3-bbc5-e4b73b9510f3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe86" diameter="DN400" length="41.0" name="Pipe86_ret" id="Pipe86_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.041739007433215" lon="4.312164208605956" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04140277280331" lon="4.311916436432105" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="aa141fc0-ce28-4d14-99be-bed98630e2e5" id="6d43e535-b95f-41ea-8489-7c85d74ff650" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e1bf78da-811b-4b3f-a3e1-6681a3dd7a77" connectedTo="ca759595-c15b-44fe-9e33-2e2ccfd2d459" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe87" diameter="DN400" length="39.8" name="Pipe87_ret" id="Pipe87_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.039092981471256" lon="4.316544271458594" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.038888193298845" lon="4.316066207877817" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="d1a8eb75-ec20-49bb-b581-e2f6e48be4a6" id="1ab46cfb-031a-49f5-a1ea-31dbe10ba17e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8681bff7-98ab-42ff-9d8b-8eb4f8df2ee6" connectedTo="2ec8715c-c5b9-4782-918c-f1b35d426904" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe88" diameter="DN400" length="73.7" name="Pipe88_ret" id="Pipe88_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.038888193298845" lon="4.316066207877817" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03954175027192" lon="4.316245240967744" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="401eae5a-7174-4c37-b64d-33c87782ed60" id="da149f20-9c37-46ca-b006-c2b63a1c6e1f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="a2e00dd2-85d1-4ad5-a8f7-c2258fb36c47" connectedTo="88e26467-fe6f-48f2-ae81-a0fe0a44f184" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe89" diameter="DN400" length="30.5" name="Pipe89_ret" id="Pipe89_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03469744888936" lon="4.318268524707502" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034823143777416" lon="4.318665895348221" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="acbbb125-fd6a-454e-a645-70ac2e831883" id="e5a5d161-9a32-4d74-8881-1f4f6848c815" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="55abdde6-75c5-4fd7-9ceb-a4f7fe531e03" connectedTo="28b38730-7167-476f-8962-7d09d9f9d417" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe90" diameter="DN400" length="46.5" name="Pipe90_ret" id="Pipe90_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.034823143777416" lon="4.318665895348221" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.034420238654135" lon="4.318482209262021" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="6f78a5f5-edd6-4641-afce-1f57fdb304b3" id="aa2d568a-0e68-41a6-9ba1-9adcf0d9dede" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="c44e43f9-9508-4a85-8d9c-c23f9ffd7083" connectedTo="b25eaf74-e967-4ca2-b8dc-0f5e2df6f719" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe91" diameter="DN400" length="38.1" name="Pipe91_ret" id="Pipe91_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03282294619615" lon="4.319378241420888" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03296207243976" lon="4.319888316571063" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="079a3f77-803c-487a-b141-a845644b0d5b" id="ccf83cae-ce9a-4ab7-89f6-180ae3884858" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="36c80210-eb59-4981-a9dd-6c707c321160" connectedTo="425cb5ab-fe83-4313-82bc-6ea70546c77e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe92" diameter="DN400" length="54.5" name="Pipe92_ret" id="Pipe92_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03296207243976" lon="4.319888316571063" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03247971888093" lon="4.319741895491652" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="122ee1ba-078e-4d41-8525-426ff44fbef8" id="407c747e-5e0f-4a08-84aa-6ffb2a6eb7ca" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="79fffb1a-d928-48ab-99f2-25e0f5fe9af0" connectedTo="d9838289-335e-4070-a7fa-12423f2c62de" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe23_a" diameter="DN200" length="376.75" name="Pipe23_a_ret" id="Pipe23_a_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03170084712326" lon="4.323172552258066" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03101116205779" lon="4.32168967919351" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03296207243976" lon="4.319888316571063" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4a3c0adb-12d0-4731-9dc5-021bf4f1faa4" id="757cd47f-b98d-4340-b5a0-af50c62ccb66" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="47825b4e-4273-4b51-8c55-f6200eb7253d" connectedTo="425cb5ab-fe83-4313-82bc-6ea70546c77e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe23_b" diameter="DN200" length="39.78" name="Pipe23_b_ret" id="Pipe23_b_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03191866855768" lon="4.323634613234376" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03170084712326" lon="4.323172552258066" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="17944386-a137-4cac-965e-e8b5a399bb3c" id="129982a4-6b66-43c8-ba16-1b2ee1caf049" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="aee25d4b-808b-41c5-b6a6-46cfd0534f61" connectedTo="adb7329e-ec83-40e4-967a-c5c33ffe7550" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe93" diameter="DN400" length="61.7" name="Pipe93_ret" id="Pipe93_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03146322253006" lon="4.323987155372704" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03170084712326" lon="4.323172552258066" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="13874a79-3fbe-418c-a19d-2c0f98228bee" id="5b07d92e-c441-49eb-a123-e81d02a2a358" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="0355616b-1d86-4553-9f23-3b87ed3966ca" connectedTo="adb7329e-ec83-40e4-967a-c5c33ffe7550" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe94" diameter="DN400" length="69.0" name="Pipe94_ret" id="Pipe94_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03170084712326" lon="4.323172552258066" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.03226850076201" lon="4.322766732139652" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="4a3c0adb-12d0-4731-9dc5-021bf4f1faa4" id="ebfa9843-4015-4256-a33c-13036255719d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="b47dfdc3-f363-4220-82d6-a9e73bb84f13" connectedTo="74f2d395-2b58-4f92-8253-5996a7e1eada" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe95" diameter="DN400" length="36.4" name="Pipe95_ret" id="Pipe95_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04288097661879" lon="4.317328180768408" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04298433842342" lon="4.317832742116763" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="1135996f-b4d2-48b7-905a-82be2f2415dd" id="2fbbe982-4037-4169-a088-28bd455d9da0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="c7ae6629-a8da-4508-be88-26992fb56543" connectedTo="e6aa2067-fbd6-4a16-8ca2-0151b8d389ee" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe96" diameter="DN400" length="43.2" name="Pipe96_ret" id="Pipe96_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04298433842342" lon="4.317832742116763" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04265001049516" lon="4.317509885975817" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="77fbb03a-8497-4aa5-b458-be03c126ca8a" id="64654e0a-ad86-491f-b5a4-69e069d340e0" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e14b3d48-c8d0-47ce-af2e-bcd5173687cd" connectedTo="9d08b748-1db7-42fd-ae46-4645ed9b83fd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe97" diameter="DN400" length="52.6" name="Pipe97_ret" id="Pipe97_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.040041443604586" lon="4.321031827826063" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0397727324572" lon="4.3216640094954055" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="7ab22d2a-a5ce-43d4-a8b1-e7d93d543c89" id="9bfc3f39-1745-45f5-a649-44cf83c54098" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="47411f3a-7951-4684-bc58-dbbc2f86147d" connectedTo="b95748d5-64e4-46fd-9dbc-0b3528f6e3ff" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe98" diameter="DN400" length="83.1" name="Pipe98_ret" id="Pipe98_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04179872504749" lon="4.323268732775283" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.041110409870306" lon="4.322794596185427" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="340abfbb-4247-49d0-ae09-deff4c6598ce" id="172d79dd-fa55-4205-9987-60ddc13ef5a6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d8ee80c7-3262-4aea-bd4f-2fc526e26fbc" connectedTo="c693a131-31b6-4b39-8999-bd57adb76b7b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe99" diameter="DN400" length="93.3" name="Pipe99_ret" id="Pipe99_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.041110409870306" lon="4.322794596185427" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04146876508759" lon="4.324029490630251" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="5c25fefc-e271-4d05-8cff-502ef0abc52e" id="729da237-9bbb-40ed-b85c-cb89d7e5aece" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="b65eee76-8d5c-46f5-aa26-6aa060435ec2" connectedTo="04f3033e-e363-4f74-bb59-c1de3f1af911" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe100" diameter="DN400" length="95.3" name="Pipe100_ret" id="Pipe100_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04283478348955" lon="4.325428315116056" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.042449880893734" lon="4.326671717170104" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="48bfbe22-4e4b-4046-9944-fce648d16f83" id="d1189e55-b3de-4a66-8a2a-4bb173c3b9bb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="4714cac6-209a-4ab0-a0ca-37bf76468f55" connectedTo="38adcf2e-fd5e-44d7-9889-e41635df0822" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe101" diameter="DN400" length="94.4" name="Pipe101_ret" id="Pipe101_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.042449880893734" lon="4.326671717170104" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04321092615362" lon="4.32606242922094" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="7ba5ab0e-cd6f-461b-a949-9ff998a32223" id="c5d770bd-55e6-47d8-b851-ad799add08d6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="60b1043e-22bd-4119-98b7-b282144c0967" connectedTo="f572e6a0-69de-40e5-ba67-cca9d0eb7452" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe102" diameter="DN400" length="116.7" name="Pipe102_ret" id="Pipe102_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04420735896327" lon="4.327256258250727" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.043710134712654" lon="4.328756837645373" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="7757e190-4910-46ca-a000-024abb383b4a" id="f676d4da-6721-43b5-9bfd-655d86de67d7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e4eb47c8-4f12-4b82-97fa-33a2f4d0af5d" connectedTo="0328e4a0-699e-4011-8090-cec6925ac527" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe103" diameter="DN400" length="92.7" name="Pipe103_ret" id="Pipe103_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.043710134712654" lon="4.328756837645373" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.043989597398486" lon="4.3274809262877705" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="5ee06323-235b-487e-b10c-24e037f15bcb" id="ec7a8701-76d6-4834-adc4-dde7340fd71e" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="dc290931-0cdf-444e-9f3a-dc8b52ff4920" connectedTo="4ffa71aa-a584-41e1-97a1-61199da28891" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe104" diameter="DN400" length="32.2" name="Pipe104_ret" id="Pipe104_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04638491625592" lon="4.329891124863761" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.046672580962834" lon="4.329945591703785" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="96a3052d-6237-49d0-85ed-bd65235522d0" id="103cda9c-f400-4499-ab0d-fe99b25e5c9d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8f7969c1-f2f5-4816-8990-93b23c48a47e" connectedTo="f108633c-9c56-48dd-9ac8-de2b8ce206cf" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe105" diameter="DN400" length="39.5" name="Pipe105_ret" id="Pipe105_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.046672580962834" lon="4.329945591703785" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.046886399267876" lon="4.330407541661346" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="c5ef7be7-3d9c-4e71-b574-921fe2b63dbf" id="f7297b6d-23d6-43a7-8afa-f5e2f1bbfdfb" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="3800b6d2-cfdb-4dd6-8cff-72156de139fe" connectedTo="95556d67-1ffd-4d8e-afb1-83c6e63e8f5f" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe106" diameter="DN400" length="78.2" name="Pipe106_ret" id="Pipe106_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.049426720172605" lon="4.3337084472668765" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0501296195939" lon="4.333667480856885" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="1c6756d2-1797-4221-94d7-e74888cdf112" id="7dbbd7a6-e225-4778-b65f-a8795c732295" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="b4e684d3-fee7-4771-ba3f-74f18cf24b60" connectedTo="5081d9f4-d1f6-444f-9222-104e4b0fe984" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe107" diameter="DN400" length="67.5" name="Pipe107_ret" id="Pipe107_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0501296195939" lon="4.333667480856885" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04988198320488" lon="4.334568017938316" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="9878324e-69e9-46c6-8d3a-b5b6fe013323" id="4db670e9-6af7-46aa-9b51-32ec5ec041d1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="f2d22a19-aaeb-418e-8bcb-cdc789d3b956" connectedTo="44e1a886-5a4e-41c8-8921-4cb53eef3dc6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe108" diameter="DN400" length="58.1" name="Pipe108_ret" id="Pipe108_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05101682158594" lon="4.336974403947429" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0508684939238" lon="4.33615860562574" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="a863d129-71e2-416d-8c1a-058a380dcd80" id="b88490b8-5cd4-4563-9903-13efe321a07d" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="10986dc7-33f8-4b27-ac49-c5944176c585" connectedTo="7bd8ead4-d8f9-40d4-a32a-5b33300cd3b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe109" diameter="DN400" length="69.3" name="Pipe109_ret" id="Pipe109_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.0508684939238" lon="4.33615860562574" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.05148526390875" lon="4.3363104970784905" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="767cda8d-cfe3-4915-9f45-c4e2b0841eeb" id="8aa495f9-bfcd-4d2f-9d76-5294443b96f8" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="fc239a2f-c737-47ee-89c3-c69baed28e51" connectedTo="d4012c03-8dfd-4a9f-8674-9774f1dd612b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe49_a" diameter="DN200" length="277.05" name="Pipe49_a_ret" id="Pipe49_a_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.048813097442256" lon="4.336775182356948" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04942476122613" lon="4.337603009306805" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0508684939238" lon="4.33615860562574" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="f5813dd3-9d4c-4aa8-9e00-a9ec870007b3" id="35d57051-f40b-4aa4-a9e2-ac4eb7c347f7" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="de415c15-2b8d-45fd-a173-bfca996554f9" connectedTo="7bd8ead4-d8f9-40d4-a32a-5b33300cd3b3" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe49_b" diameter="DN200" length="47.47" name="Pipe49_b_ret" id="Pipe49_b_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04859966024021" lon="4.337246653309526" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04875183657353" lon="4.336721366709887" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.048813097442256" lon="4.336775182356948" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="827b6304-b33d-4a73-83ec-2111e6a3c837" id="08705ddf-393d-4827-aba5-e45902bd65bd" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="bbb87a16-7099-448a-843d-e7376bcee47e" connectedTo="0e642c28-26ce-496a-9b16-a83a037accf1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe52_a" diameter="DN200" length="151.23" name="Pipe52_a_ret" id="Pipe52_a_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05221100992171" lon="4.332074580336848" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.05287394837437" lon="4.334007561882603" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="ba9aa151-e244-4a01-bbb3-1c4ead3f8e8a" id="f98c1cb7-877d-4640-bbc4-b67d43106659" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="f08f0ea0-e8f2-44b0-aee5-553ea8af18df" connectedTo="ca8dd588-f4c7-45c8-bae8-eba8c790c466" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe52_b" diameter="DN200" length="27.64" name="Pipe52_b_ret" id="Pipe52_b_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.052069743515325" lon="4.331741603326295" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.05221100992171" lon="4.332074580336848" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="df8b4206-fbd9-4b20-b477-13129693dd70" id="9f295aaf-a774-4a72-afcc-64d131904643" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="3472767c-618c-4c0f-b8a2-2891f8bb2440" connectedTo="3f5063b1-c676-4b81-bdaa-bfdd919a10ca" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe51_a" diameter="DN200" length="249.1" name="Pipe51_a_ret" id="Pipe51_a_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.054817000711104" lon="4.33219957498349" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.05287394837437" lon="4.334007561882603" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="ca7ee8ee-caa3-4ae4-ae4b-c1ab3604022e" id="442c90bc-79fa-41e4-85bb-abf96e05feee" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="a0825eaa-aa90-4df6-b0e1-115903aaae38" connectedTo="ca8dd588-f4c7-45c8-bae8-eba8c790c466" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe51_b" diameter="DN200" length="34.31" name="Pipe51_b_ret" id="Pipe51_b_ret" innerDiameter="0.2101" outerDiameter="0.315">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05507781056326" lon="4.331932043163073" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.054817000711104" lon="4.33219957498349" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="8b9f02d3-cbb2-4584-b8f7-16e181318b58" id="3b2d411f-d681-4539-a42f-4e6bc5df4f41" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e703e2e0-2a3c-48fb-89ef-19d0fd0109ed" connectedTo="621ab362-a3a1-464c-ae55-ecda1ee72c32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe110" diameter="DN400" length="63.0" name="Pipe110_ret" id="Pipe110_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.054817000711104" lon="4.33219957498349" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.05496214003323" lon="4.3330904520510565" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="ca7ee8ee-caa3-4ae4-ae4b-c1ab3604022e" id="6fa9faf7-af3b-4987-90a8-847da7591e45" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="8410eb59-05a7-4f17-aca6-9cca0adcaf05" connectedTo="49c5ab59-21bf-4d65-9071-1c73a003cfa5" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe111" diameter="DN400" length="55.8" name="Pipe111_ret" id="Pipe111_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.055226028501785" lon="4.332672723717209" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.054817000711104" lon="4.33219957498349" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="38a7d6c4-b647-47c5-97df-3bfa3cd35fa8" id="ebc2e9bc-2d17-4398-a182-9eb0026fef52" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="d4082c49-e51f-452e-b22f-8a818d534c64" connectedTo="621ab362-a3a1-464c-ae55-ecda1ee72c32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe112" diameter="DN400" length="39.9" name="Pipe112_ret" id="Pipe112_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05185473607928" lon="4.33213798622169" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.05221100992171" lon="4.332074580336848" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="44ad7156-ef46-4550-974c-fe86d2b34d73" id="3c8b30af-bfd3-44f9-9653-ac48cac85a32" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="78821a62-b0a9-47bd-b271-d58622c8d8c2" connectedTo="3f5063b1-c676-4b81-bdaa-bfdd919a10ca" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe113" diameter="DN400" length="36.1" name="Pipe113_ret" id="Pipe113_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.05221100992171" lon="4.332074580336848" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.0525013050262" lon="4.331839331594479" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="ba9aa151-e244-4a01-bbb3-1c4ead3f8e8a" id="4bb8d3f2-2399-44fc-91e4-d3d10dc7efd6" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="32a75cd3-df73-4797-988c-af0b84e781c3" connectedTo="b059f601-3773-4380-bc23-34cc480bc769" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe114" diameter="DN400" length="41.4" name="Pipe114_ret" id="Pipe114_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04844360012072" lon="4.336699044781149" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.048813097442256" lon="4.336775182356948" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="bec0449f-b062-49f5-b4bd-ba9aadf2a76c" id="d78f6cd1-f89d-41d7-adc0-986f67d9ac60" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="0b2c48de-6dce-4a48-9f33-15445eb38689" connectedTo="0e642c28-26ce-496a-9b16-a83a037accf1" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe115" diameter="DN400" length="57.8" name="Pipe115_ret" id="Pipe115_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.048813097442256" lon="4.336775182356948" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.04865474182142" lon="4.335970076217015" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="f5813dd3-9d4c-4aa8-9e00-a9ec870007b3" id="c1a2f0f9-6cc0-4483-9175-62f12eda11b4" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="e238321b-26ad-4b91-8314-7e1624b36a94" connectedTo="9548a1f0-f3eb-4a95-a592-2356ab97130c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe116" diameter="DN400" length="34.4" name="Pipe116_ret" id="Pipe116_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.03991792064847" lon="4.320570111360173" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.040041443604586" lon="4.321031827826063" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="20653a96-e8e6-40fd-9f5c-fc2d1bb4cc86" id="5671f07e-0cf8-4a47-8a05-32b05b01bd8b" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="11aa30ed-b270-4ed2-a24e-495bb069d71e" connectedTo="10ca3dd3-92cf-468a-bdef-fcd982ce8859" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
      <asset xsi:type="esdl:Pipe" related="Pipe117" diameter="DN400" length="24.1" name="Pipe117_ret" id="Pipe117_ret" innerDiameter="0.3938" outerDiameter="0.56">
        <geometry xsi:type="esdl:Line">
          <point xsi:type="esdl:Point" lat="52.04448904517101" lon="4.3095618774998306" CRS="WGS84"/>
          <point xsi:type="esdl:Point" lat="52.044559569551275" lon="4.309229489199389" CRS="WGS84"/>
        </geometry>
        <port xsi:type="esdl:InPort" name="In_ret" connectedTo="d49a9b33-62b9-43ee-aa3c-19e087899111" id="d4c7ba6d-d591-4fe9-8436-b9aaa334e1db" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
        <port xsi:type="esdl:OutPort" name="Out_ret" id="f2dc36cf-a2ab-41e1-933a-dfe5a1d8c118" connectedTo="413a8ce6-4b2e-442c-acca-21b5d928130c" carrier="39ec9ed1-30d6-40db-83a4-1bac7b0b3c75_ret"/>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
