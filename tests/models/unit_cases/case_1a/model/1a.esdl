<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem description="" id="b9f8b83d-49e2-4c8b-9992-240ed4643329" name="1a" xmlns:esdl="http://www.tno.nl/esdl" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <instance id="22cf0465-ceef-486a-b8d0-278c131cbece" name="Untitled Instance" xsi:type="esdl:Instance">
    <area id="a5845d28-ed01-4f32-91dc-654632ecc997" name="Untitled Area" xsi:type="esdl:Area">
      <asset id="2ab92324-f86e-4976-9a6e-f7454b77ba3c" minTemperature="70.0" name="HeatingDemand_2ab9" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <geometry CRS="WGS84" lat="51.98612564800895" lon="4.38157081604004" xsi:type="esdl:Point"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="95f46ccf-bac8-4a44-a854-2bbe2fb3c5e6" id="3f514c6b-fd11-4821-9e6c-4a4d13d46762" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.1" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="75a9a056-987d-41dc-9ecf-ca3edfef31d8" id="d86d3ed7-77d8-4766-bfa1-ed9209edf0b6" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="6662aebb-f85e-4df3-9f7e-c58993586fba" minTemperature="70.0" name="HeatingDemand_6662" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <geometry CRS="WGS84" lat="51.985484727746204" lon="4.381640553474427" xsi:type="esdl:Point"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="99e5f517-4fbb-4e1e-89e2-606f1121adeb" id="5f607bc1-31a6-4bc8-8911-aefd4d2cfc4d" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.1" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="27daf107-0ce8-4870-bb8c-45421a62ed83" id="560eb546-6e19-4fb7-9dce-741ed12310a6" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="8172d5d3-61a4-4d0b-a26f-5e61c2a22c64" maxTemperature="85.0" minTemperature="65.0" name="GenericProducer_8172" power="2000000.0" xsi:type="esdl:ResidualHeatSource">
        <geometry CRS="WGS84" lat="51.98561687700459" lon="4.379387497901917" xsi:type="esdl:Point"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="3c182b9c-006d-4ac1-b817-dc884e36c0b0" id="974230fb-e753-4fc1-a762-82de7d960e62" name="Out" xsi:type="esdl:OutPort"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="237e32d1-2434-4b23-988b-a0b288f858bb" id="4b5d5d6c-82ef-441b-9e0b-478cf3c8031c" name="In" xsi:type="esdl:InPort"/>
      </asset>
      <asset id="ed9a2b96-2fff-4650-b875-41c6f05a6e44" name="Joint_ed9a" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.985749025873034" lon="4.3802350759506234" xsi:type="esdl:Point"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="a4814870-c191-4856-9104-f2206e16b54d" id="adb264b4-2ba0-4a49-9fc7-f57a29fb6346" name="In" xsi:type="esdl:InPort"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="a779e3ec-8215-43c9-84ff-baa1c93abfea df318cbd-81f2-467c-a081-7feb4256dbfa 14757d8c-9685-47e2-bf2a-517ee556ab56" id="061f1df2-f724-4b08-aa5f-6f19b06c2f92" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="95802cf8-61d6-4773-bb99-e275c3bf26cc" name="Joint_9580" xsi:type="esdl:Joint">
        <geometry CRS="WGS84" lat="51.98563669935972" lon="4.380224347114564" xsi:type="esdl:Point"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="388ae7b7-2a05-4b50-acf7-adebf4c24ab4 1f153f2f-adf6-4e3b-94ca-aef1644e8c7d c7ef3ba2-332e-4c8d-bc53-449179a4bc22" id="e3bbc000-98be-43d7-898e-7a503871786f" name="In" xsi:type="esdl:InPort"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="1ad8a362-195d-458b-86c2-8c94b3993547" id="ba6017c5-bedf-411c-9e49-72f0b5053991" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="275a0b40-b6ce-4e7d-b954-efeff0f25734" innerDiameter="0.16030000000000003" length="72.13358767664306" name="Pipe_275a" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98579197417133" lon="4.380412101745606" xsi:type="esdl:Point"/>
          <point lat="51.98609261110622" lon="4.381345510482789" xsi:type="esdl:Point"/>
        </geometry>
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
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="061f1df2-f724-4b08-aa5f-6f19b06c2f92" id="a779e3ec-8215-43c9-84ff-baa1c93abfea" name="In" xsi:type="esdl:InPort"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="3f514c6b-fd11-4821-9e6c-4a4d13d46762" id="95f46ccf-bac8-4a44-a854-2bbe2fb3c5e6" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="b5babf87-0530-49b5-914d-008d733ac5bf" innerDiameter="0.16030000000000003" length="73.56064145043287" name="Pipe_b5ba" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98563339563447" lon="4.380374550819398" xsi:type="esdl:Point"/>
          <point lat="51.98549794268959" lon="4.381425976753236" xsi:type="esdl:Point"/>
        </geometry>
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
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="061f1df2-f724-4b08-aa5f-6f19b06c2f92" id="14757d8c-9685-47e2-bf2a-517ee556ab56" name="In" xsi:type="esdl:InPort"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="5f607bc1-31a6-4bc8-8911-aefd4d2cfc4d" id="99e5f517-4fbb-4e1e-89e2-606f1121adeb" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="e6c67c33-ad36-4681-b21e-01b1eabc15c4" innerDiameter="0.16030000000000003" length="31.412804660774878" name="Pipe_e6c6" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98563669935972" lon="4.379612803459168" xsi:type="esdl:Point"/>
          <point lat="51.985689558930474" lon="4.38006341457367" xsi:type="esdl:Point"/>
        </geometry>
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
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="974230fb-e753-4fc1-a762-82de7d960e62" id="3c182b9c-006d-4ac1-b817-dc884e36c0b0" name="In" xsi:type="esdl:InPort"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="adb264b4-2ba0-4a49-9fc7-f57a29fb6346" id="a4814870-c191-4856-9104-f2206e16b54d" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="5871f6ef-5d92-4a13-8843-6c56c2ac747f" innerDiameter="0.16030000000000003" length="70.32745329931934" name="Pipe_5871" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98571598869247" lon="4.380438923835755" xsi:type="esdl:Point"/>
          <point lat="51.985758937022446" lon="4.381463527679444" xsi:type="esdl:Point"/>
        </geometry>
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
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="061f1df2-f724-4b08-aa5f-6f19b06c2f92" id="df318cbd-81f2-467c-a081-7feb4256dbfa" name="In" xsi:type="esdl:InPort"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="ee56b88f-a264-4389-ba57-6f2b52a05e1c" id="c9d98764-3e85-4676-9504-73f7eeb242f5" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="506c41ac-d415-4482-bf10-bf12f17aeac6" minTemperature="70.0" name="HeatingDemand_506c" power="1000000.0" xsi:type="esdl:HeatingDemand">
        <geometry CRS="WGS84" lat="51.98578206302921" lon="4.381629824638368" xsi:type="esdl:Point"/>
        <port carrier="c362f53a-3eaf-4d96-8ee6-944e77359fed" connectedTo="c9d98764-3e85-4676-9504-73f7eeb242f5" id="ee56b88f-a264-4389-ba57-6f2b52a05e1c" name="In" xsi:type="esdl:InPort">
          <profile xsi:type="esdl:SingleValue" value="0.1" id="5317d120-2e6e-415d-a3bc-bdd6268997d3">
            <profileQuantityAndUnit xsi:type="esdl:QuantityAndUnitReference" reference="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
          </profile>
        </port>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="e3839971-e5b0-4b3a-9c55-ac2baab8966a" id="a4d4447f-683c-4daf-aafd-f20efc13bd43" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="22771285-490c-42e3-a6be-82c383b5f21e" innerDiameter="0.16030000000000003" length="72.13358767664306" name="Pipe_275a_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98607261110622" lon="4.381325510482789" xsi:type="esdl:Point"/>
          <point lat="51.98577197417133" lon="4.380392101745606" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="d86d3ed7-77d8-4766-bfa1-ed9209edf0b6" id="75a9a056-987d-41dc-9ecf-ca3edfef31d8" name="In" xsi:type="esdl:InPort"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="e3bbc000-98be-43d7-898e-7a503871786f" id="c7ef3ba2-332e-4c8d-bc53-449179a4bc22" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="e68bff9f-422c-4a1b-90b0-80a5bfc1cabf" innerDiameter="0.16030000000000003" length="70.32745329931934" name="Pipe_5871_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98573893702245" lon="4.381443527679444" xsi:type="esdl:Point"/>
          <point lat="51.98569598869247" lon="4.380418923835755" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="a4d4447f-683c-4daf-aafd-f20efc13bd43" id="e3839971-e5b0-4b3a-9c55-ac2baab8966a" name="In" xsi:type="esdl:InPort"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="e3bbc000-98be-43d7-898e-7a503871786f" id="1f153f2f-adf6-4e3b-94ca-aef1644e8c7d" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="4605c1d2-62fc-40f3-8451-c6335b8d822a" innerDiameter="0.16030000000000003" length="73.56064145043287" name="Pipe_b5ba_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.98547794268959" lon="4.381405976753236" xsi:type="esdl:Point"/>
          <point lat="51.98561339563447" lon="4.380354550819398" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="560eb546-6e19-4fb7-9dce-741ed12310a6" id="27daf107-0ce8-4870-bb8c-45421a62ed83" name="In" xsi:type="esdl:InPort"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="e3bbc000-98be-43d7-898e-7a503871786f" id="388ae7b7-2a05-4b50-acf7-adebf4c24ab4" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
      <asset id="69b66347-386f-48fb-920b-0fa1fe872bf6" innerDiameter="0.16030000000000003" length="31.412804660774878" name="Pipe_e6c6_ret" outerDiameter="0.25" xsi:type="esdl:Pipe">
        <geometry CRS="WGS84" xsi:type="esdl:Line">
          <point lat="51.985669558930475" lon="4.38004341457367" xsi:type="esdl:Point"/>
          <point lat="51.985616699359724" lon="4.379592803459168" xsi:type="esdl:Point"/>
        </geometry>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="ba6017c5-bedf-411c-9e49-72f0b5053991" id="1ad8a362-195d-458b-86c2-8c94b3993547" name="In" xsi:type="esdl:InPort"/>
        <port carrier="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" connectedTo="4b5d5d6c-82ef-441b-9e0b-478cf3c8031c" id="237e32d1-2434-4b23-988b-a0b288f858bb" name="Out" xsi:type="esdl:OutPort"/>
      </asset>
    </area>
  </instance>
  <energySystemInformation id="eebe64dd-045f-49d5-9177-1e9a96a030f3" xsi:type="esdl:EnergySystemInformation">
    <carriers id="823f0f5a-89a9-498e-a6b7-482a08ad95b2" xsi:type="esdl:Carriers">
      <carrier id="1abfe4c6-7fce-4bd2-bcbc-618f668cfb41" name="Heat_cold" returnTemperature="45.0" xsi:type="esdl:HeatCommodity"/>
      <carrier id="c362f53a-3eaf-4d96-8ee6-944e77359fed" name="Heat_hot" supplyTemperature="75.0" xsi:type="esdl:HeatCommodity"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="c7104449-b5eb-49c7-b064-a2a4b16ae6e4">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Power in MW" unit="WATT" physicalQuantity="POWER" multiplier="MEGA" id="e9405fc8-5e57-4df5-8584-4babee7cdf1b"/>
    </quantityAndUnits>
  </energySystemInformation>
</esdl:EnergySystem>
