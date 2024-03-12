<?xml version='1.0' encoding='UTF-8'?>
<esdl:EnergySystem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esdl="http://www.tno.nl/esdl" version="6" name="Untitled EnergySystem" esdlVersion="v2401" description="" id="415d9426-dae2-4194-88ad-607793871425">
  <energySystemInformation xsi:type="esdl:EnergySystemInformation" id="7997daa8-8ded-4f31-96b2-6842510f43f2">
    <carriers xsi:type="esdl:Carriers" id="6e119417-37a2-4dfd-a493-95f4b68d0235">
      <carrier xsi:type="esdl:ElectricityCommodity" voltage="132000.0" id="e1077d81-b32a-4004-9a33-db65c96b5f4c" name="Elec"/>
      <carrier xsi:type="esdl:GasCommodity" name="Hydrogen" id="14831e6c-c3bb-4763-8300-9658a365ee54" pressure="30.0"/>
    </carriers>
    <quantityAndUnits xsi:type="esdl:QuantityAndUnits" id="c3280571-4963-42db-b466-073a465b67b3">
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Energy in kWh" physicalQuantity="ENERGY" multiplier="KILO" id="12c481c0-f81e-49b6-9767-90457684d24a" unit="WATTHOUR"/>
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Energy in MWh" physicalQuantity="ENERGY" multiplier="MEGA" id="93aa23ea-4c5d-4969-97d4-2a4b2720e523" unit="WATTHOUR"/>
      <quantityAndUnit xsi:type="esdl:QuantityAndUnitType" description="Energy in GWh" physicalQuantity="ENERGY" multiplier="GIGA" id="6fcd2303-b504-4939-8312-8cba25749265" unit="WATTHOUR"/>
    </quantityAndUnits>
  </energySystemInformation>
  <instance xsi:type="esdl:Instance" id="791c9938-3b6c-47b3-bcc5-0328ffd687bd" name="Untitled Instance">
    <area xsi:type="esdl:Area" name="Untitled Area" id="3f8efe6b-ccda-4054-85d0-38cbcd6977da">
      <asset xsi:type="esdl:WindPark" technicalLifetime="20.0" id="90740caf-4fbd-45ed-8ea3-6af19c76256c" surfaceArea="1294320493" name="WindPark_9074" power="2000000000.0">
        <port xsi:type="esdl:OutPort" connectedTo="b0ed14ac-faa8-499f-aa98-53be98852d0f" name="Out" id="fd1689e6-1339-4b77-8f8f-94acc57c752f" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Polygon" CRS="WGS84">
          <exterior xsi:type="esdl:SubPolygon">
            <point xsi:type="esdl:Point" lon="2.9498291015625004" lat="52.311837071418886"/>
            <point xsi:type="esdl:Point" lon="3.8726806640625004" lat="52.48612543090347"/>
            <point xsi:type="esdl:Point" lon="3.6694335937500004" lat="52.68304276227743"/>
            <point xsi:type="esdl:Point" lon="2.9223632812500004" lat="52.47274306920925"/>
          </exterior>
        </geometry>
      </asset>
      <asset xsi:type="esdl:ElectricityDemand" technicalLifetime="20.0" id="f8339608-af60-4b32-945b-521e6f7b8098" name="ElectricityDemand_f833" power="2000000000.0">
        <port xsi:type="esdl:InPort" connectedTo="55709cf7-8ad7-4bc0-9f3d-748c0788a398" name="In" id="ac3fc355-1545-4b78-a46c-1b0908f5ccde" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Point" lon="4.872436523437501" CRS="WGS84" lat="52.36553758871974"/>
      </asset>
      <asset xsi:type="esdl:Electrolyzer" effMaxLoad="69.0" technicalLifetime="20.0" maxLoad="200000000" id="6327ee2b-9f17-432f-8d15-b034422eab79" efficiency="63.0" effMinLoad="67.0" power="200000000.0" minLoad="20000000" name="Electrolyzer_6327">
        <port xsi:type="esdl:InPort" connectedTo="1e76b7b0-9f17-4ccc-b4c3-0089b3ff6a45" name="In" id="d81bb99d-1508-4b13-bc13-1f1a149d481d" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:OutPort" connectedTo="5e660a4f-e8ca-4e9d-8568-6292d36b5994" name="Out" id="5e58c0e2-5db4-4f6a-8aab-2c880d499c14" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Point" lon="3.6419677734375004" CRS="WGS84" lat="52.579688026538726"/>
      </asset>
      <asset xsi:type="esdl:GasDemand" technicalLifetime="20.0" id="41466625-a14b-43a9-9a13-93d34a4ea6ff" name="GasDemand_4146" power="500000000.0">
        <port xsi:type="esdl:InPort" connectedTo="f0b7ab61-b983-45ed-9274-887ccaadfce0" name="In" id="121a1b8d-758b-4b35-92fa-894435965e85" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Point" lon="4.707641601562501" CRS="WGS84" lat="52.633062890594374"/>
      </asset>
      <asset xsi:type="esdl:GasStorage" technicalLifetime="20.0" workingVolume="1000.0" id="9172f2eb-e1f4-4230-9495-396504f7c3c6" maxChargeRate="100000000.0" name="GasStorage_9172" maxDischargeRate="100000000.0">
        <port xsi:type="esdl:InPort" connectedTo="0e6e6a7d-5b9a-42ff-8902-4e8db530b248" name="In" id="b2bd6049-3e2e-443c-aa36-ba893135cf82" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Point" lon="3.8177490234375004" CRS="WGS84" lat="52.566334145326486"/>
      </asset>
      <asset xsi:type="esdl:Bus" technicalLifetime="20.0" id="24cf77d5-b66e-4362-9132-1065425d2c6a" name="Bus_24cf">
        <port xsi:type="esdl:InPort" connectedTo="62807929-3c55-4545-8279-2f655ab7e943" name="In" id="8f93331f-7785-4213-a6e0-d120a8f0fba7" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:OutPort" connectedTo="801d733f-16f6-4e5c-a22f-afd204d59776 949dd468-5c21-4a52-9596-cb4a50a113a9" name="Out" id="e2422830-d8e7-4652-a281-1d841c61e2cd" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Point" lon="3.7326049804687504" lat="52.46252265785238"/>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" technicalLifetime="20.0" length="23801.2" capacity="2000000000.0" id="80a35550-3167-4240-a7ac-92c40248dae7" name="ElectricityCable_80a3">
        <port xsi:type="esdl:InPort" connectedTo="fd1689e6-1339-4b77-8f8f-94acc57c752f" name="In" id="b0ed14ac-faa8-499f-aa98-53be98852d0f" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:OutPort" connectedTo="8f93331f-7785-4213-a6e0-d120a8f0fba7" name="Out" id="62807929-3c55-4545-8279-2f655ab7e943" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="3.381008869842488" lat="52.49203060725069"/>
          <point xsi:type="esdl:Point" lon="3.7298583984375004" lat="52.466050361889515"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" technicalLifetime="20.0" length="13965.2" capacity="2000000000.0" id="91e77ee4-d2d8-4d6c-b655-835d180f4680" name="ElectricityCable_91e7">
        <port xsi:type="esdl:InPort" connectedTo="e2422830-d8e7-4652-a281-1d841c61e2cd" name="In" id="801d733f-16f6-4e5c-a22f-afd204d59776" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:OutPort" connectedTo="d81bb99d-1508-4b13-bc13-1f1a149d481d" name="Out" id="1e76b7b0-9f17-4ccc-b4c3-0089b3ff6a45" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="3.7298583984375004" lat="52.466050361889515"/>
          <point xsi:type="esdl:Point" lon="3.6419677734375004" lat="52.579688026538726"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Joint" id="2503e1c2-e928-4331-bf0b-d41b5f1289fd" name="Joint_2503">
        <port xsi:type="esdl:InPort" connectedTo="73a0c91a-a167-4a3d-b7e3-7603d32db9ac" name="In" id="272d0690-117e-4f69-bd34-ed24b2dc6f0b" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <port xsi:type="esdl:OutPort" connectedTo="f613dd95-9882-45a3-a569-4cd723444c90 1ab8ab85-2824-4606-a2ac-709596d8367b" name="Out" id="c3a78fa8-b482-4bcd-bad2-fd4c971b2565" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Point" lon="3.750457763671875" CRS="WGS84" lat="52.5992941670283"/>
      </asset>
      <asset xsi:type="esdl:Pipe" id="ec1a89cf-c7fb-4574-a93b-048b9d06640a" length="7646.2" innerDiameter="50.0" name="Pipe_ec1a">
        <port xsi:type="esdl:InPort" connectedTo="5e58c0e2-5db4-4f6a-8aab-2c880d499c14" name="In" id="5e660a4f-e8ca-4e9d-8568-6292d36b5994" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <port xsi:type="esdl:OutPort" connectedTo="272d0690-117e-4f69-bd34-ed24b2dc6f0b" name="Out" id="73a0c91a-a167-4a3d-b7e3-7603d32db9ac" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="3.6419677734375004" lat="52.579688026538726"/>
          <point xsi:type="esdl:Point" lon="3.750457763671875" lat="52.5992941670283"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" id="97ee0a46-1a47-4bec-b131-c0a13246f298" length="5839.7" innerDiameter="50.0" name="Pipe_97ee">
        <port xsi:type="esdl:InPort" connectedTo="c3a78fa8-b482-4bcd-bad2-fd4c971b2565" name="In" id="f613dd95-9882-45a3-a569-4cd723444c90" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <port xsi:type="esdl:OutPort" connectedTo="b2bd6049-3e2e-443c-aa36-ba893135cf82" name="Out" id="0e6e6a7d-5b9a-42ff-8902-4e8db530b248" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="3.750457763671875" lat="52.5992941670283"/>
          <point xsi:type="esdl:Point" lon="3.8177490234375004" lat="52.566334145326486"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:Pipe" id="910d3cd0-715a-423c-b67a-52fb967c85b4" length="64730.1" innerDiameter="50.0" name="Pipe_910d">
        <port xsi:type="esdl:InPort" connectedTo="c3a78fa8-b482-4bcd-bad2-fd4c971b2565" name="In" id="1ab8ab85-2824-4606-a2ac-709596d8367b" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <port xsi:type="esdl:OutPort" connectedTo="121a1b8d-758b-4b35-92fa-894435965e85" name="Out" id="f0b7ab61-b983-45ed-9274-887ccaadfce0" carrier="14831e6c-c3bb-4763-8300-9658a365ee54"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="3.750457763671875" lat="52.5992941670283"/>
          <point xsi:type="esdl:Point" lon="4.707641601562501" lat="52.633062890594374"/>
        </geometry>
      </asset>
      <asset xsi:type="esdl:PVInstallation" technicalLifetime="20.0" id="a3ebd136-aa0a-4d45-aa13-ea3f57c9a4a6" surfaceArea="200" name="PVInstallation_a3eb" power="10000000.0">
        <port xsi:type="esdl:OutPort" connectedTo="61c98753-7cbd-4d57-b9bb-5688765fca51" name="Out" id="8aa344a6-4bb5-4c10-aae1-7b2490063add" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Point" lon="4.574604034423829" lat="52.37745564064797"/>
      </asset>
      <asset xsi:type="esdl:Bus" technicalLifetime="20.0" name="Bus_24cf_copy" id="0e13dda6-c792-4f32-891a-6505d31bdd25">
        <port xsi:type="esdl:InPort" id="097a4440-a63d-4b37-aa61-7c5a03f94f0c" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c" name="In"/>
        <port xsi:type="esdl:OutPort" id="732d0a2d-af74-479b-999c-51fa10690ef3" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c" name="Out"/>
        <geometry xsi:type="esdl:Point" lon="3.7298083984375006" CRS="WGS84" lat="52.46600036188951"/>
      </asset>
      <asset xsi:type="esdl:Bus" id="33594e58-1629-4276-9c8e-e91b0b95b4ea" name="Bus_3359">
        <port xsi:type="esdl:InPort" connectedTo="e2422830-d8e7-4652-a281-1d841c61e2cd" name="In" id="949dd468-5c21-4a52-9596-cb4a50a113a9" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:OutPort" connectedTo="ac3fc355-1545-4b78-a46c-1b0908f5ccde" name="Out" id="55709cf7-8ad7-4bc0-9f3d-748c0788a398" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:InPort" connectedTo="1c5c65c9-3723-4dc8-b334-33c1234ba620" name="Port" id="96bbe1e5-877f-4269-8e0c-305e1fbbdab1" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Point" lon="4.585590362548829" lat="52.38694132055252" CRS="WGS84"/>
      </asset>
      <asset xsi:type="esdl:ElectricityCable" length="1291.7" id="92da7339-070c-44be-8127-e0a684b4a303" name="ElectricityCable_92da">
        <port xsi:type="esdl:InPort" connectedTo="8aa344a6-4bb5-4c10-aae1-7b2490063add" name="In" id="61c98753-7cbd-4d57-b9bb-5688765fca51" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <port xsi:type="esdl:OutPort" connectedTo="96bbe1e5-877f-4269-8e0c-305e1fbbdab1" name="Out" id="1c5c65c9-3723-4dc8-b334-33c1234ba620" carrier="e1077d81-b32a-4004-9a33-db65c96b5f4c"/>
        <geometry xsi:type="esdl:Line" CRS="WGS84">
          <point xsi:type="esdl:Point" lon="4.574604034423829" lat="52.37745564064797"/>
          <point xsi:type="esdl:Point" lon="4.585590362548829" lat="52.38694132055252"/>
        </geometry>
      </asset>
    </area>
  </instance>
</esdl:EnergySystem>
