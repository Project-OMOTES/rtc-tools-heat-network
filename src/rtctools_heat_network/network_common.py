from strenum import StrEnum


class NetworkSettings(StrEnum):
    """
    Enumeration for network settings.
    """

    NETWORK_TYPE_GAS = "Gas"
    NETWORK_TYPE_HEAT = "Heat"
    NETWORK_TYPE_ELECTRICITY = "Electricity"
