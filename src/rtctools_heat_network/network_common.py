from strenum import StrEnum


class NetworkSettings(StrEnum):
    """
    Enumeration for network settings.
    """

    NETWORK_TYPE_GAS = "Gas"  # Natural gas
    NETWORK_TYPE_HYDROGEN = "Hydrogen"
    NETWORK_TYPE_HEAT = "Heat"
    NETWORK_TYPE_ELECTRICITY = "Electricity"
