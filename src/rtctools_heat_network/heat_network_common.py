from enum import IntEnum


class PipeFlowDirection(IntEnum):
    """
    Enumeration for the possible directions a pipe can have.
    """

    NEGATIVE = -1
    DISABLED = 0
    POSITIVE = 1


class NodeConnectionDirection(IntEnum):
    """
    Enumeration for the orientation of a pipe connected to a node, or of the
    flow into a node.
    """

    OUT = -1
    IN = 1
