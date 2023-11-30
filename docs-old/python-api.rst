Python API
==========

Public API
----------

HeatMixin
^^^^^^^^^

.. autoclass:: rtctools_heat_network.heat_mixin.HeatMixin
    :members: heat_network_options
    :show-inheritance:


QTHMixin
^^^^^^^^

.. autoclass:: rtctools_heat_network.qth_mixin.QTHMixin
    :members: heat_network_options, heat_network_flow_directions
    :special-members: __init__
    :show-inheritance:


HeadLossMixin
^^^^^^^^^^^^^

.. autoclass:: rtctools_heat_network.head_loss_mixin.HeadLossOption
    :show-inheritance:


Internal API
------------

HeadLossMixin
^^^^^^^^^^^^^

.. autoclass:: rtctools_heat_network.head_loss_mixin._HeadLossMixin
    :members: heat_network_options, _hn_pipe_head_loss_constraints, _hn_get_pipe_head_loss_option, _hn_minimization_goal_class, _hn_pipe_head_loss
    :show-inheritance:

.. automodule:: rtctools_heat_network.head_loss_mixin
    :members: _MinimizeHeadLosses
    :show-inheritance:
