Heat loss
=========

Three types of heatlosses in the pipeline are considered, in accordance to the `NEN-EN 13941+A1 <https://www.nen.nl/nen-en-13941-1-2019-a1-2022-en-290896>`_.
These are heat loss through:

- Tube wall
- Subsoil
- Though neighboring pipeline

Visualized schematically, these are heat losses/fluxes as follows:

.. image:: /images/pipeline_heatloss.png


If we write it in a set of equations (which can be formulated as constraints), we get the following equation for the temperature loss inside a pipe from its in- to its outport:

.. math::
    :label: heatloss_pipe_balance

    {\left( {{c_p}\dot mT} \right)_{in}} - {\left( {{c_p}\dot mT} \right)_{out}} = {Q_{loss}}

In which the heat loss :math:`Q_{loss}` is equal to:

.. math::
    :label: heatloss_pipe_qloss

    {Q_{loss}} = L\left( {{U_1} - {U_2}} \right)\left( {{T_h} - {T_g}} \right) + L{U_2}\left( {{T_h} - {T_c}} \right)


In which:

    :math:`L`: Length of pipeline [m]

    :math:`T_h`: Temperature in hot (feed) pipeline [K]

    :math:`T_c`: Temperature in cold (return) pipeline [K]

    :math:`T_g`: Temperature at ground temperature [K]

The values for :math:`U_1` and :math:`U_2` follow from the following set of equations:

.. math::
    :label: heatloss_pipe_u1

    {U_1} = \frac{{{R_g} + {R_{iso}}}}{{{{\left( {{R_g} + {R_{iso}}} \right)}^2} - R_m^2}}

.. math::
    :label: heatloss_pipe_u2

    {U_2} = \frac{{{R_m}}}{{{{\left( {{R_g} + {R_{iso}}} \right)}^2} - R_m^2}}


In which:

    :math:`R_g`: Subsoil heat resistance [mK/W]

    :math:`R_{iso}`: Insulation heat resistance [mK/W]

    :math:`R_m`: Heat resistance due to neighboring pipeline [mK/W]

As the description above shows, :math:`U_1` and :math:`U_2` are constant values based on type, placement en dimensions of the pipelines.

For the MILP formulation of :py:class:`.HeatMixin`, the hot and cold temperature lines are fixed.
The heat loss of a pipe in the MILP formulation is therefore not dependent on the flow rate.
For the NLP formulation of :py:class:`.QTHMixin`, the temperature of a pipe is the average of its in- and outgoing temperatures.
This means that the heat loss is dependent on flow rate.
