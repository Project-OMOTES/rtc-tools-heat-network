Introduction
============

The purpose of the RTC-Tools Heat Network library is to provide a framework for design and operational optimization of heat networks.
It provides automatic construction of your heat network model based on a ESDL file.
It is of course also possible to use a PyCML or Modelica model.

For the optimization one needs to define a model of your network and objective that you want to attain.
A common way to formulate these objectives is to use hierarchical "goals", see the `RTC-Tools documentation <https://rtc-tools.readthedocs.io/en/stable/examples/optimization/goal_programming.html>`_ for more information on this.
See also the RTC-Tools Heat Network examples for some ideas on goals related to heat network optimization.

This documentation will elaborate on the motivation of the project, high level approach, how to construct a network using ESDL/Modelica and the modeling aspects.

Motivation
----------

The motivation for the design of heat networks can more adequately explained by the WarmingUP project, which funded the development of this framework (English page pending).
As part of the design and operation of heat networks, many components can be identified:

Design phase
""""""""""""

In the design phase, it is important to figure out the ideal topology of the network.
In other words, what pipes or other components go where, how big should they be, etc.
It is also required to know more-or-less how the designed system would operate for certain design conditions, e.g. if it is able to satisfy the predicted demand an a cost-effective manner.
The latter is where this model predictive control library comes into play.

Operational phase
"""""""""""""""""

In the operational setting one can imagine the control of a buffer as an archetypal example.
For short-term buffers, it is important to know when to store heat (e.g. during the night) and how much.
This depends, amongst others, on the predicted demand by consumers and the predicted heat production costs.
The predicted demand itself is often a function of the predicted weather conditions.

Constructing a network with ESDL/Modelica
------------------------------------------

You can either decide to use a Modelica or ESDL file description of a network, although the latter is strongly recommended.
Most importantly these files contain which components are in the network and how they are connected.
Some properties of the components can also be read from the file.

ESDL (Energy System Description Language) allows to describe your energy network in the conceptual phases.
You can draw networks in the `MapEditor <https://mapeditor-beta.hesi.energy>`_ and save the ESDL file.
This allows for an user friendly way of drawing networks and setting component properties.

Modelica models are the basis of many RTC-Tools models.
For more information on how to construct Modelica models, we refer to the RTC-Tools documentation.

.. note::

    The Modelica/PyCML component library is not documented in a way that works well with Sphinx yet.
    We generally recommend using ESDL models, but if Modelica/PyCML models are desired, the .mo/.py files are self-descriptive enough to construct such a model.
