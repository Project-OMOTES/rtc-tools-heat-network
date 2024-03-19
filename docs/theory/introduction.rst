Introduction
============

The purpose of the optimizer routine is to provide a framework for design and operational optimization of heat networks.
It provides automatic construction of your heat network model based on a ESDL file.

For the optimization one needs to define a model of your network and objective that you want to attain.
A common way to formulate these objectives is to use hierarchical "goals", see the `RTC-Tools documentation <https://rtc-tools.readthedocs.io/en/stable/examples/optimization/goal_programming.html>`_ for more information on this.
See also the RTC-Tools Heat Network examples for some ideas on goals related to heat network optimization.

This documentation will elaborate on the ins and outs of the optimizer routine. 
The routine depends on `RTC-Tools <https://gitlab.com/deltares/rtc-tools.git>`_, which is automatically installed as one of its dependencies.

Constructing a network with ESDL
--------------------------------

You can use an ESDL file description of a network.
Most importantly these files contain which components are in the network and how they are connected.
Some properties of the components can also be read from the file.

ESDL (Energy System Description Language) allows to describe your energy network in the conceptual phases.
You can draw networks in the `MapEditor <https://mapeditor-beta.hesi.energy>`_ and save the ESDL file.
This allows for an user friendly way of drawing networks and setting component properties.
