Introduction
============

MESIDO is the abbreviation for Multi Energy System Integrated Design and Operation. This documentation is meant to give the user of MESIDO package insight in how to use the different functionalities available.

Why MESIDO?
-----------

MESIDO aims to fill the gap between available tools used in early feasibility studies and the more detailed tools (typically simulators).
Tools for feasibilty studies like EnergyPlan, https://www.energyplan.eu, provide a good framework for initial feasibility of high level system concepts.
These tools often have little to no physics included in their models, typically energy balancing with efficiencies and capacity limits.
On the other hand there are more detailed tools, examples include detailed simulators like PandaPower and PandaPipes.
These simulators include more detailed modelling of physics allowing to analyze networks and/or assets close to real life operational conditions.

The high level system concept that comes out of the initial feasibility is typically still subjected to many uncertainties. Therefore, many design choices are still to be made like placement and exact sizing of assets, optimal routing of the network, etc.
Although, the more detailed simulators can analyze various energy system designs by running different design options, it is seen that this becomes infeasible in practice as the amount of open design choices after initial feasibility is too high.
The increased complexity of the new and future energy systems plays a significant part. New energy systems have multiple producer, multiple consumer networks, often with storage (possibly seasonal)
and most notably interconnections between different energy commodities means that possible solution space is larger than ever.

MESIDO aims to play the connecting role between the initial feasibility and the detailed analyses, by offering a techno-economic optimization framework that allows to make the design choices and come to a (limited set of) energy system design(s) that can be detailed out by the simulators.

What is MESIDO?
---------------

MESIDO is a python package that allows users to define techno-economic optimization workflows (or use the available ones). Users can define themselves which KPIs they wish to optimize for, e.g. cost or business case optimization.
Furthermore, the user can select a physics fidelity level he/she needs for the analyses. Where the physics are typically approximations of steady-state conditions, the modelling is done such that the solutions are close and conservative.
The inclusion of the physics approximations allows the user to make more design choices compared to the tools used in initial feasibility.
However, this comes at the cost of the user needing to provide more information of the energy system in order to use MESIDO in an effective way, hence MESIDO is proposed as a sequential tool for analysis.

Under the hood MESIDO is a Mixed Integer Linear Problem (MILP) formulation. This means that the physics and financial models are limited to linear constraints.
The computational power advantages of MILP optimizations allows MESIDO to optimize Multi Energy/Multi Commodity networks in an integral manner.
Nevertheless, the modelling limitations that MILP have will in the fast majority of cases require optimized networks to be detailed out with simulation tools before closing business cases and going to exploitation.







