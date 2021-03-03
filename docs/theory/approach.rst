Approach
========

Rationale
---------

For operational optimization high priority is given to a stable, robust, and fast model over a perfectly accurate one.
This is not to say that the model is inaccurate, but for model predictive control purposes this is a trade-off.
In model predictive control we often deal with uncertainties of future predictions, e.g. weather forecasts or heat demand profiles.
These uncertainties can easily be up to 10%.
Therefore it is often desirable to slightly reduce the modeling accuracy to gain more stable and fast outcomes.

The heat network optimization problem, if naively formulated, would be of the MINLP (mixed-integer non-linear programming) type.
The non-linearities in this naive formulation primarily come from the temperature and head loss equations.
The integer/discrete part follows from the temperature mixing at nodes, and other requirements like minimum power production.
This class of problems is notoriously hard to optimize, and "good" solvers tend to commercial.
Even with commercial solvers, there is are generally no guarantee of finding *any* solution within a reasonable amount of time, let alone one that is close to optimality.
Knowing how far a solution is from optimality is also harder for MINLP than it is for e.g. a MILP (mixed-integer linear programming).

To get around this issue, we split the original/naive MINLP problem into two pieces: a MILP one, and a NLP one:

- The (MI)LP has as decision variables: the discharge, the head, and most notably the **heat flow rate**.
  By using the heat flow rate as a variable instead of the temperature, we can get rid of the non-linear mixing constraints.
- The NLP that has as decision variables: the discharge, the head, and most notably the **temperature**.
  The NLP has fixed flow directions (following from the MILP).

What follows is a discussion of how the both of these problems are set up.
This split out into three parts:

 - Head loss formulation (mostly shared by the MILP and NLP problems)
 - Heat flow rate formulation (MILP)
 - Temperature (mixing) formulation (NLP)

For a detailed view into the API of their respective classes, see :py:class:`._HeadLossMixin`, :py:class:`.HeatMixin` and :py:class:`.QTHMixin`.

.. _sec_head_loss_formulation:


Heads and discharges
--------------------

Shared between the NLP and the MILP formulations is a relationship between the discharge and the head.
There are some obvious relationship for these variables, e.g. continuity and state.


Continuity equations
""""""""""""""""""""

At every connection between components, or at every node when connecting multiple components, we can say that:

.. math::
    :label: headloss_continuity_discharge

    \sum Q_{in} = \sum Q_{out}

.. math::
    :label: headloss_continuity_head

    H_{in} = H_{out} = H_{node}


Pumps (and Sources)
"""""""""""""""""""

Pumps (whether incorporated as a Source or not) can add head to the system.

.. math::
    :label: headloss_pump

    \left| \Delta H_{pump} \right| \ge 0


Demands
"""""""

Demands typically have a minimum desired pressure drop of about 10 m to have enough control range.
If there are many demands in the system, the one with the lowest pressure drop will be 10 m.
The other demands will have a pressure drop higher than that.

.. math::
    :label: headloss_demand

    \left| \Delta H \right| \ge 10


Pipes and resistances
"""""""""""""""""""""

Assuming steady state operation, one can say that the relationship for the head loss over a pipe and its discharge is approximately:

.. math::
    :label: headloss_pipe_approx_quadratic_eq

    \left| \Delta H \right| \approx C \cdot Q ^ 2

See also :py:class:`._HeadLossMixin`, and :py:class:`.HeadLossOption`.

This equation is non-convex, which would lead to loss in guarantee of global optimality.
It also tends to be harder to compute, and most free and commercial (MI)LP solvers cannot solve this type of equation.

In many cases, it is sufficient for us to give a solution that is feasible.
In that case, as long as the *exact* discharge distribution in the network does not matter, and provided every consumer/producer has the ability to throttle flow, we can get the same effective answer in a convex way by reformulating as an inequality:

.. math::
    :label: headloss_pipe_approx_quadratic_ineq

    \left| \Delta H \right| \ge C \cdot Q ^ 2

There are many scenarios in which the discharge distribution *does* matter, e.g. if the heat loss or temperature loss is dependent on the flow rate.
In this case, using the above (convex) equation might still lead to the correct solution, if there is enough freedom *inside* the network to realize this solution.
In practice, this means that for a network which has e.g. two parallel pipes, one of the pipes has a control valve or similar capabilities.

If one wants to stay fully linear (e.g. because the solver does not support quadratic inequality constraints, or is much slower when solving a problem with them), an approximation can be used.
The inequality head loss relationship can then be linearized over a known domain.
In addition, one can choose to use the Darcy-Weisbach formula to make these linear constraints, which is more slighthly accurate than the quadratic approach is.

.. math::
    :label: headloss_pipe_approx_linear_ineq

    \left| \Delta H \right| \ge \vec{a} \cdot Q + \vec{b}

with :math:`\vec{a}` and :math:`\vec{b}` the linearization coefficients.

Simplifying even further, one can use an estimated velocity, and linearize only once.
For example, if we assume that most pipe velocities in are in the order of 1 m/s, we can say that (note the lack of the square):

.. math::
    :label: headloss_pipe_approx_linear_eq

    \left| \Delta H \right| \approx C \cdot Q

.. note::

    The :math:`C` in the equation above is not equal to the :math:`C` in :eq:`headloss_pipe_approx_quadratic_eq`.
    It is only the same if the linearization velocity/discharge is 1.0 :math:`m3/s`

We use the linear formulation in the :py:class:`.HeatMixin` by default, because it is faster (fewer constraints) in giving us the correct directions.
Used on its own, and provided the constraints earlier are satisfied (enough control freedom in the system), it would be better to use the (possibly linearized) inequality formulation.

For the :py:class:`.QTHMixin`, the quadratic inequality formalation is used.
This generally leads to a lot fewer convergence issues compared to an equality formulation, as the problem is easier to solve.
If there is not enough control freedom, warnings are raised.
If so desired, the user can then choose to use the :py:class:`QTHLoopMixin` instead.
The latter does have the limitation of only solving for a single time step at a time, for stability reasons.
To let MPC shine, it then only makes sense to use this mixin in conjunction with the HeatMixin to determine the flows to the buffers in the system.

For more details on what options are available, and what set of constraints they result in, see :py:class:`.HeadLossOption`


Heat flow rate formulation
--------------------------

The heat flow rate formulation (also "heat problem" for short) uses three decision variables, most notably the heat flow rate:

:math:`H`: the head in the system [m]

:math:`Q`: the discharge [:math:`m^3/s`]

:math:`P`: the heat flow rate [:math:`W`]

:math:`Q` and :math:`H` are included to ensure that the solutions found are hydraulically feasible.
The relationship between :math:`Q` and :math:`H` has been explained in :ref:`sec_head_loss_formulation`.
What remains is equations related to the heat flow rate, including those that relate heat flow rate to the discharge.

First though, we have to decide what the heat flow rate represents.
Using temperature as an absolute value, we *could* say that

.. math::
    :label: heat_absheat

    P = \rho c_p Q T

With temperatures typically in the order of 10 - 100 °C, using the absolute heat like this would lead to a very large offset.
For optimization purposes, we generally want to have variables that are (or can be) scaled to be in the range [0 - 1].
In addition, when using the formulation above, there is still a need to then figure out what the *temperature* at various locations in the system is.
To overcome these shortcomings, we can instead use a *relative* formulation where we force the heat flow rate to be zero at some point in the system.
For example, we can force the heat flow rate to be zero on the cold side, i.e. upstream of all sources and dowstream of all demands.

In the heat problem, a fixed temperature difference :math:`dT` is assumed.
Put differently, for all hydraulically connected pipes on one side of the producers and demands we assume a certain "hot" temperature, and for all hydraulically connected pipes on the other side we assume a certain "cold" temperature.
The place where we pick the values for this "dT" is at the demands.
We can therefore say that

.. math::
    :label: heat_relheat_demand_dt

    \Delta T = T_{demand,in} - T_{demand,out}

.. math::
    :label: heat_relheat_demand

    P_{demand,in} = \rho c_p Q \Delta T

    P_{demand,out} = 0

By doing this, and if we incorporate heat losses, we can however not say that

.. math::
    :label: heat_relheat_source_wrong

    P_{source,out} = \rho c_p Q \Delta T \text{     (=invalid)}

With heat losses in the system, the sources need to produce strictly more heat than is consumed by the demands.
The equation relating the heat flow rate and discharge then resolves to:

.. math::
    :label: heat_relheat_source

    P_{source,out} \ge \rho c_p Q \Delta T

    P_{source,in} = 0

Without specifying anything for the pipes, this could still lead to infeasible solutions.
For example, an infeasible solution that we want to avoid is one in which the heat flow rate through a pipe with zero flow is much larger than its heat loss.
We would *want* to find the solution where the heat flow rate to one of these pipes is *exactly* equal to its heat loss.
Assuming positive flow:

.. math::
    :label: heat_relheat_pipe

    P_{pipe,in} \ge \rho c_p Q \Delta T

    P_{pipe,out} \ge \rho c_p Q \Delta T


Temperature (mixing) formulation
--------------------------------

The temperature mixing formulation (also "qth problem" for short) uses three decision variables, most notably the temperature:

:math:`H`: the head in the system [m]

:math:`Q`: the discharge [:math:`m^3/s`]

:math:`T`: the temperature [:math:`°C`]

:math:`Q` and :math:`H` are included to ensure that the solutions found are hydraulically feasible.
The relationship between :math:`Q` and :math:`H` has been explained in :ref:`sec_head_loss_formulation`.
What remains is equations related to the temperature.

For sources and demands we can then simply state that

.. math::
    :label: qth_eq_source

    \rho c_p Q T_{source,out} = \rho c_p Q T_{source,out} + P_{source}

.. math::
    :label: qth_eq_demand

    \rho c_p Q T_{demand,out} = \rho c_p Q T_{demand,in} - P_{demand}

with :math:`P_{source}` and :math:`P_{demand}` the heat production of the source and heat consumption of the demand respectively (in Watts).

For the temperature mixing at nodes, the following equation holds for the temperature of outgoing flows

.. math::
    :label: qth_node_tout

    T_{out} = \frac{\sum_{i \in inflows} Q_i T_i}{\sum_{i \in inflows} Q_i}
