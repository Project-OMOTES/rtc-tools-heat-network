# rtc-tools-heat-network

Rtc-tools-heat-network is an optimization application for optimal planning, design and 
operation of Energy Systems with the current main focus on District Heating Systems (DHS). The current application focuses on a Mixed Integer Linear Problem (MILP) approach, with multiple linearization strategies to conservatively approximate the steady-state physics and financial models.
All physics are combined in the HeatMixin class. When inherited this class can be combined with objective functions (that typically incorporate the financial aspects) and interface methods to create an
optimization workflow (see also running an example).

The main supported method for defining your Energy system is ESDL (Energy System Description Language), which is a modelling language for energy systems. See also:https://github.com/EnergyTransition/ESDL.
With ESDL you can define assets like demands, sources, pipes, etc. and fill in their attributes. The ESDLMixin class
will parse the ESDL file and utilize the attributes to build up the model representation.

This optimization package was originally developed for operational optimization and hosts two 
optimization approaches 1) A MILP approach 
and 2) Nonlinear Problem (NLP) approach. These two approaches were developed to run sequentially for 
operational optimization. the MILP would fix the integer decision for the NLP problem, such that only the continuous variables need to be solved. The NLP
problem would then find the optimized solution with the steady-state non-linear physics included. The existing outdated (still to be updated / update in progress) documentation can be found on: http://warmingup.pages.ci.tno.nl/rtc-tools-heat-network/

Installation
============

Installation of the RTC-Tools Heat Network library is as simple as::

    # 1a. Use pip to install directly
    pip install rtc-tools-heat-network

If you are going to develop and change the source code, you probably want to do something like::

	# 1b. Use git clone and pip to make an editable/developer installation
	git clone https://ci.tno.nl/gitlab/warmingup/rtc-tools-heat-network
	pip install -e rtc-tools-heat-network

RTC-Tools Heat Network depends on `RTC-Tools <https://gitlab.com/deltares/rtc-tools.git>`_, which is automatically installed as one of its dependencies.

Running an example
==================

To make sure that everything is set-up correctly, you can run one of the example cases.
These do not come with the installation, and need to be downloaded separately::

    # 1. Clone the repository
    git clone https://github.com/Nieuwe-Warmte-Nu/rtc-tools-heat-network.git

    # 2. Change directory to the example folder
    cd rtc-tools-heat-network/examples/pipe_diameter_sizing/src

    # 3. Run the example
    python example.py

You will see the progress of RTC-Tools in your shell.
If all is well, you should see something like the following output:

![img.png](img.png)

In this example.py file you can see a small workflow being set-up. The PipeDiameterSizingProblem class
inherits from (Note only the *classes are defined in rtc-tools-heat-network the others come from rtc-tools package):
- CollocatedIntegratedOptimizationProblem: This class does all the discretization of the state variables in your problem.
- *ESDLMixin: This class does the parsing and setting up of a model based on an ESDL file.
- GoalProgrammingMixin: This class allows you to add Goals (objective functions) with different priorities.
- LinearizedOrderGoalProgrammingMixin: This class allows you to add higher order goals (e.g. order=2) for MILP problems.
- *HeatMixin: This class adds all the heat network physics for MILP problems. 

Within the PipeDiameterSizingProblem class you can see that the path_goals() function is overwritten and that
a path_goal with priority one is added to meet the heat demands. The definition path_goal is used
to define a goal that is applied to a state variable at every time step. Furthermore, the goals() method is also overwritten
in this case where an objective with priority two is added to minimize `length*diameter`.
The goals() method is used here for global variables that do not change over time. The priorities indicate the sequential order 
in which the optimizer would be applied to the goals. In this example the heat demand is matched first, after which priority 2 `length*diameter` 
is minimized. In this example the objective of the priority one goal constraints the priority two goal optimization, which ensures that the 
optimization of the priority two goal does not have impact on the optimal result of the priority one goal.

Contribute
==========

You can contribute to this code through Pull Request on GitHub.
Please, make sure that your code is coming with unit tests to ensure full coverage and continuous integration in the API.

GitHub: https://github.com/Nieuwe-Warmte-Nu/rtc-tools-heat-network



## Release
This package is released on pypi [here](https://pypi.org/project/rtc-tools-heat-network/) whenever a new tag is pushed.
In order to release this package:

1. Make sure that all relevant merge requests and commits have been merged to the master and/or poc-release branch.
2. Run `git checkout master` or `git checkout poc-release` to switch to the release branch.
3. Run `git pull origin master` or `git pull origin poc-release` to pull all latest changes.
4. Run `git tag <new_version>` where `<new_version>` is the new version number.
5. Run `git push origin <new_version>` to push the tag to Github.
6. Check [Github](https://github.com/Nieuwe-Warmte-Nu/rtc-tools-heat-network/actions) to confirm the release is
   processed without errors.
7. Once the release has finished, confirm the new version is available on [pypi](https://pypi.org/project/rtc-tools-heat-network/).
