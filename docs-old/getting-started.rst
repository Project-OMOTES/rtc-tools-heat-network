###############
Getting Started
###############

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

The Modelica library is installed in a hard to access location to make sure RTC-Tools can find it.
If you want to load the library in an editor like OpenModelica, it is best to run `rtc-tools-copy-libraries`.
See also :ref:`the RTC-Tools documentation <rtctools:getting-started-copy-libraries>` on this.

Running an example
==================

To make sure that everything is set-up correctly, you can run one of the example cases.
These do not come with the installation, and need to be downloaded separately::

    # 1. Clone the repository
    git clone https://ci.tno.nl/gitlab/warmingup/rtc-tools-heat-network

    # 2. Change directory to the example folder
    cd rtc-tools-heat-network/examples/pipe_diameter_sizing/src

    # 3. Run the example
    python example.py

You will see the progress of RTC-Tools in your shell.
If all is well, you should see something like the following output.

.. image:: images/example-output.png

Contribute
==========

You can contribute to this code through Pull Request on GitLab_.
Please, make sure that your code is coming with unit tests to ensure full coverage and continuous integration in the API.

.. _GitLab: https://ci.tno.nl/gitlab/warmingup/rtc-tools-heat-network/merge_requests
