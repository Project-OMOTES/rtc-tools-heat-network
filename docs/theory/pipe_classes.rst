.. _chp_logstor:

Pipe insulation series
======================

Often district heating pipes are pre-insulated rigid steel pipes, which can have different insulation classes.

The optimizer routine utilizes information on the pipe properties from the `ESDL Energy Data Repository (EDR) <https://edr.hesi.energy/cat/Assets>`_.
This repository contains pipe properties for bonded pipe systems of district heating pipes
with different insulation series from the `Logstor catalog <https://www.logstor.com/media/6506/product-catalogue-uk-202003.pdf>`_.
It contains the following properties for several DN sizes:

* **name**: Type of material - insulation class - DN size.
* **inner_diameter**: inner diameter of the pipe [m].
* **u_1**: thermal transmittance (i.e., U-value) 1 [Wm\ :sup:`-2` K\ :sup:`-1`].
* **u_2**: thermal transmittance (i.e., U-value) 2 [Wm\ :sup:`-2` K\ :sup:`-1`].
* **insulation_thicknesses**: ordered list (from inner layer outwards) of thicknesses of the insulation layers in [m].
* **conductivities_insulation**: ordered list (from inner layer outwards) of conductivity in [W/mK].
* **investment_costs**: combined installation and investment cost [€/m].

For all insulation classes and pipe diameters, the conductivities are 52.15 W/mK, 0.027 W/mK and 0.4 W/mK, for the steel, PUR and PE layers, respectively.

:numref:`table_Steel_S1`, :numref:`table_Steel_S2` and :numref:`table_Steel_S3` show the available properties for insulation classes S1, S2 and S3, respectively.
Note that currently, the optimizer routine only supports the Steel-S1 class. 

.. _table_Steel_S1:

.. table:: Steel S1 properties

    +------------------+----------------+--------+---------+------------------------+------------------+
    | name             | inner_diameter | u_1    | u_2     | thicknesses            | investment_costs |
    +==================+================+========+=========+========================+==================+
    | Steel-S1-DN-20   | 0.0217         | 0.1426 | 0.00364 | 0.0026, 0.02855, 0.003 | 696.3            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-25   | 0.0285         | 0.1760 | 0.00554 | 0.0026, 0.02515, 0.003 | 709.3            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-32   | 0.0372         | 0.1795 | 0.00534 | 0.0026, 0.0308, 0.003  | 727.9            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-40   | 0.0431         | 0.2083 | 0.00719 | 0.0026, 0.02785, 0.003 | 749.7            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-50   | 0.0545         | 0.2339 | 0.00861 | 0.0029, 0.02935, 0.003 | 778.0            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-65   | 0.0703         | 0.2783 | 0.01162 | 0.0029, 0.02895, 0.003 | 822.4            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-80   | 0.0825         | 0.2869 | 0.01164 | 0.0032, 0.03255, 0.003 | 869.4            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-100  | 0.1071         | 0.2996 | 0.01144 | 0.0036, 0.03965, 0.0032| 936.1            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-125  | 0.1325         | 0.3507 | 0.01478 | 0.0036, 0.03925, 0.0034| 1026.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-150  | 0.1603         | 0.4210 | 0.02016 | 0.004, 0.03725, 0.0036 | 1126.4           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-200  | 0.2101         | 0.4576 | 0.02098 |0.0045, 0.004385, 0.0041| 1355.3           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-250  | 0.263          | 0.4357 | 0.01650 | 0.005, 0.0587, 0.0048  | 1630.7           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-300  | 0.3127         | 0.5051 | 0.02057 | 0.0056, 0.05785, 0.0052| 1962.1           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-350  | 0.3444         | 0.4880 | 0.01793 | 0.0056, 0.0666, 0.0056 | 2360.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-400  | 0.3938         | 0.5167 | 0.01861 | 0.0063, 0.0711, 0.0057 | 2840.6           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-450  | 0.4444         | 0.5157 | 0.01706 | 0.0063, 0.0805, 0.006  | 3417.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-500  | 0.4954         | 0.4959 | 0.01447 | 0.0063, 0.0944, 0.0066 | 4112.5           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-600  | 0.5958         | 0.6134 | 0.02026 | 0.0071, 0.0872, 0.0078 | 5953.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-700  | 0.695          | 0.7060 | 0.02455 | 0.008, 0.0858, 0.0087  | 8619.6           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-800  | 0.7954         | 0.8032 | 0.02928 | 0.0088, 0.0841, 0.0094 | 12479.0          |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-900  | 0.894          | 0.8980 | 0.03399 | 0.01, 0.0828, 0.0102   | 18066.3          |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-1000 | 0.994          | 1.0000 | 0.03936 | 0.011, 0.081, 0.0011   | 26155.2          |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-1100 | 1.096          | 1.1045 | 0.04510 | 0.011, 0.0792, 0.0118  | 37865.8          |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S1-DN-1200 | 1.194          | 1.2036 | 0.05052 | 0.0125, 0.078, 0.0125  | 54819.7          |
    +------------------+----------------+--------+---------+------------------------+------------------+


.. _table_Steel_S2:

.. table:: Steel S2 properties

    +------------------+----------------+--------+---------+------------------------+------------------+
    | name             | inner_diameter | u_1    | u_2     | thicknesses            | investment_costs |
    +==================+================+========+=========+========================+==================+
    | Steel-S2-DN-20   | 0.0217         | 0.1211 | 0.00244 | 0.0026, 0.03855, 0.003 | 696.3            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-25   | 0.0285         | 0.1444 | 0.00346 | 0.0026, 0.03515, 0.003 | 709.3            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-32   | 0.0372         | 0.1574 | 0.00390 | 0.0026, 0.0383, 0.003  | 727.9            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-40   | 0.0431         | 0.1791 | 0.00505 | 0.0026, 0.03535, 0.003 | 749.7            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-50   | 0.0545         | 0.2014 | 0.00609 | 0.0029, 0.03685, 0.003 | 778.0            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-65   | 0.0703         | 0.2271 | 0.00730 | 0.0029, 0.03895, 0.003 | 822.4            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-80   | 0.0825         | 0.2382 | 0.00761 | 0.0032, 0.04255, 0.003 | 869.4            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-100  | 0.1071         | 0.2478 | 0.00738 | 0.0036, 0.05195, 0.0034| 936.1            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-125  | 0.1325         | 0.2878 | 0.00943 | 0.0036, 0.05155, 0.0036| 1026.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-150  | 0.1603         | 0.3286 | 0.01156 | 0.004, 0.05195, 0.0039 | 1126.4           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-200  | 0.2101         | 0.3463 | 0.01121 |0.0045, 0.06345, 0.0045 | 1355.3           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-250  | 0.263          | 0.3346 | 0.00904 | 0.005, 0.0833, 0.0052  | 1630.7           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-300  | 0.3127         | 0.3846 | 0.01114 | 0.0056, 0.08245, 0.0056| 1962.1           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-350  | 0.3444         | 0.3681 | 0.00945 | 0.0056, 0.0962, 0.006  | 2360.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-400  | 0.3938         | 0.3816 | 0.00935 | 0.0063, 0.1052, 0.0066 | 2840.6           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-450  | 0.4444         | 0.3801 | 0.00850 | 0.0063, 0.1193, 0.0072 | 3417.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-500  | 0.4954         | 0.4310 | 0.00734 | 0.0063, 0.1381, 0.0079 | 4112.5           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S2-DN-600  | 0.5958         | 0.6134 | 0.00915 | 0.0071, 0.1363, 0.0087 | 5953.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+


.. _table_Steel_S3:

.. table:: Steel S3 properties

    +------------------+----------------+--------+---------+------------------------+------------------+
    | name             | inner_diameter | u_1    | u_2     | thicknesses            | investment_costs |
    +==================+================+========+=========+========================+==================+
    | Steel-S3-DN-20   | 0.0217         | 0.1107 | 0.00192 | 0.0026, 0.04605, 0.003 | 696.3            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-25   | 0.0285         | 0.1298 | 0.00265 | 0.0026, 0.04265, 0.003 | 709.3            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-32   | 0.0372         | 0.1419 | 0.00303 | 0.0026, 0.0458, 0.003  | 727.9            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-40   | 0.0431         | 0.1594 | 0.00381 | 0.0026, 0.04285, 0.003 | 749.7            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-50   | 0.0545         | 0.1731 | 0.00424 | 0.0029, 0.04685, 0.003 | 778.0            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-65   | 0.0703         | 0.1955 | 0.00513 | 0.0029, 0.04895, 0.003 | 822.4            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-80   | 0.0825         | 0.2075 | 0.00549 | 0.0032, 0.05235, 0.003 | 869.4            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-100  | 0.1071         | 0.2146 | 0.00525 | 0.0036, 0.06425, 0.0036| 936.1            |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-125  | 0.1325         | 0.2414 | 0.00624 | 0.0036, 0.06625, 0.0039| 1026.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-150  | 0.1603         | 0.2673 | 0.00717 | 0.004, 0.06925, 0.0041 | 1126.4           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-200  | 0.2101         | 0.2783 | 0.00674 |0.0045, 0.08565, 0.0048 | 1355.3           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-250  | 0.263          | 0.2771 | 0.00579 | 0.005, 0.1079, 0.0056  | 1630.7           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-300  | 0.3127         | 0.3060 | 0.00653 | 0.0056, 0.11205, 0.006 | 1962.1           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-350  | 0.3444         | 0.2934 | 0.00553 | 0.0056, 0.1306, 0.0066 | 2360.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-400  | 0.3938         | 0.3009 | 0.00533 | 0.0063, 0.1446, 0.0072 | 2840.6           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-450  | 0.4444         | 0.3001 | 0.00486 | 0.0063, 0.1636, 0.0079 | 3417.9           |
    +------------------+----------------+--------+---------+------------------------+------------------+
    | Steel-S3-DN-500  | 0.4954         | 0.2942 | 0.00426 | 0.0063, 0.1873, 0.0087 | 4112.5           |
    +------------------+----------------+--------+---------+------------------------+------------------+