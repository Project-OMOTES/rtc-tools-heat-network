"""Heat network models for RTC-Tools 2.

Includes Modelica models and their accompanying Mixins for heat networks.
"""
import sys

from setuptools import find_packages, setup

import versioneer

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Information Technology
License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
Programming Language :: Other
Topic :: Scientific/Engineering :: GIS
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

if sys.version_info < (3, 7):
    sys.exit("Sorry, Python 3.7 or newer is required.")

setup(
    name="rtc-tools-heat-network",
    version=versioneer.get_version(),
    description=DOCLINES[0],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    url="https://www.warmingup.info",
    author="Teresa Piovesan",
    author_email="teresa.piovesan@deltares.nl",
    maintainer="Teresa Piovesan",
    license="LGPLv3",
    keywords="heat network optimization rtc tools",
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["pyecore", "pymoca >= 0.9.0", "rtc-tools >= 2.5.0", "pyesdl >= 21.11.0"],
    tests_require=["pytest", "pytest-runner", "numpy"],
    include_package_data=True,
    python_requires=">=3.7",
    cmdclass=versioneer.get_cmdclass(),
    entry_points={
        "rtctools.libraries.modelica": ["library_folder = rtctools_heat_network:modelica"]
    },
)
