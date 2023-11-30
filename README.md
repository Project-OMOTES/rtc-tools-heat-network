# RTC Tools Heat Network

This repository is part of the 'Nieuwe Warmte Nu Design Toolkit' project. 

RTC Tools Heat Network is used for ESDLs containing a heat network. It can optimize and low-fidelity simulate OPEX and CAPEX costs as well as provide the optimal asset placements.

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
