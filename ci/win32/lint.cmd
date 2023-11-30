rem Short script to run linting
rem @echo off

pushd .
cd /D "%~dp0"
cd ..\..\
flake8 .\src\rtctools_heat_network
popd
