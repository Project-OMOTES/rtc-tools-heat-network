
pushd .
cd /D "%~dp0"

cd ..\..\
call .\venv\Scripts\activate
set PYTHONPATH=.\src\;%$PYTHONPATH%
python -m mypy ./src/rtctools_heat_network ./tests/
popd