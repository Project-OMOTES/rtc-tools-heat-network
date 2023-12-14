
pushd .
cd /D "%~dp0"

cd ..\..\
call .\venv\Scripts\activate
set PYTHONPATH=.\src\;%$PYTHONPATH%
pytest tests/
popd