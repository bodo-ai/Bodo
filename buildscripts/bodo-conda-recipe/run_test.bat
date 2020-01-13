set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

mpiexec -localonly -n 3 python -u examples/pi.py
if errorlevel 1 exit 1
