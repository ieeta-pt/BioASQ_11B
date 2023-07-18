
python3.10 -m venv venv-2nd-stage

PYTHON=venv-2nd-stage/bin/python
PIP=venv-2nd-stage/bin/pip

$PIP install --upgrade pip
$PIP install -r requirements.txt