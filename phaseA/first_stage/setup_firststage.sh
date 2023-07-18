
python3.8 -m venv venv-1st-stage

PYTHON=venv-1st-stage/bin/python
PIP=venv-1st-stage/bin/pip

$PIP install --upgrade pip
$PIP install -r requirements.txt