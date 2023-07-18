#!/bin/bash

PYTHON=venv-1st-stage/bin/python

$PYTHON bm25_testset_inference.py --testset Batch01/BioASQ-task11bPhaseA-testset1 --k1 0.5 --beta 0.3
$PYTHON bm25_testset_inference.py --testset Batch02/BioASQ-task11bPhaseA-testset2 --k1 0.5 --beta 0.3
$PYTHON bm25_testset_inference.py --testset Batch03/BioASQ-task11bPhaseA-testset3 --k1 0.5 --beta 0.3
$PYTHON bm25_testset_inference.py --testset Batch04/BioASQ-task11bPhaseA-testset4 --k1 0.5 --beta 0.3