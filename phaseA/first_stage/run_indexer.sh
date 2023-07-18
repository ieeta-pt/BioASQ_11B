#!/bin/bash

PYTHON=venv-1st-stage/bin/python

for year in {2013..2023}
do 
    echo $year
    $PYTHON -m pyserini.index --input ../../data/processed/$year --collection JsonCollection --generator DefaultLuceneDocumentGenerator --index ../../data/indexes/baseline_$year --threads 1 --storePositions --storeDocvectors --storeRaw
done