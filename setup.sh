#!/bin/bash

# Full setup script this donwloads data and creates the venvs

# Download data.zip file
DATA_ZIP_FILE="BioASQ-training11b.zip"
URL_DATA_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=9ffc4529590b4c129431f003bc534dd3&filename=$DATA_ZIP_FILE&openfolder=forcedownload&ep="

cd data
echo "Download $DATA_ZIP_FILE"
wget -c -O $DATA_ZIP_FILE $URL_DATA_ZIP_FILE

echo "Unzip $DATA_ZIP_FILE"
unzip -u $DATA_ZIP_FILE
cd -

DATA_ZIP_FILE="Batches.zip"
URL_DATA_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=0589d087f24d423c99433b0b1d03354d&filename=$DATA_ZIP_FILE&openfolder=forcedownload&ep="

cd phaseA/first_stage
echo "Download $DATA_ZIP_FILE"
wget -c -O $DATA_ZIP_FILE $URL_DATA_ZIP_FILE

echo "Unzip $DATA_ZIP_FILE"
unzip -u $DATA_ZIP_FILE
cd -
# hard assertion for python3.8 and at least python3.10
# 

# setup first-stage
echo "Preparing PhaseA first-stage virtual env"
cd phaseA/first_stage
bash setup_firststage.sh

# setup second-stage
echo "Preparing PhaseA second-stage virtual env"
cd ../second_stage
bash setup_secondstage.sh

echo "Preparing PhaseB answer generation virtual env"
# setup phaseB aka answer generation
cd ../../phaseB
bash setup.sh


