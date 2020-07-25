#!/bin/bash	
cd ../COVID-19
git fetch --all
git pull origin master
cd ..
python build_reports.py
cd datasets/countries
aws s3 cp . s3://co.data.covid19-us-east-2/countries/ --recursive


