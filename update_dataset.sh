#!/bin/bash	
cd /home/ec2-user/covid/COVID-19
git fetch --all
git pull origin master
cd /home/ec2-user/covid/etl
source env/bin/activate
python --version
python -W ignore build_reports.py
cd /home/ec2-user/covid/etl/datasets/countries
aws s3 cp . s3://co.data.covid19-us-east-2/countries/ --recursive
cd /home/ec2-user/covid/etl
aws s3 cp processed_countries.csv s3://co.data.covid19-us-east-2
