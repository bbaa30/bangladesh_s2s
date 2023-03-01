# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:09:05 2023

This script acts as a wrapper to run all the operational scripts

@author: Bob.Ammerlaan
"""
from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

script_dir = config['paths']['scripts'] 

# 1a. Download ECMWF data
try:
    print('1a. Download ECMWF data')
    exec(open(f"{script_dir}download_ecmwf_s2s_from_wi_api.py").read())
except Exception as e:
    print('The ECMWF download has ended in an error:\n')
    print(e)
    print('\n')

# 1b. Download ECCC data
try:
    print('1b. Download ECCC data')
    exec(open(f"{script_dir}download_eccc_s2s_operational.py").read())
except Exception as e:
    print('The ECCC download has ended in an error:\n')
    print(e)
    print('\n')

# 1c1. Download NCEP CFSv2 operational data
try:
    print('1c1. Download NCEP CFSv2 operational data')
    exec(open(f"{script_dir}download_cfsv2_operational.py").read())
except Exception as e:
    print('The NCEP CFSv2 operational download has ended in an error:\n')
    print(e)
    print('\n')

# 1c2. Download NCEP CFSv2 hindcast data
try:
    print('1c2. Download NCEP CFSv2 hindcast data')
    exec(open(f"{script_dir}download_cfsv2_hindcasts.py").read())
except Exception as e:
    print('The NCEP CFSv2 hindcast download has ended in an error:\n')
    print(e)
    print('\n')

# 2a. Preprocess ECMWF data
try:
    print('2a. Preprocess ECMWF data')
    exec(open(f"{script_dir}prepare_ecmwf_data.py").read())
except Exception as e:
    print('The ECMWF preprocessing has ended in an error:\n')
    print(e)
    print('\n')

# 2b. Preprocess ECMWF data
try:
    print('2b. Preprocess ECCC data')
    exec(open(f"{script_dir}prepare_eccc_data.py").read())
except Exception as e:
    print('The ECCC preprocessing has ended in an error:\n')
    print(e)
    print('\n')

# 2c. Preprocess ECMWF data
try:
    print('2c. Preprocess NCEP CFSv2 data')
    exec(open(f"{script_dir}prepare_ncep_data.py").read())
except Exception as e:
    print('The NCEP CFSv2 preprocessing has ended in an error:\n')
    print(e)
    print('\n')

# 3. Run the operational forecast
try:
    print('3. Run operational forecast')
    exec(open(f"{script_dir}s2s_operational_forecast.py").read())
except Exception as e:
    print('The operational forecast has ended in an error:\n')
    print(e)
    print('\n')

# 4. Generate the bulletin
try:
    print('4. Generate bulletin')
    exec(open(f"{script_dir}generate_bulletin.py").read())
except Exception as e:
    print('The generation of the bulletin has ended in an error:\n')
    print(e)
    print('\n')
