# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:24:38 2023

Download BMD gridded data

@author: Bob.Ammerlaan
"""
import os
import urllib.request
from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir']  + 'input_bmd_gridded_data/'

if not os.path.exists(direc):
    os.makedirs(direc)

# Set the URL to the ENACTS data
rr_folder = "http://bmdobs.rimes.int/ENACTS_Rainfall/Daily_Rainfall/"
tmax_folder = "http://bmdobs.rimes.int/ENACTS_Tmax/Daily_Tmax/"
tmin_folder = "http://bmdobs.rimes.int/ENACTS_Tmin/Daily_Tmin/"

# Define the available years
years = range(1981,2022)

# Loop over all available years
for year in years:
    
    # Download rainfall, maximum and minimum temperature
    if not os.path.exists(direc + f'merge_rr_{year}.nc'):
        urllib.request.urlretrieve(rr_folder + f'merge_rr_{year}.nc', direc + f'merge_rr_{year}.nc')
    if not os.path.exists(direc + f'merge_tmin_{year}.nc'):
        urllib.request.urlretrieve(tmin_folder + f'merge_tmin_{year}.nc', direc + f'merge_tmin_{year}.nc')
    if not os.path.exists(direc + f'merge_tmax_{year}.nc'):
        urllib.request.urlretrieve(tmax_folder + f'merge_tmax_{year}.nc', direc + f'merge_tmax_{year}.nc')