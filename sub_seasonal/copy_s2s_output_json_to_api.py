#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:41:02 2023

Script to copy the output of the S2S forecast towards the API on the production
server, so WEnR can access the data.

This script only runs at the development server!

@author: bob
"""
import datetime
import os
from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir'] 

# Set today's date
today = datetime.datetime.today()
today = datetime.datetime(today.year,
                          today.month,
                          today.day,
                          0,0)
modeldate = today - datetime.timedelta(1)

# Take todays filename
local_dir = direc + 'output_forecast/'
fn = modeldate.strftime("s2s_forecast_for_agro_%Y%m%d.json")

# Set the filename for the production server
target_dir = '/srv/data/api/bangladesh_s2s/forecast_for_agro/json/en/'
target_fn = modeldate.strftime('bangladesh_s2s.forecast_for_agro.json.en_%Y%m%d0000.json')

# Copy the data
os.system(f'sshpass -p kqnY03zANO scp {local_dir}{fn} ecmwf@172.16.239.41:{target_dir}{target_fn}')
