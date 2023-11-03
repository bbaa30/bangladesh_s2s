#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:26:26 2022

Download the ECMWF S2S files from the Weather Impact API.
The data is available on Monday and Thursday evening around 22 UTC.

@author: bob
"""
import requests
import datetime
import os
from configparser import ConfigParser
import logging
import traceback

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

today = datetime.date.today()

# Set the directories from the config file
log_dir = config['paths']['home'] + 'logs/'
output_dir = config['paths']['s2s_dir'] + 'input_ecmwf/' 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

variables = ['tmax', 'tmin', 'tp']
fc_types = ['fc','hc']

base_url = "https://service.weatherimpact.com/api/data/bangladesh_s2s/"
header = {"authkey": "b19e4b0bf632513ab7e4637a696ba247"}

# Set the date of today
today = datetime.datetime.today()
today = datetime.datetime(today.year,
                          today.month,
                          today.day,
                          0,0)
todaystr = today.strftime("%Y%m%d")

logging.basicConfig(filename=log_dir+f'download_ecmwf_{todaystr}.log', filemode='w', level=logging.INFO)
logging.info("Start script at "+str(datetime.datetime.now()))

try:

    # make a loop, starting at today until 5 days back to download the data
    for timedelta in range(6): 
        modeldate = today - datetime.timedelta(timedelta)
        modeldatestr_api = modeldate.strftime("%Y-%m-%d")
        modeldatestr_out = modeldate.strftime("%Y%m%d")

        # Download the files for all variables, resolutions and forecast types
        for var in variables:
            for fc_type in fc_types:
                
                # Create the download url
                url_add = f"ecmwf_{fc_type}_{var}?datetime={modeldatestr_api}&format=netcdf"
                url = base_url + url_add
                
                # Do the api request
                r = requests.get(url, headers=header)
                
                # Check the status code. If the code is 200, the request is successful
                # Save the data if request is successful
                if r.status_code == 200:
                
                    # Write the response in a file
                    output_fn = f'{output_dir}ecmwf_{fc_type}_{var}_{modeldatestr_out}.nc'
                    
                    file = open(output_fn, "wb")
                    file.write(r.content)
                    file.close()
                    
    logging.info("Script finished succesfully at "+str(datetime.datetime.now()))
except Exception as e:
    print(e)
    logging.warning("Script failed. Traceback:")
    logging.error(traceback.format_exc())
