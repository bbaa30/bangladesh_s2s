#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:21:29 2022

Download NCEP CFSv2 data operationally from the NOMADS server.
The CFSv2 data is available 4 times per day. Run this script on a daily basis.

@author: bob
"""

import datetime
import os
import numpy as np
import xarray as xr
import requests
import time
import shutil
import logging
import traceback

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

log_dir = config['paths']['home'] + 'logs/'
direc = config['paths']['s2s_dir'] + 'input_ncep/'

if not os.path.exists(direc):
    os.makedirs(direc)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
# Set current date
today = datetime.datetime.today()
todaystr = today.strftime("%Y%m%d")

# Define the grib filter query url
source_location = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_cfs_flx.pl?file=' 

var_level_query = '&lev_2_m_above_ground=on&lev_surface=on&var_PRATE=on&var_TMAX=on&var_TMIN=on'
region_query = '&subregion=&leftlon=87&rightlon=93&toplat=27&bottomlat=20'

# Make an array with all forecast timesteps
forecast_times = np.arange(0,768,6)

# Variables
varlist = ['tmax', 'tmin', 'prate']

# Convert units
def convert_units(ds):
    if ds.units == 'K':
        ds = ds - 273.15
        ds.attrs['units'] = 'degrees Celsius'
    elif ds.units == 'kg m**-2 s**-1':
        ds = ds * 6*3600
        ds.values[ds.values <= 0] = 0
        ds.attrs['units'] = 'mm'
    return ds

logging.basicConfig(filename=log_dir+f'download_cfsv2_operational_{todaystr}.log', filemode='w', level=logging.INFO)
logging.info("Start script at "+str(datetime.datetime.now()))

# Download the data for the latest 3 dates for all 4 modelruns (0, 6, 12 and 18 UTC)
# This gives a total of 12 modelruns to combine to a single lagged ensemble
for timelag in [4,3,2,1]:
    for modelrun in [0,6,12,18]:
                
        # Set model datetime
        modeldate = today - datetime.timedelta(timelag)
        modeldate = datetime.datetime(modeldate.year, modeldate.month, modeldate.day, modelrun, 0)

        # Define the filenames of the tmax, tmin and tp files
        fn_tmin = f"{direc}/ncep_fc_tmin_{modeldate.strftime('%Y%m%d%H')}.nc"
        fn_tmax = f"{direc}/ncep_fc_tmax_{modeldate.strftime('%Y%m%d%H')}.nc"
        fn_tp = f"{direc}/ncep_fc_tp_{modeldate.strftime('%Y%m%d%H')}.nc"
        
        # If one of the final files is not existent: download the data and prepare the nc-files
        if not os.path.exists(fn_tmin) and not os.path.exists(fn_tmax) and not os.path.exists(fn_tp):
            # Set the target location
            target_location = direc + modeldate.strftime("%Y%m%d%H/")
            
            # If already a part of the download is available, remove it and start again
            if os.path.exists(target_location):
                shutil.rmtree(target_location)
            # Create a new and emtpy directory
            os.makedirs(target_location)
            
            # Loop over all forecast times. Each time step has a separate file on 
            # the NOMADS server and needs to be downloaded separately
            for fc_time in forecast_times:
                forecast_timestep = modeldate + datetime.timedelta(hours = int(fc_time))
                
                # Build up the grib-filter query
                file_query = forecast_timestep.strftime('flxf%Y%m%d%H.01.') + modeldate.strftime('%Y%m%d%H.grb2')
                dir_query = '&dir=%2Fcfs.' + modeldate.strftime('%Y%m%d') + '%2F' + modeldate.strftime('%H') + '%2F6hrly_grib_01'
                
                query = source_location + file_query + var_level_query + region_query + dir_query
                
                if not os.path.exists(target_location+file_query):
                    # Do the request
                    try:
                        # There is a maximum requests per minute for gfs download
                        r = requests.get(query)
                        time.sleep(2)
                    except Exception as e:
                        # So if an error is raised because this number has been reached
                        # Wait for 30 seconds and continue
                        print(e)
                        print('Wait for 30 seconds and try again')
                        time.sleep(30)
                        r = requests.get(query)            
        
                    if len(r.content) > 100.:
                        # Make the local file and save the downloaded data in the file
                        ftarget = open(target_location + file_query, 'wb')
                        ftarget.write(r.content)
                        ftarget.close()
                    else:
                        # If the file contains less than 1000 bytes, it is emty, so try again once more.
                        time.sleep(4)
                        r = requests.get(query)
                        time.sleep(2)
                        if len(r.content) > 100.:
                            # Make the local file and save the downloaded data in the file
                            ftarget = open(target_location + file_query, 'wb')
                            ftarget.write(r.content)
                            ftarget.close()
                        else:
                            raise(Exception('File '+file_query+' not yet available.'))
    
            try:
                # Convert to netcdf
                ds = xr.open_mfdataset(target_location + 'flxf*', concat_dim='step', combine='nested')
                
                # Take the data out of the dataset
                da_tmin = convert_units(ds.tmin)
                da_tmax = convert_units(ds.tmax)
                da_tp = convert_units(ds.rename({'prate':'tp'}).tp)
                
                # Convert the time and modeldate dimensions
                da_tmin = da_tmin.rename({'time': 'modeldate'})
                da_tmin = da_tmin.assign_coords(step=da_tmin.valid_time).drop('valid_time').rename({'step':'time'})
        
                da_tmax = da_tmax.rename({'time': 'modeldate'})
                da_tmax = da_tmax.assign_coords(step=da_tmax.valid_time).drop('valid_time').rename({'step':'time'})
                
                da_tp = da_tp.rename({'time': 'modeldate'})
                da_tp = da_tp.assign_coords(step=da_tp.valid_time).drop('valid_time').rename({'step':'time'})
                
                # Convert to daily values
                da_tmin = da_tmin.resample(time='24H', base=6).min('time')
                da_tmax = da_tmax.resample(time='24H', base=6).max('time')
                da_tp = da_tp.resample(time='24H', base=6).sum('time')
                
                # Save as netcdf
                da_tmin.to_netcdf(fn_tmin)
                da_tmax.to_netcdf(fn_tmax)
                da_tp.to_netcdf(fn_tp)
                
                logging.info(f"Succesfully saved {fn_tmin}, {fn_tmax} and {fn_tp} at "+str(datetime.datetime.now()))
            
            except:
                logging.error(traceback.format_exc())
                print(f'Could not convert to daily values for modeldate: {modeldate}')
