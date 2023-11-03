#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:55:09 2023

Prepare the ECMWF data to daily regridded values

@author: bob
"""
import xarray as xr
import xcast as xc
import datetime
import os
import numpy as np
import logging
import traceback

import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir'] 

input_dir_ec = direc + 'input_ecmwf/'
input_dir_obs = config['paths']['data_dir'] + 'input_bmd_gridded_data/'
output_dir = direc + 'input_regrid/'
log_dir = config['paths']['home'] + 'logs/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
# Set the date of today
today = datetime.datetime.today()
today = datetime.datetime(today.year,
                          today.month,
                          today.day,
                          0,0)
todaystr = today.strftime("%Y%m%d")

# Configure logging
logging.basicConfig(filename=log_dir+f'prepare_ecmwf_{todaystr}.log', filemode='w', level=logging.INFO)
logging.info("Start script at "+str(datetime.datetime.now()))

# List the variable names with a key and name in the ECMWF files
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'resample': 'max'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'resample': 'min'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'resample': 'sum'}}

# Loop over the last 5 days
for timedelta in range(6): 
    # Set the modeldate
    modeldate = today - datetime.timedelta(timedelta)
    modeldatestr = modeldate.strftime("%Y%m%d")
    datestr = modeldate.strftime("%d%b").lower()

    # Try to prepare the data for the specific modeldate
    # And continue to the next day if there is no data available
    try:
        # Make a forecast for each variable
        for var in varnames.keys():            
            # Load the ECMWF hindcast and forecast
            fn_ec_hc = f"{input_dir_ec}ecmwf_hc_{var}_{modeldatestr}.nc"
            fn_ec_fc = f"{input_dir_ec}ecmwf_fc_{var}_{modeldatestr}.nc"
            
            ec_fc = xr.open_dataarray(fn_ec_fc)            
            ec_hc = xr.open_dataarray(fn_ec_hc)
            
            print(f'Start forecasts for {var}.')
            
            # Load the BMD gridded data
            obs_var_fn = varnames[var]['obs_filename']
            obs_var_nc = varnames[var]['obs_ncname']
            obs = xr.open_mfdataset(f'{input_dir_obs}merge_{obs_var_fn}_*')
            obs = obs[obs_var_nc]
                        
            # Generate daily values, use a time zone offset of 6 hours, so the daily
            # value is calculated from 6UTC-6UTC to match the Bangladesh day best
            resample = varnames[var]['resample']
            len_1yr = len(ec_fc)
            nr_years = int(len(ec_hc) / len_1yr)
            
            print('Resample data to daily values')
            if resample == 'max':
                ec_fc_daily = ec_fc.resample(time='24H', base=6).max('time')
                for yy in range(nr_years):
                    ec_hc_daily_yr = ec_hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).max('time')
                    if yy == 0:
                        ec_hc_daily = ec_hc_daily_yr
                    else:
                        ec_hc_daily = xr.concat((ec_hc_daily, ec_hc_daily_yr), dim='time')
            elif resample == 'min':
                ec_fc_daily = ec_fc.resample(time='24H', base=6).min('time')
                for yy in range(nr_years):
                    ec_hc_daily_yr = ec_hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).min('time')
                    if yy == 0:
                        ec_hc_daily = ec_hc_daily_yr
                    else:
                        ec_hc_daily = xr.concat((ec_hc_daily, ec_hc_daily_yr), dim='time')
            elif resample == 'sum':
                ec_fc_daily = ec_fc.resample(time='24H', base=6).sum('time')
                for yy in range(nr_years):
                    ec_hc_daily_yr = ec_hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).sum('time')
                    if yy == 0:
                        ec_hc_daily = ec_hc_daily_yr
                    else:
                        ec_hc_daily = xr.concat((ec_hc_daily, ec_hc_daily_yr), dim='time')
            else:
                raise(Exception(f'Unkown resample type {resample}'))
            
            print('Regrid ECMWF data to BMD grid')
            try:
                ec_fc_daily = xc.regrid(ec_fc_daily, obs.coords['Lon'].values, obs.coords['Lat'].values)
                ec_hc_daily = xc.regrid(ec_hc_daily, obs.coords['Lon'].values, obs.coords['Lat'].values)
            except:
                ec_fc_daily = xc.regrid(ec_fc_daily, obs.coords['Lon'].values, obs.coords['Lat'].values, x_sample_dim= 'member')
                ec_hc_daily = xc.regrid(ec_hc_daily, obs.coords['Lon'].values, obs.coords['Lat'].values, x_sample_dim= 'member')
                
            print('Take overlapping time steps')
            # Change the ECMWF time array with 6 hours because of time difference
            ec_fc_daily = ec_fc_daily.assign_coords(time=ec_fc_daily.time - np.timedelta64(6, 'h'))
            ec_hc_daily = ec_hc_daily.assign_coords(time=ec_hc_daily.time - np.timedelta64(6, 'h'))
            
            # Start the observations at the same time as the hindcasts
            obs = obs[obs.time >= np.datetime64(ec_hc_daily.time.values[0])-np.timedelta64(2, 'D')]
            
            # Take the overlapping timesteps in hindcast and observations
            days_to_take_obs = [obs.time[ii].values in ec_hc_daily.time for ii in range(len(obs.time))]
            days_to_take_hc = [ec_hc_daily.time[ii].values in obs.time for ii in range(len(ec_hc_daily.time))]
            
            # Make sure that all dimensions are at the correct position
            ec_hc_daily = ec_hc_daily.transpose("time", "member", "latitude", "longitude")
            ec_fc_daily = ec_fc_daily.transpose("time", "member", "latitude", "longitude")
            
            obs = obs[days_to_take_obs]
            ec_hc_daily = ec_hc_daily[days_to_take_hc]

            output = output_dir + datestr + "/"
            if not os.path.exists(output):
                os.makedirs(output)
                        
            print('Save the data')
            fn_hc = f"{output}ecmwf_hc_regrid_{var}_{modeldatestr}.nc"
            fn_fc = f"{output}ecmwf_fc_regrid_{var}_{modeldatestr}.nc"
            fn_obs = f"{output}obs_ecmwf_{var}_{modeldatestr}.nc"
        
            obs.to_netcdf(fn_obs)
            ec_fc_daily.to_netcdf(fn_fc)
            ec_hc_daily.to_netcdf(fn_hc)
        
            del obs, ec_fc_daily, ec_hc_daily
            
        logging.info("Script finished successful at "+str(datetime.datetime.now()))

    except Exception as e:
        print(e)
        print(f'No data available for {modeldate}. Continue to previous day.')
        logging.warning(f"No data available for {modeldate}. Continue to previous day.")
        logging.warning(traceback.format_exc())
        continue
