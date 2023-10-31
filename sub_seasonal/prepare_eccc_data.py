#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:55:09 2023

Prepare the data to daily regridded values

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

input_dir_model = direc + 'input_eccc/'
input_dir_obs = config['paths']['data_dir']  + 'input_bmd_gridded_data/'
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
logging.basicConfig(filename=log_dir+f'prepare_eccc_{todaystr}.log', filemode='w', level=logging.INFO)
logging.info("Start script at "+str(datetime.datetime.now()))

# List the variable names with a key and name in the ECCC files
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'varname_input': 'tasmax',
                     'resample': 'max'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'varname_input': 'tasmin',
                     'resample': 'min'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'varname_input': 'pr',
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
            print(f'Start forecasts for {var}.')
            
            var_input = varnames[var]['varname_input']
            
            # Load the ECCC hindcast and forecast
            fn_hc = f"{input_dir_model}ECCC_hc_{var_input}_{modeldatestr}.nc"
            fn_fc = f"{input_dir_model}ECCC_fc_{var_input}_{modeldatestr}.nc"
            
            fc_data = xr.open_dataarray(fn_fc)    
            hc_data = xr.open_dataarray(fn_hc)
            
            # Rename the dimensions of the forecast
            fc_data=fc_data.rename({'M':'member','L':'time','Y':'latitude','X':'longitude'}).drop('S')
            hc_data=hc_data.rename({'M':'member','L':'time','Y':'latitude','X':'longitude'})
            
            # Load the BMD gridded data
            obs_var_fn = varnames[var]['obs_filename']
            obs_var_nc = varnames[var]['obs_ncname']
            obs = xr.open_mfdataset(f'{input_dir_obs}merge_{obs_var_fn}_*')
            obs = obs[obs_var_nc]
            
            # Regrid the ECCC data to the BMD gridded data
            print('Regrid ECCC data to BMD grid')
            fc_daily = xc.regrid(fc_data, obs.coords['Lon'].values, obs.coords['Lat'].values, x_feature_dim='member')
            hc_daily = xc.regrid(hc_data, obs.coords['Lon'].values, obs.coords['Lat'].values, x_feature_dim='member')
            
            # Remove the old variables
            del fc_data, hc_data
            
            print('Take overlapping time steps')
            
            # Start the observations at the same time as the hindcasts
            obs = obs[obs.time >= np.datetime64(hc_daily.time.values[0])-np.timedelta64(2, 'D')]
            
            # Take the data from the observations to create the same time series
            obs = obs.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in obs.coords['time'].values], dims='time')})
            fc_daily = fc_daily.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in fc_daily.coords['time'].values], dims='time')})
            hc_daily = hc_daily.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in hc_daily.coords['time'].values], dims='time')})
            
            doy_fc = np.unique(fc_daily.coords['doy'].values)
            days_to_take = [obs.doy[ii].values in doy_fc for ii in range(len(obs.doy))]
            
            obs = obs[days_to_take]
            
            # Make sure we take only the 20 years of hindcast
            obs = obs[:len(hc_daily.time)]
            hc_daily = hc_daily[:,:len(hc_daily.time)]
            
            # Transpose the data to time, member, lat, lon
            hc_daily = hc_daily.transpose('time','member','latitude','longitude')
            fc_daily = fc_daily.transpose('time','member','latitude','longitude')
        
        
            output = output_dir + datestr + "/"
            if not os.path.exists(output):
                os.makedirs(output)
        
            print('Save the data')
            fn_hc = f"{output}eccc_hc_regrid_{var}_{modeldatestr}.nc"
            fn_fc = f"{output}eccc_fc_regrid_{var}_{modeldatestr}.nc"
            fn_obs = f"{output}obs_eccc_{var}_{modeldatestr}.nc"
        
            obs.to_netcdf(fn_obs)
            fc_daily.to_netcdf(fn_fc)
            hc_daily.to_netcdf(fn_hc)

            logging.info("Script finished successful at "+str(datetime.datetime.now()))

    except:
        print(f'No data available for {modeldate}. Continue to previous day.')
        logging.error(traceback.format_exc())
        continue
