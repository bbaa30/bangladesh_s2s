#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:55:09 2023

Prepare the data to daily regridded values.
The script combines the runs with a different initial time to a bigger ensemble

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

input_dir_model = direc + 'input_ncep/'
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
logging.basicConfig(filename=log_dir+f'prepare_ncep_{todaystr}.log', filemode='w', level=logging.INFO)
logging.info("Start script at "+str(datetime.datetime.now()))

# List the variable names with a key and name in the NCEP files
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'varname_input': 'mx2t6',
                     'resample': 'max'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'varname_input': 'mn2t6',
                     'resample': 'min'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'varname_input': 'tp',
                   'resample': 'sum'}}

def convert_units(ds):
    if ds.units == 'K':
        ds = ds - 273.15
        ds.attrs['units'] = 'degrees Celsius'
    elif ds.units == 'kg m**-2':
        ds = ds * 0.998
        ds.values[ds.values <= 0] = 0
        ds.attrs['units'] = 'mm'
    return ds

# Loop over the last 5 days
for timedelta in range(1,6): 
    # Set the modeldate
    modeldate = today - datetime.timedelta(timedelta)
    modeldatestr = modeldate.strftime("%Y%m%d")
    datestr = modeldate.strftime("%d%b").lower()
    
    try:
        
        # Load the hindcast dataset
        fn_hc_old = input_dir_model + modeldate.strftime('ncep_hc_19992010_%Y%m%d.nc')
        fn_hc_new = input_dir_model + modeldate.strftime(f'ncep_hc_2015{modeldate.year-1}_%Y%m%d.nc')
        
        hc_old_all = xr.open_dataset(fn_hc_old)
        hc_new_all = xr.open_dataset(fn_hc_new)
        
        hc_years_old = np.arange(1999,2011)
        hc_years_new = np.arange(2015,modeldate.year)
            
        
        # Make the ensemble for each variable
        for var in varnames.keys():
            resample = varnames[var]['resample']
            
            output = output_dir + datestr + "/"
            if not os.path.exists(output):
                os.makedirs(output)
                
            fn_fc = f"{output}ncep_fc_regrid_{var}_{modeldatestr}.nc"
            fn_hc = f"{output}ncep_hc_regrid_{var}_{modeldatestr}.nc"
            fn_obs = f"{output}obs_ncep_{var}_{modeldatestr}.nc"
            
            if os.path.exists(fn_fc) and os.path.exists(fn_hc) and os.path.exists(fn_obs):
                # Data is already available, continue
                print(f'Data already prepared for {var} on {modeldate}.')
                continue
                        
            print(f'Prepare {var} data for {modeldate}')
            # Create an emsemble of 12 members, using the 4 runs of 3 days
            # Create a list with the datetimes
            dts = [modeldate - datetime.timedelta(hours=tt) for tt in range(-6,66,6)]
            fns = [input_dir_model + dt.strftime(f'ncep_fc_{var}_%Y%m%d%H.nc') for dt in dts]
            
            # Open the 12 files
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='modeldate')
            fc_data = ds[var]
            
            # Remove the time steps that contain NaN values
            fc_data = fc_data[:, np.logical_and(fc_data.time > np.datetime64(modeldate),
                                                fc_data.time < np.datetime64(modeldate + datetime.timedelta(30))), :, :]
            
            # Change modeldate dimension to member dimension
            fc_data = fc_data.assign_coords(modeldate=np.arange(1,len(fc_data.modeldate)+1)).rename({'modeldate': 'member'})
                    
            # Take the correct variable from the hindcast data
            varname_hc = varnames[var]['varname_input']
            hc_old = hc_old_all[varname_hc]
            hc_new = hc_new_all[varname_hc]
            
            if varname_hc == 'tp':
                hc_old.data[1:] = (hc_old.data[1:] - hc_old.data[:-1])
                hc_old.data[hc_old.data < 0] = 0.
                hc_new.data[1:] = (hc_new.data[1:] - hc_new.data[:-1])
                hc_new.data[hc_new.data < 0] = 0.
                
            # Use a trick to match the number of ensemble members
            hc_old = xr.concat([hc_old.assign_coords(number=hc_old.number+ii*3) for ii in range(5)], dim='number')
                    
            # Combine hindcast datsets and convert units
            hc = xr.concat([hc_old, hc_new], dim='time')
            hc = hc.transpose('time','number','latitude','longitude')
            hc = convert_units(hc)

            n_years = len(hc_years_old) + len(hc_years_new)
            len_1yr = int(len(hc)/n_years)            
            
            # Resample to daily values
            print('Resample data to daily values')
            for yy in range(n_years):
                if resample == 'max':
                    hc_daily_yr = hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).max('time')
                elif resample == 'min':
                    hc_daily_yr = hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).min('time')
                elif resample == 'sum':
                    hc_daily_yr = hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).sum('time')
                    
                if yy == 0:
                    hc_daily = hc_daily_yr
                else:
                    hc_daily = xr.concat((hc_daily, hc_daily_yr), dim='time')
            
            # Change the NCEP time array with 6 hours because of the time difference
            fc_data = fc_data.assign_coords(time=fc_data.time - np.timedelta64(6, 'h'))
            hc_daily = hc_daily.assign_coords(time=hc_daily.time - np.timedelta64(6, 'h'))
 
            # Load the BMD gridded data
            obs_var_fn = varnames[var]['obs_filename']
            obs_var_nc = varnames[var]['obs_ncname']
            obs = xr.open_mfdataset(f'{input_dir_obs}merge_{obs_var_fn}_*')
            obs = obs[obs_var_nc]
            
            # Regrid the NCEP data to the BMD gridded data
            print('Regrid NCEP CFSv2 data to BMD grid')
            fc_daily = xc.regrid(fc_data, obs.coords['Lon'].values, obs.coords['Lat'].values, x_feature_dim='member')        
            fc_daily = fc_daily.transpose('time','member','latitude','longitude')
            hc_daily = xc.regrid(hc_daily, obs.coords['Lon'].values, obs.coords['Lat'].values, x_feature_dim='number')        
            hc_daily = hc_daily.transpose('time','number','latitude','longitude').rename({'number': 'member'})

            # Match the observation timesteps with the hindcast timesteps
            obs = obs[obs.time >= np.datetime64(hc_daily.time.values[0])-np.timedelta64(2, 'D')]
            
            days_to_take_obs = [obs.time[ii].values in hc_daily.time for ii in range(len(obs.time))]
            days_to_take_hc = [hc_daily.time[ii].values in obs.time for ii in range(len(hc_daily.time))]
            
            obs = obs[days_to_take_obs]
            hc_daily = hc_daily[days_to_take_hc]
            
            print('Save the data')        
            obs.to_netcdf(fn_obs)
            fc_daily.to_netcdf(fn_fc)
            hc_daily.to_netcdf(fn_hc)

        logging.info("Script finished successful at "+str(datetime.datetime.now()))

    except Exception as e:
        print(e)
        print(f'No data available for {modeldate}. Continue to previous day.')
        logging.warning(f"No data available for {modeldate}. Continue to previous day.")
        logging.warning(traceback.format_exc())
        continue
