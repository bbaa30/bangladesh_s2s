#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:36:59 2022

Open the just downloaded Copernicus climate files, and make similar data files.
BMD gridded data is also loaded and regridded to the CDS data.

@author: weatherimpact
"""
import numpy as np
import os
import datetime
import xarray as xr
import xcast as xc
from dateutil.relativedelta import relativedelta

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

date = datetime.date.today()
if date.day < 16:
    year = (datetime.date(date.year,date.month,1)-relativedelta(months=1)).year
    month = (datetime.date(date.year,date.month,1)-relativedelta(months=1)).month
else:
    year = date.year
    month = date.month

# Define the correct paths
direc = config['paths']['seasonal_dir'] 
data_dir_cds = direc + 'input_cds_files/' # Directory where Copernicus forecasts are stored
data_dir_clim = direc + 'input_climatology/' # Directory where climatology is stored
output_dir = direc + 'input_regrid/'
input_dir_obs = config['paths']['data_dir'] + 'input_bmd_gridded_data/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

name_map = {'ecmwf': 'ECMWF',
            'ukmo': 'Met Office',
            'meteo_france': 'Meteo France',
            'dwd': 'DWD',
            'cmcc': 'CMCC',
            'ncep': 'NCEP',
            'jma': 'JMA',
            'eccc': 'ECCC'}

centre_list = name_map.keys()

# Set the metadata for the variables
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'cds_varname': 'mx2t24',
                     'resample': 'mean'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'cds_varname': 'mn2t24',
                     'resample': 'mean'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'cds_varname': 'tprate',
                   'resample': 'sum'}}

# Define a grid of 0.25 degrees where to regrid the data to
lats = np.arange(20,27.1,0.25)
lons = np.arange(87,93.1,0.25)

def convert_units(ds):
    if ds.units == 'K':
        ds = ds - 273.15
        ds.attrs['units'] = 'degrees Celsius'
    elif ds.units == 'm s**-1':
        # We count with a month of 30 days for the conversion of precipitation
        ds = ds * 30* 24*3600*1000
        ds.values[ds.values <= 0] = 0
        ds.attrs['units'] = 'mm'
    return ds

#%%
# Loop over all centres
for ic, centre in zip(np.arange(len(centre_list)),centre_list):
    
    for var in varnames.keys():
        print(f"Load CDS files from {centre} and for variable: {var}")
        # Load the CDS data files
        varname_cds = varnames[var]['cds_varname']
        resample = varnames[var]['resample']
    
        fn_forecast = data_dir_cds + centre + '_' + str(year) + '_{:02d}.nc'.format(month)
        fn_climate = data_dir_cds + centre + '_climate_' + str(year) + '_{:02d}.nc'.format(month)

        ds_forecast = xr.open_dataset(fn_forecast)
        ds_climate = xr.open_dataset(fn_climate)
        
        da_forecast = ds_forecast[varname_cds]
        da_climate = ds_climate[varname_cds]
        
        print("Convert units")
        da_forecast = convert_units(da_forecast)
        da_climate = convert_units(da_climate)
        
        # Load the BMD gridded data
        obs_var_fn = varnames[var]['obs_filename']
        obs_var_nc = varnames[var]['obs_ncname']
        obs = xr.open_mfdataset(f'{input_dir_obs}merge_{obs_var_fn}_*')
        obs = obs[obs_var_nc]
        obs = obs.expand_dims({'number': [0]})
        
        # Combine observation data to monthly values
        if resample == 'mean':
            obs_monthly = obs.resample(time='1MS').mean('time', skipna=True)
        elif resample == 'sum':
            obs_monthly = obs.resample(time='1MS').sum('time', skipna=True)
        else:
            raise(Exception(f"Unknown resample type: {resample}"))
        
        print('Regrid all datasets')
        # Regrid the BMD observations to the climate data
        obs_regrid = xc.regrid(obs_monthly, lons, lats,
                               x_sample_dim= 'number', x_feature_dim='time')
        fc_regird = xc.regrid(da_forecast, lons, lats,
                              x_sample_dim= 'number', x_feature_dim='time')
        hc_regrid = xc.regrid(da_climate, lons, lats,
                              x_sample_dim= 'number', x_feature_dim='time')
        
        days_to_take_obs = [obs_regrid.time[ii].values in da_climate.time for ii in range(len(obs_regrid.time))]
        obs_regrid = obs_regrid[days_to_take_obs]
        
        print('Save the data')
        fn_hc = f"{output_dir}{centre}_hc_regrid_{var}_"+ str(year) + '_{:02d}'.format(month)+".nc"
        fn_fc = f"{output_dir}{centre}_fc_regrid_{var}_"+ str(year) + '_{:02d}'.format(month)+".nc"
        fn_obs = f"{output_dir}obs_{centre}_{var}_"+ str(year) + '_{:02d}'.format(month)+".nc"
    
        obs_regrid.to_netcdf(fn_obs)
        fc_regird.to_netcdf(fn_fc)
        hc_regrid.to_netcdf(fn_hc)
    
