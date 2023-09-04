#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:10:25 2023

@author: bob
"""
import numpy as np
import os
import cdsapi
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')


#%%
# Set the correct modeldate. The data is updated around the 10th - 15th of the month,
# so after the 16th, the new data is used.
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
meta_dir = direc + 'input_metadata/' # Directory with metadata

for direc in [data_dir_cds, data_dir_clim]:
    if not os.path.exists(direc):
        os.makedirs(direc)

# Mapping dictionairies to obtain the correct system version number and centre name
system_map_fn = 'seasonal_forecast_system_versions.csv'
maplist_full = pd.read_csv(meta_dir+system_map_fn)
maplist = maplist_full[np.logical_and(maplist_full['year']==year,maplist_full['month']==month)]
if maplist.empty:
    newmonth = True
    lastmonth = datetime.date(year,month,1)-relativedelta(months=1)
    maplist = maplist_full[np.logical_and(maplist_full['year']==lastmonth.year,maplist_full['month']==lastmonth.month)]
    maplist.year = year
    maplist.month = month
else:
    newmonth = False

name_map = {'ecmwf': 'ECMWF',
            'ukmo': 'Met Office',
            'meteo_france': 'Meteo France',
            'dwd': 'DWD',
            'cmcc': 'CMCC',
            'ncep': 'NCEP',
            'jma': 'JMA',
            'eccc': 'ECCC'}

centre_list = name_map.keys()

# Initialize the connection to CDS API
c = cdsapi.Client()

#%%
# Loop over all centres
for ic, centre in zip(np.arange(len(centre_list)),centre_list):
    
    try:
        print('Download the Copernicus data from the {} model'.format(centre))
        
        system = str(maplist[centre].values[0])        
        target = data_dir_cds + centre + '_' + str(year) + '_{:02d}.nc'.format(month)
        target_climate = data_dir_cds + centre + '_climate_' + str(year) + '_{:02d}.nc'.format(month)
        
        if system == '0':
            # Skip the centre if system version = 0. Then this model is not 
            # available for the given time
            continue
    
        ######################################################################
        ##                                                                  ##
        ##                Download data from Copernicus                     ##
        ##                                                                  ##
        ######################################################################
       
        # Download data if not already done so
        if not os.path.exists(target):
            try:
                c.retrieve(
                    'seasonal-monthly-single-levels',
                    {
                        'originating_centre':centre,
                        'system':system,
                        'variable': ['maximum_2m_temperature_in_the_last_24_hours',
                                     'minimum_2m_temperature_in_the_last_24_hours',
                                     'total_precipitation'],
                        'product_type':'monthly_mean',
                        'year':str(year),
                        'month':str(month),
                        'leadtime_month':[
                            '1','2','3',
                            '4','5','6'
                        ],
                        'area': [
                            27, 87, 20, 93,
                        ],
                        'format':'netcdf'
                    },
                    target)
                
            except:
                # Upgrade of system verion: add 1 to system version number
                # This is the issue that stops the download 
                maplist[centre].values[0] = maplist[centre].values[0] + 1
                system = str(maplist[centre].values[0])
                c.retrieve(
                    'seasonal-monthly-single-levels',
                    {
                        'originating_centre':centre,
                        'system':system,
                        'variable': ['maximum_2m_temperature_in_the_last_24_hours',
                                     'minimum_2m_temperature_in_the_last_24_hours',
                                     'total_precipitation'],
                        'product_type':'monthly_mean',
                        'year':str(year),
                        'month':str(month),
                        'leadtime_month':[
                            '1','2','3',
                            '4','5','6'
                        ],
                        'area': [
                            27, 87, 20, 93,
                        ],
                        'format':'netcdf'
                    },
                    target)
                # If this still fails, the script will stop automatically
            
            # Download the climatology, if not existent
        if not os.path.exists(target_climate):
            c.retrieve(
                    'seasonal-monthly-single-levels',
                    {
                        'originating_centre':centre,
                        'system':system,
                        'variable': ['maximum_2m_temperature_in_the_last_24_hours',
                                     'minimum_2m_temperature_in_the_last_24_hours',
                                     'total_precipitation'],
                        'product_type':'monthly_mean',
                        'year':[str(cl_yr) for cl_yr in np.arange(1993,2017)],
                        'month':str(month),
                        'leadtime_month':[
                            '1','2','3',
                            '4','5','6'
                        ],
                        'area': [
                            27, 87, 20, 93,
                        ],
                        'format':'netcdf'
                    },
                    target_climate)
    except:
        print('Unable to download the data from {}'.format(centre))
        
    ######################################################################
    ##                                                                  ##
    ##                     Process climate data                         ##
    ##                                                                  ##
    ######################################################################
    

    


# End of the loop, save the new system versions in metadata file if new month
if newmonth:
    maplist_full = pd.concat([maplist_full, maplist], ignore_index=True)
    # Overwrite the system mapping
    maplist_full.to_csv(meta_dir+system_map_fn, index=False)
