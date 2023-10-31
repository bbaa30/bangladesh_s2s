#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:13:24 2022

Generate operational forecast from ECMWF, NCEP CFSv2 and ECCC.

Read in the prepared netcdf files with forecast and hindcast.
Regrid the data to the BMD gridded data-grid

Calibrate the forecast and fit current forecast.

TO DO: include KMA/JMA/CMA models in forecast if available

@author: bob
Modified by: Lorenzo: place the data aggregation before the forecast processing
"""
import xarray as xr
import xcast as xc
import datetime
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shapefile
import geopandas as gpd
import pandas as pd
import json
import logging
import traceback
from scipy.stats import percentileofscore

import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir']
home_dir = config['paths']['home']
log_dir = config['paths']['home'] + 'logs/'

lib_dir = config['paths']['library']
sys.path.append(lib_dir)
import s2s_library as s2s

input_gen_dir = direc + 'input_regrid/'
fig_dir_hc = direc + 'output_figures_hindcast/' 
fig_dir_fc = direc + 'output_figures_forecast/' 
output_dir = direc + 'output_forecast/' 

# Make output directories if not existent
if not os.path.exists(fig_dir_hc):
    os.makedirs(fig_dir_hc)
if not os.path.exists(fig_dir_fc):
    os.makedirs(fig_dir_fc)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set today's date
today = datetime.datetime.today()
today = datetime.datetime(today.year,
                          today.month,
                          today.day,
                          0,0)
todaystr = today.strftime("%Y%m%d")

# Configure logging
logging.basicConfig(filename=log_dir+f's2s_forecast_{todaystr}.log', filemode='w', level=logging.INFO)
logging.info("Start script at "+str(datetime.datetime.now()))

######## OPTION TO EXCLUDE MODELS FROM MULTIMODEL! ###########
include_ecmwf = True
include_eccc = True
include_cfsv2 = True
##############################################################

######### OPTION TO EXECUTE SINGLE MODEL FCST  ###############
single_model = True
##############################################################

############# OPTION TO EXCLUDE SKILL ANALYSIS  ##############
skill_analysis = False
##############################################################

# List the variable names with some properties
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'resample': 'mean',
                     'unit': 'degrees C'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'resample': 'mean',
                     'unit': 'degrees C'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'resample': 'sum',
                   'unit': 'mm'}}

# Make a dictionary with start and end times per period
start_end_times = {'week1': {'start': 1,
                             'end': 8,
                             'wknr': '1'},
                   'week2': {'start': 8,
                             'end': 15,
                             'wknr': '2'},
                   'week3+4': {'start': 15,
                               'end': 29,
                               'wknr': '34'}
                   }

# Make a dictionary with information on the spatial aggregation
shape_info = {0: {'name_column': 1,             # Level 0 is the country
                  'name_column_higher': 0,
                  'name_shapes': 'country'},
              1: {'name_column': 3,             # Level 1 are divisions
                  'name_column_higher': 2,
                  'name_shapes': 'division'},
              2: {'name_column': 6,             # Level 2 are districts
                  'name_column_higher': 4,
                  'name_shapes': 'district'},
              3: {'name_column': 9,             # Level 3 are upazillas
                  'name_column_higher': 7,
                  'name_shapes': 'upazilla'},
              4: {'name_column': 9,             # Level 4 are unions
                  'name_column_higher': 8,
                  'name_shapes': 'union'}}

# Set level for shape aggregation, choose from 0, 1, 2, 3 or 4
shp_level = 2

# Load shape information
shp_fn = home_dir + f'gadm41_BGD_shp/gadm41_BGD_{shp_level}.shp'
shp_file = shapefile.Reader(shp_fn)
shape_mask_dir = home_dir + f'shape_mask/{shp_level}/'
gpd_data = gpd.read_file(shp_fn)

shape_name = shape_info[shp_level]['name_shapes']
name_col = shape_info[shp_level]['name_column']
name_col_higher = shape_info[shp_level]['name_column_higher']

# Load district ID and name mapping
district_mapping = pd.read_csv(home_dir + 'gadm41_BGD_shp/bd_district_codes.csv')

# Generate a pandas table for the divisional values
shape_data = pd.DataFrame(index=range(len(gpd_data)), columns=[shape_name, 'tmax_week1', 'tmin_week1', 'tp_week1',
                                                               'tmax_week2', 'tmin_week2', 'tp_week2', 
                                                               'tmax_week3+4', 'tmin_week3+4', 'tp_week3+4'])
shape_data[shape_name] = gpd_data.iloc[:,name_col].values
shape_data = shape_data.set_index(shape_name)

# Generate a pandas table for the district output for agriculture
res_cols = ['district_id', 'indicator_variable','indicator_week','indicator_type','indicator_value']
results = pd.DataFrame(columns=res_cols)

# Make a dictionary for all models
model_info = {}

# Create an empty array for data integrated to divisions
fc_shp = {}

# Check until 5 days ago if there is data available, and run the forecast
# for the most recent data

for timedelta in range(7):
    modeldate = today - datetime.timedelta(timedelta)
    modeldatestr = modeldate.strftime("%Y%m%d")
    datestr = modeldate.strftime("%d%b").lower()
    
    # Make a forecast for each variable
    for var in varnames.keys():
        print(f'Start forecasts for {var}.')
    
        resample = varnames[var]['resample']
        varunit = varnames[var]['unit']
        
        input_dir = input_gen_dir + datestr + "/"
        
        ######################################################
        ##                                                  ##
        ##      Check data availability and load data       ##
        ##      NOTE THAT other models will be added        ##
        ##                                                  ##
        ######################################################
        
        # Try to open ECMWF data
        try:
            print('Load the regridded data')
            fn_hc = f"{input_dir}ecmwf_hc_regrid_{var}_{modeldatestr}.nc"
            fn_fc = f"{input_dir}ecmwf_fc_regrid_{var}_{modeldatestr}.nc"
            fn_obs = f"{input_dir}obs_ecmwf_{var}_{modeldatestr}.nc"
        
            obs_ecm = xr.open_dataarray(fn_obs)
            ecm_fc_daily = xr.open_dataarray(fn_fc)
            ecm_hc_daily = xr.open_dataarray(fn_hc)
            
            ecmwf_available = True
            model_info['ecmwf'] = {'forecast': ecm_fc_daily,
                                   'hindcast': ecm_hc_daily,
                                   'observations': obs_ecm}
            
            # Set ECMWF off if include_ecmwf == False
            if include_ecmwf == False:
                ecmwf_available = False
                
        except:
            ecmwf_available = False
            obs_ecm = None
            ecm_fc_daily = None
            ecm_hc_daily = None
            
        # Try to open ECCC data
        try:
            fn_hc = f"{input_dir}eccc_hc_regrid_{var}_{modeldatestr}.nc"
            fn_fc = f"{input_dir}eccc_fc_regrid_{var}_{modeldatestr}.nc"
            fn_obs = f"{input_dir}obs_eccc_{var}_{modeldatestr}.nc"
        
            obs_ecc = xr.open_dataarray(fn_obs)
            ecc_fc_daily = xr.open_dataarray(fn_fc)
            ecc_hc_daily = xr.open_dataarray(fn_hc)
            
            eccc_available = True
            model_info['eccc'] = {'forecast': ecc_fc_daily,
                                  'hindcast': ecc_hc_daily,
                                  'observations': obs_ecc}

            # Set ECCC off if include_eccc == False
            if include_eccc == False:
                eccc_available = False
                
        except:
            eccc_available = False
            obs_ecc = None
            ecc_fc_daily = None
            ecc_hc_daily = None

        # Try to open NCEP data
        try:
            fn_hc = f"{input_dir}ncep_hc_regrid_{var}_{modeldatestr}.nc"
            fn_fc = f"{input_dir}ncep_fc_regrid_{var}_{modeldatestr}.nc"
            fn_obs = f"{input_dir}obs_ncep_{var}_{modeldatestr}.nc"
        
            obs_cfsv2 = xr.open_dataarray(fn_obs)
            cfsv2_fc_daily = xr.open_dataarray(fn_fc)
            cfsv2_hc_daily = xr.open_dataarray(fn_hc)
            
            ncep_available = True
            model_info['ncep'] = {'forecast': cfsv2_fc_daily,
                                  'hindcast': cfsv2_hc_daily,
                                  'observations': obs_cfsv2}

            # Set CFSv2 off if include_cfsv2 == False
            if include_cfsv2 == False:
                cfsv2_available = False
        
        except:
            ncep_available = False
            obs_cfsv2 = None
            cfsv2_fc_daily = None
            cfsv2_hc_daily = None
            
        # Combine models
        print("Combining the models")
        obs_mm, fc_daily_mm, hc_daily_mm, models = s2s.combine_models(ecmwf_available, eccc_available, ncep_available,
                                                                      obs_ecm, obs_ecc, obs_cfsv2,
                                                                      ecm_fc_daily, ecc_fc_daily, cfsv2_fc_daily,
                                                                      ecm_hc_daily, ecc_hc_daily, cfsv2_hc_daily)
        
        # If there is no data available, continue
        if models == 'No data':
            logging.info(f'No data available for {modeldate}, continue')
            print(f'No data available for {modeldate}, continue')
            break
        
        if len(model_info) > 1:
            model_info['multi_model'] = {'forecast': fc_daily_mm,
                                         'hindcast': hc_daily_mm,
                                         'observations': obs_mm}
            
        if (len(model_info) == 1) and (single_model == False):
            logging.info('No multi-model forecast available for {modeldate}, continue')
            print('No multi-model: continue')
            continue
        else:
            logging.info('Perform single model forecast')
            print('Performing single model forecast')
    
        # Set mask to mask all data points outside of Bangladesh
        # Use the Tmax observations to load this mask
        if var == 'tmax':
            mask = np.isnan(obs_mm[0].values)
        print(f'Make the forecast for {models}')
        
        ################################
        ###    Spatial aggregation   ###
        ################################
        
        model_info_aggr = {}
        for hc_model in model_info.keys():
            
            obs = model_info[hc_model]['observations']
            fc_daily = model_info[hc_model]['forecast']
            hc_daily = model_info[hc_model]['hindcast']
                    
            shape_names, shapes, shape_ids = s2s.load_polygons(shp_file, name_col, name_col_higher, obs.Lat.values, obs.Lon.values, shape_mask_dir)
            
            # Create an empty array for the integrated data
            obs_shp =  np.zeros((obs.shape[:1] +( len(shape_names), 1)))
            fc_daily_shp =  np.zeros((fc_daily.shape[:2] +( len(shape_names), 1)))
            hc_daily_shp =  np.zeros((hc_daily.shape[:2] +( len(shape_names), 1)))
            
            print(f'Integrate the grid variables to polygons for {hc_model} data')
                
            # Loop over all the shape names
            # shapedirlist = os.listdir(shape_mask_dir)
            
            # shape_name_right_order = []
            # for maskfile in shapedirlist:
            #     shape_name_right_order.append(maskfile[11:-4])
            
            for (district_name, ss) in zip(shape_names, range(len(shape_names))):
                
                # Load the shape mask
                maskfile = 'shape_mask_' + district_name + '.npy'
                mask_2d = np.load(shape_mask_dir + maskfile, allow_pickle=True)
                
                input_grid_obs = obs[:].values
                input_grid_obs[np.isnan(input_grid_obs)] = 0.
                input_grid_fc_daily = fc_daily[:].values
                input_grid_fc_daily[np.isnan(input_grid_fc_daily)] = 0.
                input_grid_hc_daily = hc_daily[:].values
                input_grid_hc_daily[np.isnan(input_grid_hc_daily)] = 0.
                method = 'average'
                        
                obs_shp[:,ss,0] = s2s.integrate_grid_to_polygon(input_grid_obs, mask_2d, time_axis = 0, 
                                                               method = method)
                for member in range(fc_daily_shp.shape[1]):
                    fc_daily_shp[:,member,ss,0] = s2s.integrate_grid_to_polygon(input_grid_fc_daily[:,member,:,:], mask_2d, time_axis = 0, 
                                                                   method = method)
                for member in range(hc_daily_shp.shape[1]):
                    hc_daily_shp[:,member,ss,0] = s2s.integrate_grid_to_polygon(input_grid_hc_daily[:,member,:,:], mask_2d, time_axis = 0, 
                                                                   method = method)
                
            # Set back to xarray
            obs_shp = xr.DataArray(data = obs_shp,
                                   coords = (obs.time, range(obs_shp.shape[1]), range(obs_shp.shape[2])), 
                                   dims= ['time','latitude','longitude'])
            fc_daily_shp = xr.DataArray(data = fc_daily_shp,
                                   coords = (fc_daily.time, range(fc_daily_shp.shape[1]), range(fc_daily_shp.shape[2]), range(fc_daily_shp.shape[3])), 
                                   dims= ['time','member','latitude','longitude'])
            hc_daily_shp = xr.DataArray(data = hc_daily_shp,
                                   coords = (hc_daily.time, range(hc_daily_shp.shape[1]), range(hc_daily_shp.shape[2]), range(hc_daily_shp.shape[3])), 
                                   dims= ['time','member','latitude','longitude'])
            
            model_info_aggr[hc_model] = {'forecast': fc_daily_shp,
                                         'hindcast': hc_daily_shp,
                                         'observations': obs_shp}
        
        
        # Loop over the different periods (week 1, week 2 and week 3+4)
        for period in start_end_times.keys():
            print(f'Generate forecast for {period}.')
            wknr = start_end_times[period]['wknr']

            all_det_models = {}
            all_prob_models = {}
            
            # Do the hindcast skill analysis and forecast for all individual models
            for hc_model in model_info.keys():
                
                obs = model_info_aggr[hc_model]['observations']
                fc_daily = model_info_aggr[hc_model]['forecast']
                hc_daily = model_info_aggr[hc_model]['hindcast']
                
                ######################################################
                ##                                                  ##
                ##     Do short skill analysis of hindcast data     ##
                ##                                                  ##
                ######################################################

                if skill_analysis == True:
                    
                    # Set name of datadirectory for this run
                    fig_dir_hc = fig_dir_hc + datestr + "/"
                    if not os.path.exists(fig_dir_hc):
                        os.makedirs(fig_dir_hc)
                
                    # Resample data to only specific period
                    obs_wk, fc_wk, hc_wk = s2s.resample_data(obs, fc_daily, hc_daily, var,
                                                             start_end_times, period, resample)
                            
                    # Add members to obs_wk to be able to use it in X-Cast
                    obs_wk = obs_wk.expand_dims({"M":[0]})
                    obs_wk = obs_wk.transpose('time', 'M', 'latitude', 'longitude')
                    
                    # Add one time step to fc_wk to be able to use it in X-Cast
                    fc_wk = fc_wk.expand_dims({'time': [modeldate]})
                    
                    # Calculate BMD tercile categories
                    try:
                        # The old version of Xcast
                        bdohc = xc.RankedTerciles()
                    except:
                        # The new version of Xcast
                        bdohc = xc.OneHotEncoder()
                    bdohc.fit(obs_wk)
                    bd_ohc_wk = bdohc.transform(obs_wk)
                        
                    print('Start with cross validation on hindcasts of '+hc_model)
                    window = 3
                       
                    mlr_xval = []
                    mc_xval = []
                    
                    i_test=0
                    
                    # Do a cross validation of the hindcast models
                    for x_train, y_train, x_test, y_test in xc.CrossValidator(hc_wk, obs_wk, window=window, x_feature_dim='member'):
                        
                        try:
                            # The old version of Xcast
                            ohc_train = xc.RankedTerciles()
                        except:
                            # The new version of Xcast
                            ohc_train = xc.OneHotEncoder()
                        ohc_train.fit(y_train)
                        ohc_y_train = ohc_train.transform(y_train)
                        
                        # Use Quantile-Delta mapping for deterministic forecast
                        mlr_preds = s2s.calculate_qd_correction(y_train, x_test, x_train) # obs/fc/hc = y_train/x_text/x_train
                        mlr_xval.append(mlr_preds.isel(time=window // 2))
                        
                        try:
                            mc = xc.cMemberCount()
                            mc.fit(x_train.rename({'member': 'M'}), ohc_y_train, x_feature_dim='M')
                            mc_preds = mc.predict(x_test, x_feature_dim='member')
                            mc_xval.append(mc_preds.isel(time=window // 2))
                        except:
                            mc = xc.Ensemble()
                            mc.fit(x_train.rename({'member': 'M'}), ohc_y_train, x_feature_dim='M')
                            mc_preds = mc.predict_proba(x_test, x_feature_dim='member')
                            mc_xval.append(mc_preds.isel(time=window // 2))
                        
                        i_test += 1
                        print('Loop '+str(i_test)+' / '+str((len(obs_wk.time))))
                    
                    # Combine the cross validated output to a single dataset
                    try:
                        mlr_hcst = xr.concat(mlr_xval, 'time').mean('member').expand_dims({'M': [0]})
                    except:
                        mlr_hcst = xr.concat(mlr_xval, 'time').mean('ND')
                    mc_hcst_non_calibrated = xr.concat(mc_xval, 'time')
                    
                    # The mc_xval contain values that are slightly below 0
                    # These values need to be set at 0
                    mc_hcst = xr.DataArray(data=mc_hcst_non_calibrated.values,
                                           coords=mc_hcst_non_calibrated.coords)
                    mc_hcst.values[mc_hcst.values < 0] = 0.
                    
                    # Show skill of hindcasts
                    print('Calculate and plot skill scores of hindcast')
            
                    # Calculate and plot Pearson Correlation
                    try:
                        pearson = np.squeeze( xc.Pearson(mlr_hcst, obs_wk, x_feature_dim='M'), axis=[2,3])
                    except:
                        pearson = xc.Pearson(mlr_hcst, obs_wk, x_feature_dim='M').values
                    levels_pearson = np.linspace(-1,1,21)
                    s2s.plot_skill_score_aggregated(pearson, levels_pearson, 'RdBu', 'both', 
                                          f'{hc_model}'.upper(),
                                          fig_dir_hc,
                                          f'hc_{var}_pearson_{period}_{modeldatestr}_{hc_model}.png', shp_fn)
                    
                    # Calculate and plot Index of Agreement
                    try:
                        ioa = np.squeeze( xc.IndexOfAgreement(mlr_hcst, obs_wk, x_feature_dim='M'), axis=[2,3])
                    except:
                        ioa = xc.IndexOfAgreement(mlr_hcst, obs_wk, x_feature_dim='M').values
                    levels_ioa = np.linspace(0,1,11)
                    s2s.plot_skill_score_aggregated(ioa, levels_ioa, 'Blues', 'both',
                                          f'{hc_model}'.upper(),
                                          fig_dir_hc,
                                          f'hc_{var}_ioa_{period}_{modeldatestr}_{hc_model}.png', shp_fn)        
                    
                    # Calculate and plot Generalized ROC
                    try:
                        groc = np.squeeze(xc.GeneralizedROC(mc_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [2,3])
                    except:
                        groc = xc.GROCS(mc_hcst, bd_ohc_wk, x_feature_dim='member').values
                    levels_groc = np.linspace(0.5,1,11)
                    cmapg = plt.get_cmap('autumn_r').copy()
                    cmapg.set_under('lightgray')
                    s2s.plot_skill_score_aggregated(groc, levels_groc, cmapg, 'min',
                                          f'{hc_model}'.upper(),
                                          fig_dir_hc,
                                          f'hc_{var}_groc_{period}_{modeldatestr}_{hc_model}.png', shp_fn)  
                    
                    # Calculate and plot Rank Probability Skill Score
                    
                    # Get the climatological RPS for the skill score
                    climatological_odds = xr.ones_like(mc_hcst) * 0.333
                    
                    try:
                        skillscore_prec_wk = np.squeeze(xc.RankProbabilityScore( mc_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [2,3])
                        skillscore_climate_wk = np.squeeze(xc.RankProbabilityScore( climatological_odds, bd_ohc_wk, x_feature_dim='member'), axis=[2,3])
                    except:
                        skillscore_prec_wk = xc.RankProbabilityScore( mc_hcst, bd_ohc_wk, x_feature_dim='member').values
                        skillscore_climate_wk = xc.RankProbabilityScore( climatological_odds, bd_ohc_wk, x_feature_dim='member').values
                      
                    rpss = 1 - skillscore_prec_wk / skillscore_climate_wk
                    
                    levels_rpss = np.linspace(0.,0.2,11)
                    s2s.plot_skill_score_aggregated(rpss, levels_rpss, cmapg, 'both',
                                          f'{hc_model}'.upper(),
                                          fig_dir_hc,
                                          f'hc_{var}_rpss_{period}_{modeldatestr}_{hc_model}.png', shp_fn) 
                    
                    # Calculate and plot the Brier Skill Score for all 3 categories (AN/NN/BN)
                    try:
                        skillscore_prec_wk = np.squeeze(xc.BrierScore( mc_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [3])
                        skillscore_climate_wk = np.squeeze(xc.BrierScore( climatological_odds, bd_ohc_wk, x_feature_dim='member'), axis=[3])
              
                        bss = 1 - skillscore_prec_wk / skillscore_climate_wk
                        
                        levels_bss = np.linspace(0.,1,11)
                           
                        for idx, cat in zip([0,1,2],['BN','NN','AN']):
                            
                            data_plot = bss[:,:,idx]
                            
                            s2s.plot_skill_score_aggregated(data_plot, levels_bss, cmapg, 'min',
                                                  f'{hc_model}'.upper(),
                                                  fig_dir_hc,
                                                  f'hc_{var}_bss_{cat}_{period}_{modeldatestr}_{hc_model}.png'.lower(), shp_fn) 
                    except:
                        print("In new version of x-cast Brier Score calculation has changed")
                        #bd_ohc_wk = bd_ohc_wk.transpose("time", "M", "Lat", "Lon")
                        #skillscore_prec_wk = xc.BrierScore( mc_hcst, bd_ohc_wk, x_feature_dim='member').values
                        #skillscore_climate_wk = xc.BrierScore( climatological_odds, bd_ohc_wk, x_feature_dim='member').values

                
            
                ######################################################
                ##                                                  ##
                ##            Make operational forecast             ##
                ##                                                  ##
                ######################################################
                # Resample data to only specific period
                obs_wk, fc_wk, hc_wk = s2s.resample_data(obs, fc_daily, hc_daily, var,
                                                         start_end_times, period, resample)
                        
                # Add members to obs_wk to be able to use it in X-Cast
                obs_wk = obs_wk.expand_dims({"M":[0]})
                
                # Add one time step to fc_wk to be able to use it in X-Cast
                fc_wk = fc_wk.expand_dims({'time': [modeldate]})
                
                # Calculate BMD tercile categories
                try:
                    # The old version of XCast
                    bdohc = xc.RankedTerciles()
                except:
                    # The new version of Xcast
                    bdohc = xc.OneHotEncoder()
                    
                bdohc.fit(obs_wk)
                bd_ohc_wk = bdohc.transform(obs_wk)
                    
                print('Generate operational forecast')
                
                if len(model_info) > 1:
                    if hc_model != 'multi_model':
                        ## Calculate the multi-model forecast based on quantile-delta mapping
                        deterministic_forecast = s2s.calculate_qd_correction(obs_wk, fc_wk, hc_wk)
                        # The following IS NOT a smoothing (kernel = 0), the function is called only to have the data in the correct format
                        deterministic_fc_smooth = xc.gaussian_smooth(deterministic_forecast, x_sample_dim='time', x_feature_dim='member',  kernel=0)
                        # Perform the mean over the model members
                        deterministic_fc_smooth = deterministic_fc_smooth.mean('member')[0]
                        # deterministic_fc_smooth.values[mask] = np.nan
                        deterministic_anomaly = deterministic_fc_smooth - obs_wk.mean('time').mean('M')
                        all_det_models[hc_model] = deterministic_anomaly
                        
                        ## Calculate the probabilistic forecast based on member count
                        probabilistic_forecast = s2s.calculate_prob_forecast(obs_wk, fc_wk, hc_wk)
                        # The following IS NOT a smoothing (kernel = 0), the function is called only to have the data in the correct format
                        probabilistic_fc_smooth = xc.gaussian_smooth(probabilistic_forecast, x_sample_dim='time', x_feature_dim= 'member',  kernel=0)
                        # probabilistic_fc_smooth.values[:,:,mask] = np.nan
                        all_prob_models[hc_model] = probabilistic_fc_smooth
    
                    else:
                        # Combine the anomalies and probabilities for the multi-model forecast
                        all_anomalies = xr.full_like(deterministic_anomaly, 0.)
                        all_probabilities = xr.full_like(probabilistic_fc_smooth, 0.)
                        
                        starter = 1
                        for model in all_det_models.keys():
                            if starter == 1:
                                anom = np.expand_dims(all_det_models[model], axis=0)
                                low = np.expand_dims(all_prob_models[model][0], axis=0)
                                normal = np.expand_dims(all_prob_models[model][1], axis=0)
                                high = np.expand_dims(all_prob_models[model][2], axis=0)
                                starter = 0
                            else:
                                anom = np.append(anom,
                                                 np.expand_dims(all_det_models[model], axis=0),
                                                 axis=0)
                                low = np.append(low,
                                                np.expand_dims(all_prob_models[model][0], axis=0),
                                                axis=0)
                                normal = np.append(normal,
                                                   np.expand_dims(all_prob_models[model][1], axis=0),
                                                   axis=0)
                                high = np.append(high,
                                                 np.expand_dims(all_prob_models[model][2], axis=0),
                                                 axis=0)
                                
                        # Calculate the mean of the anomalies
                        all_anomalies.values = np.mean(anom, axis=0)
                        deterministic_anomaly = all_anomalies
                        
                        # Add the average observation for the deterministic value
                        deterministic_fc_smooth = all_anomalies + obs_wk.mean('time').mean('M')
                        
                        # Average the tercile probabilities and combine
                        all_probabilities.values[0] = np.mean(low, axis=0)
                        all_probabilities.values[1] = np.mean(normal, axis=0)
                        all_probabilities.values[2] = np.mean(high, axis=0)
                        
                        probabilistic_fc_smooth = all_probabilities
                else:
                    ## Calculate the multi-model forecast based on quantile-delta mapping
                    deterministic_forecast = s2s.calculate_qd_correction(obs_wk, fc_wk, hc_wk)
                    # The following IS NOT a smoothing (kernel = 0), the function is called only to have the data in the correct format
                    deterministic_fc_smooth = xc.gaussian_smooth(deterministic_forecast, x_sample_dim='time', x_feature_dim='member',  kernel=0)
                    # Perform the mean over the model members
                    deterministic_fc_smooth = deterministic_fc_smooth.mean('member')[0]
                    deterministic_anomaly = deterministic_fc_smooth - obs_wk.mean('time').mean('M')
                    
                    ## Calculate the probabilistic forecast based on member count
                    probabilistic_forecast = s2s.calculate_prob_forecast(obs_wk, fc_wk, hc_wk)
                    # The following IS NOT a smoothing (kernel = 0), the function is called only to have the data in the correct format
                    probabilistic_fc_smooth = xc.gaussian_smooth(probabilistic_forecast, x_sample_dim='time', x_feature_dim= 'member',  kernel=0)
                    
                # Plot the forecast
                fig_dir_fc = fig_dir_fc + datestr + "/"
                if not os.path.exists(fig_dir_fc):
                    os.makedirs(fig_dir_fc)
                
                s2s.plot_forecast_aggregated(var, period, deterministic_fc_smooth,
                                  deterministic_anomaly, probabilistic_fc_smooth,
                                  fig_dir_fc, hc_model, modeldatestr, 'sub-seasonal', shp_fn)

            
            ####################################################
            ####                                            ####
            ####   Save output only from the multi-model    ####
            ####                                            ####
            ####################################################
            
            # Calculate percentile of score based on uncalibrated data
            # The result will be the percentile where the ensemble mean forecast
            # falls into the big ensemble of hindcast values
            percentilescore = np.zeros((len(fc_wk.latitude),len(fc_wk.longitude)))
            for xx in range(np.shape(percentilescore)[0]):
                for yy in range(np.shape(percentilescore)[1]):
                    hcs = hc_wk[:,:,xx,yy].values.flatten()
                    hcs = hcs[~np.isnan(hcs)]
                    percentilescore[xx,yy] = percentileofscore(hcs,np.mean(fc_wk[:,:,xx,yy].values))
            
            
            print('Save forecast output')
            
            # Loop over all the shape names
            district_list = {}
            for (district_name, ss) in zip(shape_names, range(len(shape_names))):
                
                district_id = int(district_mapping[district_mapping['Name_GADM41'] == district_name]['District_no'].values[0])
                district_name_official = district_mapping[district_mapping['Name_GADM41'] == district_name]['Name_official'].values[0]

                district_list[str(district_id)] = district_name_official
                
                # Integrate the gridded value to a single value for this polygon
                if "cox" in district_name.lower():
                    key = var+wknr+'_'+district_name.lower()[:3]
                    key_ano = 'an'+var+wknr+'_'+district_name.lower()[:3]
                    key_bn = 'bn_'+var+wknr+'_'+district_name.lower()[:3]
                    key_nn = 'nn_'+var+wknr+'_'+district_name.lower()[:3]
                    key_an = 'an_'+var+wknr+'_'+district_name.lower()[:3]
                elif "feni" in district_name.lower():
                    key = var+wknr+'_'+district_name.lower()[:4]
                    key_ano = 'an'+var+wknr+'_'+district_name.lower()[:4]
                    key_bn = 'bn_'+var+wknr+'_'+district_name.lower()[:4]
                    key_nn = 'nn_'+var+wknr+'_'+district_name.lower()[:4]
                    key_an = 'an_'+var+wknr+'_'+district_name.lower()[:4]                    
                else:
                    key = var+wknr+'_'+district_name.lower()[:5]
                    key_ano = 'an'+var+wknr+'_'+district_name.lower()[:5] 
                    key_bn = 'bn_'+var+wknr+'_'+district_name.lower()[:5]
                    key_nn = 'nn_'+var+wknr+'_'+district_name.lower()[:5]
                    key_an = 'an_'+var+wknr+'_'+district_name.lower()[:5] 
                    
                                  
                # Integrate forecast and rank values
                fc_value = int(np.round(deterministic_fc_smooth.values[ss,0],0))
                ano_value = int(np.round(deterministic_anomaly.values[ss,0],0))
                bn_value = int(np.round(probabilistic_fc_smooth[0,0,ss,0].values,0))              
                nn_value = int(np.round(probabilistic_fc_smooth[1,0,ss,0].values,0)) 
                an_value = int(np.round(probabilistic_fc_smooth[2,0,ss,0].values,0))   
                rank_value = int(np.round(percentilescore[ss,0],0))
                
                # Integrate skill scores
                if skill_analysis == True:
                    skill_values = {}
                    for skill_value, skill_scores in (('pearson_value', pearson),
                                                      ('ioa_value', ioa),
                                                      ('groc_value', groc),
                                                      ('rpss_value', rpss)):#,
                                                      # ('bss_bn_value', bss[:,:,0]),
                                                      # ('bss_nn_value', bss[:,:,1]),
                                                      # ('bss_an_value', bss[:,:,2])):
                        

                        skill_scores[np.isnan(skill_scores)] = 0
                        skill_values[skill_value] = np.round(skill_scores[ss,0],0)


                
                # Set very low precipitation values at 0 mm
                if var == 'tp' and fc_value < 2:
                    fc_value = 0
                
                # Store value in json and dataframe
                fc_shp[key] = fc_value
                shape_data.at[district_name, var+'_'+period] = fc_value
                
                fc_shp[key_ano] = ano_value
                shape_data.at[district_name, 'an'+var+'_'+period] = ano_value
                
                fc_shp[key_bn] = bn_value
                shape_data.at[district_name, 'bn_'+var+'_'+period] = bn_value
                
                fc_shp[key_nn] = nn_value
                shape_data.at[district_name, 'nn_'+var+'_'+period] = nn_value
                
                fc_shp[key_an] = an_value
                shape_data.at[district_name, 'an_'+var+'_'+period] = an_value
                
                # Add value in the agro_results dictionary
                if period == 'week1' or period == 'week2':
                    if skill_analysis == True:
                        res_add = pd.DataFrame([[district_id, var, period, 'fc_value', fc_value],
                                                [district_id, var, period, 'ano_value', ano_value],
                                                [district_id, var, period, 'bn_value', bn_value],
                                                [district_id, var, period, 'nn_value', nn_value],
                                                [district_id, var, period, 'an_value', an_value],
                                                [district_id, var, period, 'rank_value', rank_value],
                                                [district_id, var, period, 'pearson_value', skill_values['pearson_value']],
                                                [district_id, var, period, 'ioa_value', skill_values['ioa_value']],
                                                [district_id, var, period, 'groc_value', skill_values['groc_value']],
                                                [district_id, var, period, 'rpss_value', skill_values['rpss_value']],
                                                # [district_id, var, period, 'bss_bn_value', skill_values['bss_bn_value']],
                                                # [district_id, var, period, 'bss_nn_value', skill_values['bss_nn_value']],
                                                # [district_id, var, period, 'bss_an_value', skill_values['bss_an_value']]
                                                ], 
                                               columns=res_cols)
                    else:
                        res_add = pd.DataFrame([[district_id, var, period, 'fc_value', fc_value],
                                                [district_id, var, period, 'ano_value', ano_value],
                                                [district_id, var, period, 'bn_value', bn_value],
                                                [district_id, var, period, 'nn_value', nn_value],
                                                [district_id, var, period, 'an_value', an_value],
                                                [district_id, var, period, 'rank_value', rank_value],
                                                # [district_id, var, period, 'bss_bn_value', skill_values['bss_bn_value']],
                                                # [district_id, var, period, 'bss_nn_value', skill_values['bss_nn_value']],
                                                # [district_id, var, period, 'bss_an_value', skill_values['bss_an_value']]
                                                ], 
                                               columns=res_cols)

                    results = pd.concat([results, res_add], ignore_index=True)
                elif period == 'week3+4':
                    for week in ['week3','week4']:
                        if skill_analysis == True:
                            res_add = pd.DataFrame([[district_id, var, week, 'fc_value', fc_value],
                                                    [district_id, var, week, 'ano_value', ano_value],
                                                    [district_id, var, week, 'bn_value', bn_value],
                                                    [district_id, var, week, 'nn_value', nn_value],
                                                    [district_id, var, week, 'an_value', an_value],                                                
                                                    [district_id, var, week, 'rank_value', rank_value],
                                                    [district_id, var, week, 'pearson_value', skill_values['pearson_value']],
                                                    [district_id, var, week, 'ioa_value', skill_values['ioa_value']],
                                                    [district_id, var, week, 'groc_value', skill_values['groc_value']],
                                                    [district_id, var, week, 'rpss_value', skill_values['rpss_value']],
                                                    # [district_id, var, week, 'bss_bn_value', skill_values['bss_bn_value']],
                                                    # [district_id, var, week, 'bss_nn_value', skill_values['bss_nn_value']],
                                                    # [district_id, var, week, 'bss_an_value', skill_values['bss_an_value']]
                                                    ], 
                                                   columns=res_cols)
                        else:
                            res_add = pd.DataFrame([[district_id, var, week, 'fc_value', fc_value],
                                                    [district_id, var, week, 'ano_value', ano_value],
                                                    [district_id, var, week, 'bn_value', bn_value],
                                                    [district_id, var, week, 'nn_value', nn_value],
                                                    [district_id, var, week, 'an_value', an_value],                                                
                                                    [district_id, var, week, 'rank_value', rank_value]
                                                    ], 
                                                   columns=res_cols)
                        results = pd.concat([results, res_add], ignore_index=True)
            
    # Save the data if there is data
    if len(model_info) > 1 or len(model_info) == 1 and single_model:
        
        logging.info('Save data and exit script')
        
        output_dir = output_dir + datestr + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the divisional output in a json file for the bulletin
        with open(output_dir+f'{shape_name}_forecast_{modeldatestr}.json', 'w') as jsfile:
            json.dump(fc_shp, jsfile)
        
        # Save the divisional output in a csv format to be used on BAMIS
        shape_data.to_csv(output_dir+f'{shape_name}_forecast_{modeldatestr}.csv')
        
        # Save the results for the agro-advisories in a csv for WEnR
        # results.to_csv(output_dir+f's2s_forecast_for_agro_{modeldatestr}.csv', index=False)
        res_js = results.to_dict(orient='records')
        if skill_analysis == True:
            agro_js = {
                 "metadata": {"description": "S2S forecast for Bangladesh based on a multi model ensemble.",
                              "model_combination": models,
                              "modeldate": str(modeldate),
                              "forecast_start": str(modeldate + datetime.timedelta(days=start_end_times['week1']['start'])),
                              "district_list": district_list,
                              "indicator_description": {"fc_value": "Deterministic forecast values",
                                                       "rank_value": "The ensemble average value expressed as percentile of the model climatology",
                                                       "pearson_value": "The pearson correlation coefficient based on hindcasts",
                                                       "ioa_value": "The index of agreement based on hindcasts",
                                                       "groc_value": "The generalized ROC based on hindcasts",
                                                       "rpss_value": "The Rank Probability Skill Score based on hindcats",
                                                       "bss_bn_value": "Brier Skill Score for the Below Normal category based on hindcasts",
                                                       "bss_nn_value": "Brier Skill Score for the Near Normal category based on hindcasts",
                                                       "bss_an_value": "Brier Skill Score for the Above Normal category based on hindcasts"}
                              },
                 "weather_variables": res_js
                }
        else:
            agro_js = {
                 "metadata": {"description": "S2S forecast for Bangladesh based on a multi model ensemble.",
                              "model_combination": models,
                              "modeldate": str(modeldate),
                              "forecast_start": str(modeldate + datetime.timedelta(days=start_end_times['week1']['start'])),
                              "district_list": district_list,
                              "indicator_description": {"fc_value": "Deterministic forecast values",
                                                       "rank_value": "The ensemble average value expressed as percentile of the model climatology"}
                              },
                 "weather_variables": res_js
                }
        for key, value in agro_js.items():
            if type(value) is int:
                continue
            elif isinstance(value, int):
                agro_js[key] = value
            elif isinstance(value, np.int64):
                agro_js[key] = int(value)
                
        with open(output_dir+f's2s_forecast_for_agro_{modeldatestr}.json', 'w') as jsfile:
            json.dump(agro_js, jsfile)
        
        # Exit the script
        sys.exit()
