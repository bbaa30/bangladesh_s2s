#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:01:33 2023

@author: bob
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import math
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.mpl.ticker as cticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from osgeo import osr,gdal
from pyproj import Proj, transform
import collections
import os
from shapely.geometry import shape, Polygon
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
import geopandas as gpd

def combine_models(ecmwf_available, eccc_available, ncep_available,
                   obs_ecm, obs_ecc, obs_cfsv2,
                   ecm_fc_daily, ecc_fc_daily, cfsv2_fc_daily,
                   ecm_hc_daily, ecc_hc_daily, cfsv2_hc_daily):
    '''
    Function that combines the multiple models to a single dataset.
    Currently only ECMWF and ECCC are implemented. Later on, other datasets
    from other centres will be included.
    
    Input:
        ecmwf_available: boolean: states if ecmwf data is available
        eccc_available: boolean: states if eccc data is available
        var: str: set the variable name
        obs_ecm: xr.DataArray: the observations matching the ECMWF time steps
        obs_ecc: xr.DataArray: the observations matching the ECCC time steps
        ecm_fc_daily: xr.DataArray: the ECMWF forecast
        ecc_fc_daily: xr.DataArray: the ECCC forecast
        ecm_hc_daily: xr.DataArray: the ECMWF hindcasts
        ecc_hc_daily: xr.DataArray: the ECCC hindcasts
    
    Output:
        obs: xr.DataArray: the observations matching the hindcast data timesteps
        fc_daily: xr.DataArray: the multi-model forecast
        hc_daily: xr.DataArray: the multi-model hindcasts
        models: str: string with the models that are used
    '''
    n_models = 0
    
    if ecmwf_available == True and eccc_available == True and ncep_available == True:
        # Combine the ECMWF, ECCC and NCEP data
        n_models = 3
    
        # Check for overlapping dates in the observations and hindcasts
        days_to_take_ecm = [(obs_ecm.time[ii].values in obs_ecc.time.values and 
                             obs_ecm.time[ii].values in obs_cfsv2.time.values) 
                            for ii in range(len(obs_ecm.time))]
        days_to_take_ecc = [(obs_ecc.time[ii].values in obs_ecm.time.values and 
                             obs_ecc.time[ii].values in obs_cfsv2.time.values) 
                            for ii in range(len(obs_ecc.time))]
        days_to_take_cfs = [(obs_cfsv2.time[ii].values in obs_ecm.time.values and 
                             obs_cfsv2.time[ii].values in obs_ecc.time.values) 
                            for ii in range(len(obs_cfsv2.time))]
        
        # And take the overlapping forecast time steps
        days_to_take_ecm_fc = [(ecm_fc_daily.time[ii].values in ecc_fc_daily.time.values and
                                ecm_fc_daily.time[ii].values in cfsv2_fc_daily.time.values)
                               for ii in range(len(ecm_fc_daily.time))]
        days_to_take_ecc_fc = [(ecc_fc_daily.time[ii].values in ecm_fc_daily.time.values and
                                ecc_fc_daily.time[ii].values in cfsv2_fc_daily.time.values)
                               for ii in range(len(ecc_fc_daily.time))]
        days_to_take_cfs_fc = [(cfsv2_fc_daily.time[ii].values in ecm_fc_daily.time.values and
                                cfsv2_fc_daily.time[ii].values in ecc_fc_daily.time.values)
                               for ii in range(len(cfsv2_fc_daily.time))]
        
        # Take out the overlapping dates
        obs_ecm = obs_ecm[days_to_take_ecm]
        obs_ecc = obs_ecc[days_to_take_ecc]
        obs_cfsv2 = obs_cfsv2[days_to_take_cfs]
        
        ecm_hc_daily = ecm_hc_daily[days_to_take_ecm]
        ecc_hc_daily = ecc_hc_daily[days_to_take_ecc]
        cfsv2_hc_daily = cfsv2_hc_daily[days_to_take_cfs]
        
        ecm_fc_daily = ecm_fc_daily[days_to_take_ecm_fc]
        ecc_fc_daily = ecc_fc_daily[days_to_take_ecc_fc]
        cfsv2_fc_daily = cfsv2_fc_daily[days_to_take_cfs_fc]
        
        # Set eccc members after ecmwf members to avoid conflicts with merging
        ecc_fc_daily = ecc_fc_daily.assign_coords(member=ecc_fc_daily['member']+ecm_fc_daily['member'][-1]+1)
        ecc_hc_daily = ecc_hc_daily.assign_coords(member=ecc_hc_daily['member']+ecm_hc_daily['member'][-1]+1)
        
        # Set cfsv2 members after eccc members
        cfsv2_fc_daily = cfsv2_fc_daily.assign_coords(member=cfsv2_fc_daily['member']+ecc_fc_daily['member'][-1]+1)
        cfsv2_hc_daily = cfsv2_hc_daily.assign_coords(member=cfsv2_hc_daily['member']+ecc_hc_daily['member'][-1]+1)
        
        # Merge datasets
        obs = obs_ecm.copy()
        fc_daily = xr.concat((ecm_fc_daily, ecc_fc_daily, cfsv2_fc_daily), dim='member')
        hc_daily = xr.concat([ecm_hc_daily, ecc_hc_daily, cfsv2_hc_daily], dim='member')
    
        # Remove old variables
        del obs_ecm, obs_ecc, ecm_fc_daily, ecc_fc_daily, ecm_hc_daily, ecc_hc_daily
        
        models = 'mm_ecmwf_ncep_eccc'
    
    elif ecmwf_available == True and eccc_available == True and ncep_available == False:
        fc1 = ecm_fc_daily.copy()
        fc2 = ecc_fc_daily.copy()
        hc1 = ecm_hc_daily.copy()
        hc2 = ecc_hc_daily.copy()
        obs1 = obs_ecm.copy()
        obs2 = obs_ecc.copy()
        
        n_models = 2
        models = 'mm_ecmwf_eccc'
    
    elif ecmwf_available == True and eccc_available == False and ncep_available == True:
        fc1 = ecm_fc_daily.copy()
        fc2 = cfsv2_fc_daily.copy()
        hc1 = ecm_hc_daily.copy()
        hc2 = cfsv2_hc_daily.copy()
        obs1 = obs_ecm.copy()
        obs2 = obs_cfsv2.copy()
        
        n_models = 2
        models = 'mm_ecmwf_ncep'

    elif ecmwf_available == False and eccc_available == True and ncep_available == True:
        fc1 = cfsv2_fc_daily.copy()
        fc2 = ecc_fc_daily.copy()
        hc1 = cfsv2_hc_daily.copy()
        hc2 = ecc_hc_daily.copy()
        obs1 = obs_cfsv2.copy()
        obs2 = obs_ecc.copy()
        
        n_models = 2
        models = 'mm_ncep_eccc'
    
    if n_models == 2:
    
        # Check for overlapping dates in the observations and hindcasts
        days_to_take_1 = [obs1.time[ii].values in obs2.time.values for ii in range(len(obs1.time))]
        days_to_take_2 = [obs2.time[ii].values in obs1.time.values for ii in range(len(obs2.time))]
        
        # And take the overlapping forecast time steps
        days_to_take_1_fc = [fc1.time[ii].values in fc2.time.values for ii in range(len(fc1.time))]
        days_to_take_2_fc = [fc2.time[ii].values in fc1.time.values for ii in range(len(fc2.time))]
        
        # Take out the overlapping dates
        obs1 = obs1[days_to_take_1]
        obs2 = obs2[days_to_take_2]

        hc1 = hc1[days_to_take_1]
        hc2 = hc2[days_to_take_2]
        
        fc1 = fc1[days_to_take_1_fc]
        fc2 = fc2[days_to_take_2_fc]
        
        # Set eccc members after ecmwf members to avoid conflicts with merging
        fc2 = fc2.assign_coords(member=fc2['member']+fc1['member'][-1]+1)
        hc2 = hc2.assign_coords(member=hc2['member']+hc1['member'][-1]+1)
        
        # Merge datasets
        obs = obs1.copy()
        fc_daily = xr.concat((fc1, fc2), dim='member')
        hc_daily = xr.concat([hc1, hc2], dim='member')
        
        # Remove old variables
        del obs_ecm, obs_ecc, ecm_fc_daily, ecc_fc_daily, ecm_hc_daily, ecc_hc_daily
                
    if ecmwf_available == True and eccc_available == False and ncep_available == False:
        
        n_models = 1
        
        # Only take ECMWF datasets
        obs = obs_ecm.copy()
        fc_daily = ecm_fc_daily.copy()
        hc_daily = ecm_hc_daily.copy()
        
        models = 'ecmwf'
        
    elif ecmwf_available == False and eccc_available == True and ncep_available == False:

        n_models = 1

        # Only take ECCC datasets
        obs = obs_ecc.copy()
        fc_daily = ecc_fc_daily.copy()
        hc_daily = ecc_hc_daily.copy()
    
        models = 'eccc'

    elif ecmwf_available == False and eccc_available == False and ncep_available == True:

        n_models = 1

        # Only take ECCC datasets
        obs = obs_cfsv2.copy()
        fc_daily = cfsv2_fc_daily.copy()
        hc_daily = cfsv2_hc_daily.copy()
    
        models = 'ncep'
        
    if n_models < 1:
        
        # Return Nones
        obs = None
        fc_daily = None
        hc_daily = None
        
        models = 'No data'
    
    return obs, fc_daily, hc_daily, models

def combine_seasonal_models(model_info, var):
    '''
    Function to combine multiple seasonal forecast models into multi-model ensemble.
    
    Input:
        model_info: dictionairy with for each model an xarray for forecast, 
                    hindcast and observations.
    
    Output:
        mm_fc: xr.DataArray with the multi-model forecast
        mm_hc: xr.DataArray with the multi-model hindcasts
        mm_obs: xr.DataArray with the observations with same timestep as hindcasts
        model_list: list with all the models that are used
    '''
    
    mm_fc = xr.DataArray([])
    mm_hc = xr.DataArray([])
    mm_obs = xr.DataArray([])
    model_list = []

    for model in model_info.keys():
        
        model_list.append(model)
        
        # Get the forecast of the single model
        single_fc = model_info[model]['forecast']
        single_hc = model_info[model]['hindcast']
        single_obs = model_info[model]['observations']
        
        single_fc.name = var
        single_hc.name = var
        single_obs.name = var
        
        if len(mm_fc) == 0:
            mm_fc = single_fc
            mm_hc = single_hc
            mm_obs = single_obs
        else:
            # Add the number of members of the previous mm_ensemble
            single_fc = single_fc.assign_coords(number=single_fc['number'] + mm_fc['number'][-1] + 1)
            single_hc = single_hc.assign_coords(number=single_hc['number'] + mm_hc['number'][-1] + 1)
                        
            # Merge the new model into the ensemble
            mm_fc = xr.merge((mm_fc, single_fc))
            mm_hc = xr.merge((mm_hc, single_hc))

            # Days to take
            days_to_take = [mm_obs.time.values[ii] in single_obs.time.values for ii in range(len(mm_obs.time))]            
            mm_obs = mm_obs[days_to_take]
        
    # Convert the xr.Dataset to xr.DataArray
    mm_fc = mm_fc.to_array()[0]
    mm_hc = mm_hc.to_array()[0]

    
    return mm_fc, mm_hc, mm_obs, model_list

def get_start_end_times_seasonal(month):
    '''
    The function provides for each month the start and end times of the different
    forecast months. It also provides a list with the keys
    '''
    start_end_times = {1: {'start': 0,
                           'end': 3},
                       2: {'start': 1,
                           'end': 4},
                       3: {'start': 2,
                           'end': 5},
                       4: {'start': 3,
                           'end': 6}}
    if month == 1:
        start_end_times[1]['months'] = 'JFM'
        start_end_times[2]['months'] = 'FMA'
        start_end_times[3]['months'] = 'MAM'
        start_end_times[4]['months'] = 'AMJ'
    elif month == 2:
        start_end_times[1]['months'] = 'FMA'
        start_end_times[2]['months'] = 'MAM'
        start_end_times[3]['months'] = 'AMJ'
        start_end_times[4]['months'] = 'MJJ'
    elif month == 3:
        start_end_times[1]['months'] = 'MAM'
        start_end_times[2]['months'] = 'AMJ'
        start_end_times[3]['months'] = 'MJJ'    
        start_end_times[4]['months'] = 'JJA'
    elif month == 4:
        start_end_times[1]['months'] = 'AMJ'
        start_end_times[2]['months'] = 'MJJ'    
        start_end_times[3]['months'] = 'JJA'
        start_end_times[4]['months'] = 'JAS'
        start_end_times[5] = {'start': 2,
                              'end': 6,
                              'months': 'JJAS'}
    elif month == 5:
        start_end_times[1]['months'] = 'MJJ'    
        start_end_times[2]['months'] = 'JJA'
        start_end_times[3]['months'] = 'JAS'
        start_end_times[4]['months'] = 'ASO'
        start_end_times[5] = {'start': 1,
                              'end': 5,
                              'months': 'JJAS'}
    elif month == 6:
        start_end_times[1]['months'] = 'JJA'
        start_end_times[2]['months'] = 'JAS'
        start_end_times[3]['months'] = 'ASO'
        start_end_times[4]['months'] = 'SON'    
        start_end_times[5] = {'start': 0,
                              'end': 4,
                              'months': 'JJAS'}
    elif month == 7:
        start_end_times[1]['months'] = 'JAS'
        start_end_times[2]['months'] = 'ASO'
        start_end_times[3]['months'] = 'SON' 
        start_end_times[4]['months'] = 'OND'
    elif month == 8:
        start_end_times[1]['months'] = 'ASO'
        start_end_times[2]['months'] = 'SON' 
        start_end_times[3]['months'] = 'OND'
        start_end_times[4]['months'] = 'NDJ'
    elif month == 9:
        start_end_times[1]['months'] = 'SON' 
        start_end_times[2]['months'] = 'OND'
        start_end_times[3]['months'] = 'NDJ'
        start_end_times[4]['months'] = 'DJF'
    elif month == 10:
        start_end_times[1]['months'] = 'OND'
        start_end_times[2]['months'] = 'NDJ'
        start_end_times[3]['months'] = 'DJF'
        start_end_times[4]['months'] = 'JFM' 
    elif month == 11:
        start_end_times[1]['months'] = 'NDJ'
        start_end_times[2]['months'] = 'DJF'
        start_end_times[3]['months'] = 'JFM' 
        start_end_times[4]['months'] = 'FMA'
    elif month == 12:
        start_end_times[1]['months'] = 'DJF'
        start_end_times[2]['months'] = 'JFM' 
        start_end_times[3]['months'] = 'FMA'
        start_end_times[4]['months'] = 'MAM'
    
    return start_end_times
        
def calculate_qd_correction(obs, fc, hc):
    '''
    Function to calculate a corrected forecast using quantile delta mapping
    
    Input:
        obs: observations
        fc: forecast
        hc: hindcast
    
    Output:
        deterministic_fc: deterministic output
    
    '''
    # Perform the bias correction with observations based on quantile to quantile mapping
    fc_bc = xr.full_like(fc, 0.)
    fc_bias_corrected = np.zeros(np.shape(fc))
    
    for tt in range(fc_bias_corrected.shape[0]):
        for xx in range(fc_bias_corrected.shape[2]):
            for yy in range(fc_bias_corrected.shape[3]):
                obs_i = obs[tt,:,xx,yy].values
                hc_i = hc[tt,:,xx,yy].values
                fc_i = fc[tt,:,xx,yy].values
                
                fc_bias_corrected[tt,:,xx,yy] = q_q_map(obs_i, hc_i, fc_i, nbins=51)
    
    fc_bc.data = fc_bias_corrected
    
    return fc_bc

def calculate_prob_forecast(obs, fc, hc):
    '''
    Function to calculate a corrected forecast using quantile delta mapping
    
    Input:
        obs: observations
        fc: forecast
        hc: hindcast
    
    Output:
        probabilistic_fc: probabilistic output (already in %)
    
    '''
    
    # Calculate climate percentiles
    N_ens = len(fc['member'])
    p33 = hc.quantile(q=0.33, dim=('member','time'))
    p67 = hc.quantile(q=0.67, dim=('member','time'))
    
    # Calculate the likelihood for all terciles
    low = (fc<p33).sum('member') / N_ens * 100.
    normal = ((fc >= p33) * (fc <= p67)).sum('member') / N_ens * 100.
    high = (fc>p67).sum('member') / N_ens * 100.
    
    # Change back to xarray
    prob_fc = xr.DataArray([low,normal,high], coords={'member': ['BN','NN','AN'],
                                                      'time': fc.time,
                                                      'latitude': fc.latitude,
                                                      'longitude': fc.longitude})
    
    prob_fc = prob_fc.transpose("time", "member", "latitude", "longitude")
    
    return prob_fc
    
        
def q_q_map(obs, p, s, nbins=10, extrapolate=None):
    '''
    Function to perform a quantile to quantile mapping from the forecast to the
    CHIRPS climatology. 
    
    Input:
        obs: 1D array with climatology of CHIRPS for single location and month
        p: 1D array with climatoloty of the forecast model for single location and month
        s: 1D array of the current forecast for single location and month
        nbins: int: the amount of output bins
        extrapolate: None or 'constant'. With None, take the highest value from
                     the climatology if the value is out of range. With 'contant'
                     take a linear extrapolation
    
    Output:
        c: 1D array of current bias corrected forecast using quantile to 
           quantile mapping for single location and month
    '''
    
    binmid = np.arange((1./nbins)*0.5, 1., 1./nbins)
    qo = mquantiles(obs[np.isfinite(obs)], prob=binmid)
    qp = mquantiles(p[np.isfinite(p)], prob=binmid)
    p2o = interp1d(qp, qo, kind='linear', bounds_error=False)
    c = p2o(s)
    if extrapolate is None:
        c[s > np.max(qp)] = qo[-1]
        c[s < np.min(qp)] = qo[0]
    elif extrapolate == 'constant':
        c[s > np.max(qp)] = s[s > np.max(qp)] + qo[-1] - qp[-1]
        c[s < np.min(qp)] = s[s < np.min(qp)] + qo[0] - qp[0]
    
    return c

def resample_data(obs, fc_daily, hc_daily, var, start_end_times, period, resample):
    '''
    Function to resample the data (observations, forecast and hindcast ) to 
    weekly values, based on the resample method for the variable and the start
    and end times.
    
    Input:
        obs: xr.DataArray: the daily observations
        fc_daily: xr.DataArray: the daily forecast data
        hc_daily: xr.DataArray: the daily hindcast data
        var: str: the variable name
        start_end_times: dict: dictionary with start and end times (in days) per period
        period: str: period to resample to
        resample: str: resample method. Can be max, min, mean or sum
    
    Output:
        obs_wk: xr.DataArray: weekly observational values
        fc_wk: xr.DataArray: weekly forecast values
        hc_wk: xr.DataArray: weekly hindcast values
    '''
    
    # Select and agregate to period
    start = start_end_times[period]['start']
    end = start_end_times[period]['end']
    
    len_yr = len(fc_daily)
    n_years = int(len(obs)/len(fc_daily))
    
    if resample == 'max':
        fc_wk = fc_daily[start:end].max('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].max('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].max('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))
    
    elif resample == 'min':
        fc_wk = fc_daily[start:end].min('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].min('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].min('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))

    elif resample == 'mean':
        fc_wk = fc_daily[start:end].mean('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].mean('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].mean('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))
                
    elif resample == 'sum':
        fc_wk = fc_daily[start:end].sum('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].sum('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].sum('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))
    else:
        raise(Exception(f'Unkown resample type {resample}'))

    obs_wk = obs_wk.to_array()[0]
    hc_wk = hc_wk.to_array()[0]
    
    return obs_wk, fc_wk, hc_wk
        
# Define a function to plot skill scores
def plot_skill_score(value, obs, levels, cmap, extend, title, fig_dir, filename):
    '''
    Plot the skill scores of hindcast skill analysis.
    
    Input:
        value: np.array: array with the values to plot
        obs: xr.DataArray: the observational data, will be used for the coordinates
        levels: list: the levels to show in the figrue
        cmap: str or mpl.colormap: the colormap to use
        extend: str: extend option of matplotlib
        title: str: title of the figure
        fig_dir: str: directory to store the figure
        filename: str: filename of the figure
    
    Output:
        The figure is stored under filename in fig_dir
    '''


    plt.figure(figsize=(10,8.5))

    # Set the axes using the specified map projection
    ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([87,93,20,27])
     
    # Make a filled contour plot
    cs=ax.contourf(obs['Lon'], obs['Lat'], value,
                   transform = ccrs.PlateCarree(),levels = levels, cmap=cmap, extend=extend)
  
    cbar = plt.colorbar(cs, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    
    # Add coastlines
    ax.coastlines()
    ax.add_feature(cf.BORDERS)
    
    # Define the xticks for longitude
    ax.set_xticks(np.arange(87,93,2), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    
    # Define the yticks for latitude
    ax.set_yticks(np.arange(20,27,2), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    plt.title(title, fontsize=25)
    plt.tight_layout()
    plt.savefig(fig_dir + filename)
    
# Define a function to plot skill scores
def plot_skill_score_aggregated(value, levels, cmap, extend, title, fig_dir, filename, shp_fn):
    '''
    Plot the skill scores of hindcast skill analysis, from aggregated data
    
    Input:
        value: np.array: array with the values to plot
        levels: list: the levels to show in the figure
        cmap: str or mpl.colormap: the colormap to use
        extend: str: extend option of matplotlib
        title: str: title of the figure
        fig_dir: str: directory to store the figure
        filename: str: filename of the figure
        shp_fn: str: path and name of the .shp file, containing the shape data
    
    Output:
        The figure is stored under filename in fig_dir
    '''
    gpd_data = gpd.read_file(shp_fn)
    
    gpd_data['value'] = value[:,0]

    plt.figure(figsize=(10,8.5))

    # Set the axes using the specified map projection
    ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([87,93,20,27])
    
    # Generate descrete colormaps
    cmap_d = plt.get_cmap(cmap).copy()
    norm = plt.Normalize(vmin=levels[0],vmax=levels[-1])
    cmap_descrete = mpl.colors.ListedColormap(cmap_d(norm(levels[:-1])))
    cmap_descrete.set_bad(color='grey')
     
    # Make a filled contour plot
    cs = gpd_data.plot(column='value', ax=ax, cmap=cmap_descrete, norm=norm)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_descrete, norm=norm)
    sm.set_array([])
  
    cbar = plt.colorbar(sm, ax=ax, extend=extend)
    cbar.ax.tick_params(labelsize=18)
    
    # Add coastlines
    ax.coastlines()
    ax.add_feature(cf.BORDERS)
    
    # Define the xticks for longitude
    ax.set_xticks(np.arange(87,93,2), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.xaxis.set_tick_params(labelsize=14)
    
    # Define the yticks for latitude
    ax.set_yticks(np.arange(20,27,2), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_tick_params(labelsize=14)

    plt.title(title, fontsize=25)
    plt.tight_layout()
    plt.savefig(fig_dir + filename)

def plot_forecast(var, period, deterministic_fc_smooth, deterministic_anomaly, 
                  probabilistic_fc_smooth, fig_dir_fc, model, modeldatestr,
                  fc_type):
    '''
    Make a figure with the deterministic, anomaly and probabilistic forecast 
    as png and as eps format.
    
    Input:
        var: str: variable code
        deterministic_fc_smooth: xr.DataArray: the smoothed deterministic forecast
        deterministic_anomaly: xr.DataArray: the deterministic anomaly forecast
        probabilictic_fc_smooth: xr.DataArray: the smoothed probabilistic forecast
        fig_dir_fc: str: the directory to store the figures
        model: str: name of the model
        modeldatestr: str: string with the modeldate
        fc_type: str: 'seasonal' or 'sub-seasonal'
    
    Ouput:
        2 figures with the forecast are stored in fig_dir_fc:
            a png figure
            a eps figure (vector format)
    '''
    
    # Set the metadata for the figures
    if var == 'tmin':
        cmap_det = 'gist_ncar'
        cmap_anom = 'RdBu_r'
        cmap_below = 'Blues'
        cmap_above = 'YlOrRd'
        norm_det = mpl.colors.Normalize(vmin=0, vmax=55)
        levels_det = np.linspace(5,30,26)
        ticks_det = np.linspace(5,30,6)
        levels_anom = np.linspace(-6,6,25)
        ticks_anom = np.linspace(-6,6,5)
        label_det = u'Minimum temperature (\N{DEGREE SIGN}C)'
        label_anom = u'Temperature anomaly (\N{DEGREE SIGN}C)'
    elif var == 'tmax':
        cmap_det = 'gist_ncar'
        cmap_anom = 'RdBu_r'
        cmap_below = 'Blues'
        cmap_above = 'YlOrRd'
        norm_det = mpl.colors.Normalize(vmin=0, vmax=55)
        levels_det = np.linspace(15,45,31)
        ticks_det = np.linspace(15,45,7)
        levels_anom = np.linspace(-6,6,25)
        ticks_anom = np.linspace(-6,6,5)
        label_det = u'Maximum temperature (\N{DEGREE SIGN}C)'
        label_anom = u'Temperature anomaly (\N{DEGREE SIGN}C)'
    elif var == 'tp':
        # Set negative values to 0
        deterministic_fc_smooth.values[deterministic_fc_smooth.values <= 0.] = 0.
        
        # Make the maximum of precipitation dyanmic
        max_tp = np.nanmax(deterministic_fc_smooth)
        
        # Ceil value up to next 10
        max_tp = math.ceil(max_tp/10)*10
        
        if max_tp < 30:
            # Set the maximum at 30 mm if the maximum is lower
            max_tp = 30.
        
        norm_det = mpl.colors.Normalize(vmin=0, vmax=max_tp)
        cmap_det = cmocean.cm.haline_r
        cmap_anom = 'BrBG'
        cmap_below = 'YlOrRd'
        cmap_above = 'Greens'
        levels_det = np.linspace(0,max_tp,17)
        ticks_det = np.linspace(0,max_tp,5)
        if fc_type == 'sub-seasonal':
            levels_anom = np.linspace(-50,50,21)
            ticks_anom = np.linspace(-50,50,5)
        elif fc_type == 'seasonal':
            levels_anom = np.linspace(-200,200,21)
            ticks_anom = np.linspace(-200,200,5)
        label_det = 'Precipitation (mm)'
        label_anom = 'Precipitation anomaly (mm)'

    # Preprocess the probabilistic data
    if float(probabilistic_fc_smooth.max()) < 1.:
        bn_fc = 100*probabilistic_fc_smooth[0,0]
        nn_fc = 100*probabilistic_fc_smooth[1,0]
        an_fc = 100*probabilistic_fc_smooth[2,0]
    else:
        bn_fc = probabilistic_fc_smooth[0,0]
        nn_fc = probabilistic_fc_smooth[1,0]
        an_fc = probabilistic_fc_smooth[2,0]
    
    levels_outer = np.linspace(40,80,9)
    levels_inner = np.linspace(40,50,3)
    
    cmap_nn = plt.get_cmap('Greys').copy()
    cmap_nn.set_over('lightgray')
    norm_nn = mpl.colors.Normalize(vmin=30,vmax=100)
    
    # Make the figure
    fig, axes = plt.subplots(1,3, figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set the axes using the specified map projection
    for ax in axes:
        ax.set_extent([88.0,92.7,20.6,26.7])
        ax.set_axis_off()
        ax.coastlines(zorder=5)
        ax.add_feature(cf.BORDERS, zorder=5)
     
    # Make a filled contour plot
    cs0=axes[0].contourf(deterministic_fc_smooth['longitude'], deterministic_fc_smooth['latitude'], deterministic_fc_smooth,
                         transform = ccrs.PlateCarree(), levels = levels_det, 
                         norm=norm_det, cmap=cmap_det, extend='both')

    cax0 = inset_axes(axes[0], width='100%', height='5%', loc='lower left', 
                      bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[0].transAxes, borderpad=0.1)
    cbar0 = plt.colorbar(cs0, ax=axes[0], cax=cax0, orientation='horizontal', ticks=ticks_det)
    cbar0.set_label(label_det, fontsize=14)
    cbar0.ax.tick_params(labelsize=12)

    # Make a filled contour plot
    cs1=axes[1].contourf(deterministic_anomaly['longitude'], deterministic_anomaly['latitude'], deterministic_anomaly,
                         transform = ccrs.PlateCarree(), levels = levels_anom, 
                         cmap=cmap_anom, extend='both')
    cax1 = inset_axes(axes[1], width='100%', height='5%', loc='lower left', 
                      bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[1].transAxes, borderpad=0.1)
    cbar1 = plt.colorbar(cs1, ax=axes[1], cax=cax1, orientation='horizontal', ticks=ticks_anom)
    cbar1.set_label(label_anom, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Make a filled contour plot
    cs_nn=axes[2].contourf(nn_fc['longitude'], nn_fc['latitude'], nn_fc, transform = ccrs.PlateCarree(),
                           levels = levels_inner, cmap=cmap_nn, norm=norm_nn, extend='max')
    cs_bn=axes[2].contourf(bn_fc['longitude'], bn_fc['latitude'], bn_fc, transform = ccrs.PlateCarree(),
                           levels = levels_outer, cmap=cmap_below, extend='max')
    cs_an=axes[2].contourf(an_fc['longitude'], an_fc['latitude'], an_fc, transform = ccrs.PlateCarree(),
                           levels = levels_outer, cmap=cmap_above, extend='max')
    
    cax2a = inset_axes(axes[2], width='35%', height='5%', loc='lower left', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.1)
    cax2b = inset_axes(axes[2], width='20%', height='5%', loc='lower center', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.05)
    cax2c = inset_axes(axes[2], width='35%', height='5%', loc='lower right', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.1)
    
    cbar2a = plt.colorbar(cs_bn, ax=axes[2], cax=cax2a, orientation='horizontal', ticks=[40,60,80])
    cbar2b = plt.colorbar(cs_nn, ax=axes[2], cax=cax2b, orientation='horizontal', ticks=[40,50])
    cbar2c = plt.colorbar(cs_an, ax=axes[2], cax=cax2c, orientation='horizontal', ticks=[40,60,80])

    cbar2a.ax.tick_params(labelsize=12)
    cbar2b.ax.tick_params(labelsize=12)
    cbar2c.ax.tick_params(labelsize=12)
    
    cbar2a.set_label('BN (%)', fontsize=14)
    cbar2b.set_label('NN (%)', fontsize=14)
    cbar2c.set_label('AN (%)', fontsize=14)

    axes[0].set_title('Deterministic forecast', fontsize=18)
    axes[1].set_title('Forecast anomaly', fontsize=18)
    axes[2].set_title('Probabilistic forecast', fontsize=18)

    plt.subplots_adjust(wspace=0.025, hspace=0.025, bottom=0.15)
    plt.tight_layout()
    plt.savefig(fig_dir_fc + f'fc_{var}_{period}_{model}_{modeldatestr}.eps', format='eps', bbox_inches='tight')    
    plt.savefig(fig_dir_fc + f'fc_{var}_{period}_{model}_{modeldatestr}.png', format='png', bbox_inches='tight')
    
def plot_forecast_aggregated(var, period, deterministic_fc_smooth, deterministic_anomaly, 
                  probabilistic_fc_smooth, fig_dir_fc, model, modeldatestr,
                  fc_type, shp_fn):
    '''
    Make a figure with the deterministic, anomaly and probabilistic forecast 
    as png and as eps format.
    
    Input:
        var: str: variable code
        period: str: lead time of the fcst
        deterministic_fc_smooth: xr.DataArray: the smoothed deterministic forecast
        deterministic_anomaly: xr.DataArray: the deterministic anomaly forecast
        probabilictic_fc_smooth: xr.DataArray: the smoothed probabilistic forecast
        fig_dir_fc: str: the directory to store the figures
        model: str: name of the model
        modeldatestr: str: string with the modeldate
        fc_type: str: 'seasonal' or 'sub-seasonal'
        shp_fn: str: path and name of the .shp file, containing the shape data
    
    Ouput:
        2 figures with the forecast are stored in fig_dir_fc:
            a png figure
            a eps figure (vector format)
    '''
    
    gpd_data = gpd.read_file(shp_fn)
    
    maxval = np.nanmax(deterministic_fc_smooth)
    minval = np.nanmin(deterministic_fc_smooth)
    
    # Set the metadata for the figures
    if var == 'tmin':
        cmap_det = 'Reds'
        cmap_anom = 'RdBu_r'
        cmap_below = 'Blues'
        cmap_above = 'YlOrRd'
        
        min_tmin = np.floor(minval) - 1
        max_tmin = np.ceil(maxval) + 1
        dt = max_tmin - min_tmin + 1
        
        levels_det = np.linspace(min_tmin,max_tmin,dt)
        ticks_det = np.linspace(min_tmin,max_tmin,dt)
        levels_anom = np.linspace(-6,6,25)
        ticks_anom = np.linspace(-6,6,5)
        label_det = u'Minimum temperature (\N{DEGREE SIGN}C)'
        label_anom = u'Temperature anomaly (\N{DEGREE SIGN}C)'
        cbar_fr_min = 0.1
        cbar_fr_max = 0.6
    elif var == 'tmax':
        cmap_det = 'Reds'
        cmap_anom = 'RdBu_r'
        cmap_below = 'Blues'
        cmap_above = 'YlOrRd'
        
        min_tmax = np.floor(minval) - 1
        max_tmax = np.ceil(maxval) + 1
        dt = max_tmax - min_tmax + 1
        
        levels_det = np.linspace(min_tmax,max_tmax,dt)
        ticks_det = np.linspace(min_tmax,max_tmax,dt)
        levels_anom = np.linspace(-6,6,25)
        ticks_anom = np.linspace(-6,6,5)
        label_det = u'Maximum temperature (\N{DEGREE SIGN}C)'
        label_anom = u'Temperature anomaly (\N{DEGREE SIGN}C)'
        cbar_fr_min = 0.3
        cbar_fr_max = 0.85
    elif var == 'tp':
        # Set negative values to 0
        deterministic_fc_smooth.values[deterministic_fc_smooth.values <= 0.] = 0.
        
        # Make the maximum of precipitation dyanmic
        if maxval <= 40:
            max_tp = 40.
        elif maxval > 40 and maxval <= 80:
            max_tp = 80.
        elif maxval > 80 and maxval <= 120:
            max_tp = 120.
        elif maxval > 120 and maxval <= 200:
            max_tp = 200.
        elif maxval > 200 and maxval <= 300:
            max_tp = 300.
        elif maxval > 300 and maxval <= 400:
            max_tp = 400.
        elif maxval > 400 and maxval <= 600:
            max_tp = 600.
        else:
            max_tp = 1000.
        
        norm_det = mpl.colors.Normalize(vmin=0, vmax=max_tp)
        cmap_det = cmocean.cm.haline_r
        cmap_anom = 'BrBG'
        cmap_below = 'YlOrRd'
        cmap_above = 'Greens'
        levels_det = np.linspace(0,max_tp,17)
        ticks_det = np.linspace(0,max_tp,5)
        if fc_type == 'sub-seasonal':
            levels_anom = np.linspace(-50,50,21)
            ticks_anom = np.linspace(-50,50,5)
        elif fc_type == 'seasonal':
            levels_anom = np.linspace(-200,200,21)
            ticks_anom = np.linspace(-200,200,5)
        label_det = 'Precipitation (mm)'
        label_anom = 'Precipitation anomaly (mm)'
        cbar_fr_min = 0
        cbar_fr_max = 1

    # Preprocess the probabilistic data
    if float(probabilistic_fc_smooth.max()) < 1.:
        bn_fc = 100*probabilistic_fc_smooth[0,0]
        nn_fc = 100*probabilistic_fc_smooth[1,0]
        an_fc = 100*probabilistic_fc_smooth[2,0]
    else:
        bn_fc = probabilistic_fc_smooth[0,0]
        nn_fc = probabilistic_fc_smooth[1,0]
        an_fc = probabilistic_fc_smooth[2,0]
    
    levels_outer = np.linspace(40,80,9)
    levels_inner = np.linspace(40,50,3)
    
    cmap_nn = plt.get_cmap('Greys').copy()
    cmap_nn.set_over('lightgray')
    
    # Generate descrete colormaps
    cmap_det_d = plt.get_cmap(cmap_det).copy()
    cmap_det_dnew = cmap_det_d(np.linspace(cbar_fr_min, cbar_fr_max, 100))
    norm_det = plt.Normalize(vmin=levels_det[0],vmax=levels_det[-1])
    cmap_det_descrete_rest = mpl.colors.LinearSegmentedColormap.from_list('cmap_det_descrete', cmap_det_dnew)
    cmap_det_descrete = mpl.colors.ListedColormap(cmap_det_descrete_rest(norm_det(levels_det[:-1])))
    cmap_det_descrete.set_bad(color='grey')
    
    cmap_anom_d = plt.get_cmap(cmap_anom).copy()
    norm_anom = plt.Normalize(vmin=levels_anom[0],vmax=levels_anom[-1])
    cmap_anom_descrete = mpl.colors.ListedColormap(cmap_anom_d(norm_anom(levels_anom[:-1])))
    cmap_anom_descrete.set_bad(color='grey')
    
    cmap_an_d = plt.get_cmap(cmap_above).copy()
    norm_an = plt.Normalize(vmin=levels_outer[0],vmax=levels_outer[-1])
    cmap_an_descrete = mpl.colors.ListedColormap(cmap_an_d(norm_an(levels_outer[:-1])))
    cmap_an_descrete.set_bad(color='grey')
    
    cmap_nn_d = plt.get_cmap(cmap_nn).copy()
    norm_nn = mpl.colors.Normalize(vmin=levels_inner[0],vmax=levels_inner[-1])
    cmap_nn_descrete = mpl.colors.ListedColormap(cmap_nn_d(norm_an(levels_inner[:-1])))
    cmap_nn_descrete.set_bad(color='grey')
    
    cmap_bn_d = plt.get_cmap(cmap_below).copy()
    norm_bn = plt.Normalize(vmin=levels_outer[0],vmax=levels_outer[-1])
    cmap_bn_descrete = mpl.colors.ListedColormap(cmap_bn_d(norm_bn(levels_outer[:-1])))
    cmap_bn_descrete.set_bad(color='grey')
    
    # Insert aggregated data in geopandas
    
    gpd_data['deterministic'] = deterministic_fc_smooth[:,0].values
    gpd_data['anomaly'] = deterministic_anomaly[:,0].values
    # Set the values that need to be shown as the cat. > 33%
    bn_fc_masked = xr.where(bn_fc[:,0] > 40, bn_fc[:,0], np.nan)
    nn_fc_masked = xr.where(nn_fc[:,0] > 40, nn_fc[:,0], np.nan)
    an_fc_masked = xr.where(an_fc[:,0] > 40, an_fc[:,0], np.nan)
    gpd_data['bn_fc'] = bn_fc_masked.values
    gpd_data['nn_fc'] = nn_fc_masked.values
    gpd_data['an_fc'] = an_fc_masked.values
    
    # Make the figure
    fig, axes = plt.subplots(1,3, figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set the axes using the specified map projection
    for ax in axes:
        ax.set_extent([88.0,92.7,20.6,26.7])
        ax.set_axis_off()
        ax.coastlines(zorder=5)
        ax.add_feature(cf.BORDERS, zorder=5)
     
    # Make a filled contour plot
    gpd_data.plot(column='deterministic', ax=axes[0], cmap=cmap_det_descrete, norm=norm_det)
    sm_det = plt.cm.ScalarMappable(cmap=cmap_det_descrete, norm=norm_det)
    

    cax0 = inset_axes(axes[0], width='100%', height='5%', loc='lower left', 
                      bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[0].transAxes, borderpad=0.1)
    cbar0 = plt.colorbar(sm_det, ax=axes[0], cax=cax0, orientation='horizontal', ticks=ticks_det, extend='both')
    cbar0.set_label(label_det, fontsize=14)
    cbar0.ax.tick_params(labelsize=12)

    # Make a filled contour plot
    gpd_data.plot(column='anomaly', ax=axes[1], cmap=cmap_anom_descrete, norm=norm_anom)
    sm_anom = plt.cm.ScalarMappable(cmap=cmap_anom_descrete, norm=norm_anom)

    cax1 = inset_axes(axes[1], width='100%', height='5%', loc='lower left', 
                      bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[1].transAxes, borderpad=0.1)
    cbar1 = plt.colorbar(sm_anom, ax=axes[1], cax=cax1, orientation='horizontal', ticks=ticks_anom, extend='both')
    cbar1.set_label(label_anom, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Make a filled contour plot
    gpd_data.plot(column='nn_fc', ax=axes[2], cmap=cmap_nn_descrete, norm=norm_nn)
    gpd_data.plot(column='an_fc', ax=axes[2], cmap=cmap_an_descrete, norm=norm_an)
    gpd_data.plot(column='bn_fc', ax=axes[2], cmap=cmap_bn_descrete, norm=norm_bn)
    
    sm_an = plt.cm.ScalarMappable(cmap=cmap_bn_descrete, norm=norm_bn)
    sm_nn = plt.cm.ScalarMappable(cmap=cmap_nn_descrete, norm=norm_nn)
    sm_bn = plt.cm.ScalarMappable(cmap=cmap_an_descrete, norm=norm_an)
    
    cax2a = inset_axes(axes[2], width='35%', height='5%', loc='lower left', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.1)
    cax2b = inset_axes(axes[2], width='15%', height='5%', loc='lower center', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.05)
    cax2c = inset_axes(axes[2], width='35%', height='5%', loc='lower right', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.1)
    
    cbar2a = plt.colorbar(sm_an, ax=axes[2], cax=cax2a, orientation='horizontal', ticks=[40,60,80], extend='max')
    cbar2b = plt.colorbar(sm_nn, ax=axes[2], cax=cax2b, orientation='horizontal', ticks=[40,50], extend='max')
    cbar2c = plt.colorbar(sm_bn, ax=axes[2], cax=cax2c, orientation='horizontal', ticks=[40,60,80], extend='max')

    cbar2a.ax.tick_params(labelsize=12)
    cbar2b.ax.tick_params(labelsize=12)
    cbar2c.ax.tick_params(labelsize=12)
    
    cbar2a.set_label('BN (%)', fontsize=14)
    cbar2b.set_label('NN (%)', fontsize=14)
    cbar2c.set_label('AN (%)', fontsize=14)

    axes[0].set_title('Deterministic forecast', fontsize=18)
    axes[1].set_title('Forecast anomaly', fontsize=18)
    axes[2].set_title('Probabilistic forecast', fontsize=18)

    plt.subplots_adjust(wspace=0.025, hspace=0.025, bottom=0.15)
    plt.tight_layout()
    plt.savefig(fig_dir_fc + f'fc_{var}_{period}_{model}_{modeldatestr}.eps', format='eps', bbox_inches='tight')    
    plt.savefig(fig_dir_fc + f'fc_{var}_{period}_{model}_{modeldatestr}.png', format='png', bbox_inches='tight')
    
def save_forecast(varname, varunit, data, lat, lon, datapath_output,
                  filename, projection='epsg:3857', fill_value=-9999):
    '''
    This function writes output data to a tif file
    
    Input
    -----
    varname: string
        contains the name of the variable
    varunit: string
        contains the variable unit
    data: array
        contains the data to be saved. The format is (time, lat, lon), where 
        the latitudes decrease
    lat: array
        contains the latitudes, from 90 N to 90 S.
    lon: array
        contains the longitudes
    timevec: array
        contains the time steps to be saved
    datapath_output: string
        contains the output directory where the tiff needs to be saved
    filename: string
        contains the filename of the tiff file (without .tiff)
    fill_value: float
        if no data is available at a point, the fill_value is used
    
    Returns
    -------
    The function writes a tif file with name 'varname.tiff' in the folder
    datapath_output
    '''
    
    if not filename[-4:] == 'tiff':
        filename = datapath_output + filename + '.tiff'
    else:
        filename = datapath_output + filename
        
    # set geotransform
    # projection 4326 means lat-lon coordinates
    # out projection can be chosen to preference: EPSG 3857 is WGS84  
    outProj = Proj(projection)
    inProj = Proj('epsg:4326')

    # Find max and min coordinates in new projection
    xmin,ymin = transform(inProj,outProj,lon.min(),lat.min())
    xmax,ymax = transform(inProj,outProj,lon.max(),lat.max())
    
    # Numbers of gridpoints in x and y
    nx = data.shape[1]
    ny = data.shape[0]
    
    # Find the resolution of the new grid and make the new grid
    # Divide by number minus one because one cell too little is taken into account
    # by taking xmax and xmin
    xres = (xmax - xmin) / float(nx-1)
    yres = (ymax - ymin) / float(ny-1)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    
    # create the raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(filename, nx, ny, 1, 
                                                  gdal.GDT_Float32)
    dst_ds.SetDescription(varname) # Set varname as description
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(int(projection[-4:])) # Use correct epsg from outproj
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    
    # Write the data to the raster band
    dst_ds.GetRasterBand(1).WriteArray(data[:,:])
    dst_ds.GetRasterBand(1).SetNoDataValue(fill_value)
    dst_ds.GetRasterBand(1).SetUnitType(varunit)
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close




def integrate_grid_to_polygon(input_grid, mask_2d, time_axis = 0, method = 'average', quantile = None):
    """
    Function translates gridded data into values per polygon
    
    Input:
        input_grid - the data on a grid that needs to be interpoplated to polygon values
        mask_2d - mask that tells for each grid-cell how much it overlaps with a certain polygon.
                  created by the function mask_from_polygon. maks2d is unique for each polygon that
                  you want to calculate
        time_axis - axis that indicates if input_grid data is 2d or 3d
        method - integration technique, possible options;
                 average:  take the average values of the gridboxes in the polygon
                 max:      take the maximum value of the gridboxes in the polygon. 
                           In this method, only the gridboxes that are significantly 
                           overlapping with the polygon are considered
                 quentile: take a certain quantile value of the gridboxes in the polygon
    Returns: an integrated value per polygon (where the polygon is represented by mask_2d)
    
    """    

    if time_axis != 0: 
        input_grid = np.moveaxis(input_grid, time_axis, 0)
    
    mask_3d = np.repeat(          
            mask_2d[np.newaxis, :, :], 
            input_grid.shape[time_axis], 
            axis=0)
        
    if method == 'average':
        value_integrated = np.sum(input_grid * mask_3d, axis=(1, 2)) / np.sum(mask_2d)
        
    elif method == 'max':
        max_mask = np.max(mask_3d)
        mask_3d[mask_3d >= 0.5*max_mask] = 1.
        mask_3d[mask_3d < 0.5*max_mask] = 0.
        value_integrated = np.max(input_grid * mask_3d, axis=(1, 2))  
          
    elif method == 'quantile': 
        if quantile is None: 
            raise Exception('quantile not selected')
        if type(input_grid) is xr.core.dataarray.DataArray: 
            input_grid = input_grid.values
        weight = mask_2d.flatten()
        value_integrated = [weighted_quantile(input_grid[i_fc].flatten(), quantile, weight) for i_fc in range(input_grid.shape[time_axis])]
        value_integrated = np.array(value_integrated)
    else: 
        raise Exception('Invalid method.') 
        
        
    return(value_integrated)
      

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):      
    """ 
    This function calculates a weighted quantile. It is similar to numpy.percentile, but supports weights.
    This is used for obtaining a weighted quantile of the gridded risk values in each township
    The weights are based on the percentual overlap of each grid cell with the township shape.
    
    NOTE: quantiles should be in [0, 1]!
    
    Input
        values: numpy.array with data
        quantiles: array with the desired quantiles [0,1]
        sample_weight: array (same lenght as values) with the weights
        values_sorted: bool, if True, then will avoid sorting of initial array
        old_style: if True, will correct output to be consistent with numpy.percentile.
        
     Returns   
         numpy.array with computed quantiles.
    
    source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    theoretical background: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
        
    return np.interp(quantiles, weighted_quantiles, values)

def load_polygons(shpfile, name_column, name_column_higher_level, 
                  lats=None, lons=None, shape_mask_dir=None, make_shapemask=True):
    '''
    Function to load the names of the polygons and to create shape masks from 
    a shapefile, if non-existent.
    
    For each polygon in the provided shape file, the name will be given. If the
    corresponding shape mask is not existent, the shape masks will be created.
    The shape mask gives the amount of overlap for each polygon in the shape 
    file on the provided input grid. The shape mask has values ranging from 0
    (no overlap between polygon and grid cell) to 1 (100% overlap).
    
    Input:
        shpfile: shapefile.Reader() object: the shapefile contents
        name_column: int: the column number which contains the name of the polygon
        name_column_higher_level: int: the column number of the name of one 
                administrative level higher. This is added if double names occur
        lats: array: the latitudes of the input grid
        lons: array: the longitudes of the input grid
        shape_mask_dir: str: the directory where the shape masks are saved
        
    Output:
        polygon_names: list: list with names of the polygons
        polygon_shapes: list: list with the shapes
    
    '''
    
    # Load the names of the polygons
    polygon_names = []
    polygon_shapes = []
    polygon_ids = []
    for i_t in range(shpfile.numRecords):
        polygon_name = shpfile.record(i_t)[name_column].title()
        polygon_names.append(polygon_name.replace('/','-'))
        polygon_shape = shpfile.shapeRecords()[i_t].shape
        polygon_shapes.append(polygon_shape)
        
        polygon_ids.append(i_t+1) # To start at 1
        polygon_ids.append(i_t)

    if not len(polygon_names) == len(set(polygon_names)):
        # If there are duplicates in the township names, add the name of the 
        # administrative level above
        duplicate_names = [item for item, count in collections.Counter(polygon_names).items() if count > 1]
        
        # Loop again over all township names:
        for i_t in range(len(polygon_names)):
            if polygon_names[i_t] in duplicate_names:
                # Add the higher level name
                polygon_names[i_t]  = polygon_names[i_t]  + '_' + shpfile.record(i_t)[name_column_higher_level]
                
    # Check again if all township names are unique
    assert len(polygon_names) == len(set(polygon_names))
     
    if make_shapemask == True:
        for i_p in range(shpfile.numRecords):
            if not os.path.isfile(shape_mask_dir + 'shape_mask_' + polygon_names[i_p].replace('/','-') + '.npy'):
                town_shape = shpfile.shapeRecords()[i_p]
                # We want a mask with just zeroes and ones, as landuse resolution is high enough (so fraction overlap not really important)
                # If more than 50% of a grid cell overlaps with township, this is rounded to 1 (integer). 
                # Otherwise, the overlap is rounded to 0 (integer)
                # This creates, for every township, a 2d mask (size of landuse file) with zeroes and ones.
                township_mask = mask_from_polygon(lons, lats, town_shape)
                
                
                if np.max(township_mask)<0.5:
                    # If the township is so small that it only falls into 1 gridpoint: ceil
                    township_mask = np.ceil(township_mask)
                township_mask = np.round(township_mask).astype(int)
                # township_mask = data_sanity_check(township_mask,0,1, 'township mask calculation')
                print('finished mask for shape nr ' + str(i_p+1))
                
                # make sure every township has overlap with at least one grid cell
                assert np.any(township_mask > 0.0)
                
                if not os.path.exists(shape_mask_dir):
                    os.makedirs(shape_mask_dir)
                
                np.save(shape_mask_dir + '/shape_mask_' + polygon_names[i_p].replace('/','-'),
                        township_mask)  
        
    # Extra check on '/'
    for ts in range(len(polygon_names)):
        if '/' in polygon_names[ts]:
            polygon_names[ts] = polygon_names[ts].replace('/','-')    
    
    return polygon_names, polygon_shapes, polygon_ids


    
def mask_from_polygon(lon_in, lat_in, input_shape):   
    """
    calculate mask for every township containing per grid cell the percentage of overlap
    Ranges from 0 to 1
    0 = forecast grid cell and township do not overlap
    1 = forecast grid cell and township fully overlap
    0.3 = township covers 30% of the grid cell
    
    input:  latitude and longitude arrays
            shape from a shapefile of the form:
                <class 'shapefile.ShapeRecord'> or 
                <class 'shapely.geometry.polygon.Polygon'
    output: mask_2D; numpy array of the shape (lon, lat) containing percentage of overlap
    
    """
    
    llons, llats = np.meshgrid(lon_in, lat_in)
    mask_2D = np.zeros_like(llons)
    res = np.around(np.diff(llons)[0, 0], decimals=2)
    
    if str(type(input_shape)) == "<class 'shapefile.ShapeRecord'>": 
        multi = shape(input_shape.shape.__geo_interface__)
    elif str(type(input_shape)) == "<class 'shapely.geometry.polygon.Polygon'>": 
        multi = input_shape
    else:
       raise ValueError('this shape is not of the right class. Function only works \
                        for <class shapefile.ShapeRecord> or <class shapely.geometry.polygon.Polygon')

    for xx in range(llons.shape[0]):
        for yy in range(llons.shape[1]):
            x1 = llons[xx, yy] - res / 2
            x2 = llons[xx, yy] + res / 2
            y1 = llats[xx, yy] - res / 2
            y2 = llats[xx, yy] + res / 2
            poly_grid = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            mask_2D[xx, yy] = multi.intersection(
                    poly_grid).area / poly_grid.area
    
    return(mask_2D)
