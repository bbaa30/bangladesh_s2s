#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:50:17 2023

Download NCEP hindcasts from the ECMWF MARS system.

@author: bob
"""
import datetime
import os
import numpy as np
from ecmwfapi import ECMWFDataServer

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
cwd = os.getcwd()
parent = os.path.abspath(os.path.join(cwd, os.pardir))
config.read(parent+'/config_bd_s2s.ini')

direc = config['paths']['s2s_dir'] + 'input_ncep/'

if not os.path.exists(direc):
    os.makedirs(direc)

# Make connection with the ECMWF MARS api
server = ECMWFDataServer()

# Set current date
modeldate = datetime.datetime.today() - datetime.timedelta(1)

# Make 2 arrays with the historical hindcast and forecast dates
hc_years = np.arange(1999,2011)
fc_years = np.arange(2015,modeldate.year)

hc_dates = [datetime.datetime(hc_year, modeldate.month, modeldate.day) for hc_year in hc_years]
fc_dates = [datetime.datetime(fc_year, modeldate.month, modeldate.day) for fc_year in fc_years]

# Download the old hindcast files (1999 to 2010)

#Download the data
target= f"{direc}ncep_hc_19992010_"+modeldate.strftime("%Y%m%d")+".nc"
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": "2011-03-01",
    "expver": "prod",
    "hdate": "".join([single_date.strftime("%Y-%m-%d/") for single_date in hc_dates])[:-1],
    "levtype": "sfc",
    "model": "glob",
    "number": "1/2/3",
    "origin": "kwbc",
    "param": "121/122/228228",
    "step": "6/to/840/by/6",
    "stream": "enfh",
    "time": "00:00:00",
    "type": "pf",
    "format":"netcdf",
    "target":target,
    "area": "27/87/20/93",
    "grid": "1/1"
})

# Download the old forecast files (since 2015)
#Download the data
target= f"{direc}ncep_hc_2015{fc_years[-1]}_"+modeldate.strftime("%Y%m%d")+".nc"

server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "expver": "prod",
    "date": "".join([single_date.strftime("%Y-%m-%d/") for single_date in fc_dates])[:-1],
    "levtype": "sfc",
    "model": "glob",
    "number": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15",
    "origin": "kwbc",
    "param": "121/122/228228",
    "step": "6/to/840/by/6",
    "stream": "enfo",
    "time": "00:00:00",
    "type": "pf",
    "format":"netcdf",
    "target":target,
    "area": "27/87/20/93",
    "grid": "1/1"
})
