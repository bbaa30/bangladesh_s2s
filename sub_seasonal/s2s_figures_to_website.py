#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:31:40 2023

Upload the S2S figures to the WI-website

@author: bob
"""
import os
import sys
import shutil
import datetime

import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('/srv/config/config_bd_s2s.ini')

# Also load the wi_library for the ftp upload
config_wi = ConfigParser()
config_wi.read('/srv/config/config_scripts.ini')
lib_dir = config_wi['paths']['lib']
sys.path.append(lib_dir)
import wi_library as wi

# Set the directories from the config file
direc = config['paths']['s2s_dir'] 

fig_dir_hc = direc + 'output_figures_hindcast/'
fig_dir_fc = direc + 'output_figures_forecast/'
fig_dir_web = direc + 'ouput_figures_website/'

if not os.path.exists(fig_dir_web):
    os.makedirs(fig_dir_web)

# List all variables, types and scores
variables = ['tmax', 'tmin', 'tp']
periods = ['week1', 'week2', 'week3+4']
models = ['ecmwf', 'eccc', 'ncep', 'multi_model']
scores = ['pearson', 'ioa', 'groc', 'rpss', 'bss_an', 'bss_nn', 'bss_bn']

# Set the modeldate
modeldate = datetime.datetime.today() - datetime.timedelta(1)
modeldate = datetime.datetime(modeldate.year, modeldate.month, modeldate.day, 0,0)
modeldatestr = modeldate.strftime("%Y%m%d")

# Settings for ftp upload    
block_ftp_upload = False # config.getboolean('tests','block_ftp_upload')
wi.settings['block_ftp_upload'] = block_ftp_upload
host = 'ftp.weatherimpact.com'
username = 'weatherimpactcom'
passwd = '993560a792e622201090d67041f7882e'
ftp_target = '/srv/home/weatherimpactcom/domains/weatherimpact.com/htdocs/www/wp-content/uploads/automatic_uploads/'
hostkey = b'AAAAB3NzaC1yc2EAAAADAQABAAABAQDS1LdqVD/mDYGgi9eYRiiLDvbwJEIHMV1lEjPevlQVQ9UD+2wbgB5UsMsVvH2bSWkouDfRI5NAVmKdwlu6r879zyvzOy10r6HsGnLxKJAYbiF9nnU8Gv9sRo7cimKkn+ztopvDMMqWABWxHHtOzwyLSZS2RyZEI3Y6LO5dMMvpfrTWK9e7yHS6m9vfkyfXyKMrw8cCQbcLWYxT6gPYsrePExZ5ha5WDe8KPSH/N6Dd6BwGE9adsmFR+ayDO8Doso3ChSAJcf1gQ4OK/rSLU3cApzbeWSS0+Z0aDiuaU/Mf8COIdkTdkTF/D3h8T7KQAzu0Ib4Bjguun+OrQ//VOjGJ'


# Take the forecast figures and convert to the website figure names (without date)
for var in variables:
    for period in periods:
        # Copy figure to website directory and rename
        os.remove(f'{fig_dir_web}s2s_bangladesh_fc_{var}_{period}.png')
        shutil.copy(f'{fig_dir_fc}fc_{var}_{period}_{modeldatestr}.png',
                    f'{fig_dir_web}s2s_bangladesh_fc_{var}_{period}.png')
        
# Do the same for all the hindcast figures
for var in variables:
    for period in periods:
        for model in models:
            for score in scores:
                # Copy figure to website directory and rename
                try:
                    if model == 'multi_model':
                        model_web = 'mm'
                    else:
                        model_web = model
                    os.remove(f'{fig_dir_web}s2s_bangladesh_hc_{var}_{score}_{period}_{model_web}.png')
                    shutil.copy(f'{fig_dir_hc}hc_{var}_{score}_{period}_{modeldatestr}_{model}.png',
                                f'{fig_dir_web}s2s_bangladesh_hc_{var}_{score}_{period}_{model_web}.png')
                except:
                    continue

# Make a list with files to upload
web_figures = os.listdir(fig_dir_web)
for file in web_figures:
    wi.sftp_upload(host, username, passwd, fig_dir_web+file, 
                   target_location = ftp_target, overwrite=True,
                   hostkey=hostkey)