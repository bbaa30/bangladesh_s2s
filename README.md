# S2S Bangladesh
This repository contains scripts to create an S2S forecast for Bangladesh. There are 2 subdirectories: one for sub-seasonal forecasts up to 4 weeks and one for seasonal forecasts up to 5 months.

## Installation
### Linux
The scripts run in Python 3.9, installed through anaconda. 
Use the script install_packages.sh in the /bin/ folder to install the correct package versions.

### Windows
The scripts run in Python 3.9 and have a set of required packages (see requirements.txt in the /bin/ folder). Copy the .ecmwfapirc to your home folder.
 
## Preparation
The repository contains a configuration file (config_bd_s2s.ini). Please edit this file before using the scripts! The config file contains the directories where the data is stored. The important directory is the s2s_dir, which will be the sub-directory where all the data will be stored and the scripts directory, where all scripts are located.

## Sub-seasonal scripts
The procedure of producing a sub-seasonal forecast is as follows:
 1. Run the script download_ecmwf_s2s_from_wi_api.py to collect the ECMWF netcdf input files
 2. Run the script download_eccc_s2s_operational.py to collect the ECCC netcdf input files
 3. Run the script download_cfsv2_operational.py to collect the NCEP CFSv2 netcdf forecast files
 4. Run the script download_cfsv2_hindcasts.py to collect the NCEP CFSv2 netcdf hindcast files
 5. Run prepare_ecmwf_data.py to prepare the ECMWF data input
 6. Run prepare_eccc_data.py to prepare the ECCC data input
 7. Run prepare_ncep_data.py to prepare the NCEP CFSv2 data input
 8. Run the script s2s_operational_forecast.py for the forecast
 9. Run the script generate_bulletin.py to generate a bulletin with the latest forecast

This procedure is also captured in run_all_operational_scripts.py.

Note that the ECMWF data is available twice a week: on Monday and Thursday. The ECCC data is available every Thursday. The NCEP CFSv2 data is available on a daily basis. It is recommended to run the system once per week on Friday, using all 3 models as input. Other models will be added in a later phase.

The operational forecast script checks which data is available and creates a multi-model output if both models are available, or a single model output if only one is available.

The scripts check the latest available data and will create a forecast for this date.

## Seasonal scripts
The seasonal scripts run once per month, around the 15th day of the month. Run the following scripts:
 1. download_seasonal_forecast_bangladesh_cds.py
 2. prepare_cds_files.py
 3. operational_seasonal_forecast.py

## Output
The scripts have several output files:
 - Figures with the skill analysis of the hindcast per model
 - Figures with the current forecast
 - JSON and CSV file with the forecast values on district level
 - Bulletin with forecast

For the seasonal forecast, figures with skills and figures with the forecast are outputted.

