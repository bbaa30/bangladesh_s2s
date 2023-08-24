## Install packages in conda using this sh script
#
# Add the conda-forge channel
conda config --add channels conda-forge
conda config --add channels hallkjc01
#
# Install from file
while read requirements; do conda install --yes $requirements; done < requirements.txt
#
# Move ecmwfapirc and cdsapirc file
mv ecmwfapirc ~/.ecmwfapirc
mv cdsapirc ~/.cdsapirc
#
# Move the seasonal forecast systems version
mv seasonal_forecast_system_versions.csv ~/data/seasonal/input_metadata/seasonal_forecast_system_verions.csv