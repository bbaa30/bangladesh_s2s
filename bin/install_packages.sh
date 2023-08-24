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