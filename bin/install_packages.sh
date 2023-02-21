# Install packages in conda using this sh script

# Add the conda-forge channel and the hallkjc01 channel (xcast)
conda config --add channels conda-forge
conda config --add channels hallkjc01

# Install from file
while read requirement; do conda install --yes $requirement; done < requirements.txt