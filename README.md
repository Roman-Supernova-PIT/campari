# To create an environment to run this code
## The following 'module load' will be necessary on NERSC
## On other systems 'conda' may be already in your path.
## Consult the documentation for the relevant system.
module load conda

# Create our conda environment.  This is rather simple here
# Will fill it with pip installable Python libraries below.
conda create -n multismp ipykernel python=3.11

conda activate multismp
conda install -c conda-forge fitsio
pip install -r requirements.txt

# Create a Jupyter kernel with this environment
python -m ipykernel install --user --name multismp --display-name multismp
