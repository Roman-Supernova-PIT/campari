module load conda
conda create -n multismp ipykernel python=3.11

conda activate multismp
pip install -r requirements.txt

# Create a Jupyter kernel
python -m ipykernel install --user --name multismp --display-name multismp
