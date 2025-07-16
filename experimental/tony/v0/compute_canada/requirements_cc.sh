
#!/bin/bash

# Get the name of the current virtual environment
ENV_NAME=$(basename "$VIRTUAL_ENV")

# Print the name of the current virtual environment to the terminal
echo "Installing things into the environment '$ENV_NAME'"

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-index lightning

pip install --no-index typing_extensions
pip install --no-index torchmetrics
pip install --no-index tqdm

pip install --no-index wandb
pip install --no-index gdown
pip install datamol
pip install omegaconf

# prevent pip from installing binary packages and force it to build packages from sources.
# Cannot use posecheck
# pip install --no-binary hydride
# pip install git+https://github.com/cch1999/posecheck.git #avoid fixed panda version issues


# TODO be able to install
pip install biotite<=0.39.0 # can only do this because issues from hatch-cython
pip install prolif==2.0.3
pip install MDAnalysis==2.7.0

pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

echo "Installation completed successfully!"
