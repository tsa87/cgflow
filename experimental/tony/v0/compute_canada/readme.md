# Install environment
1. Activate python module
```
module purge
module load python/3.11 rdkit/2024.03.4 scipy-stack/2024a openbabel/3.1.1
```

2. Create a Python virtual environment
```
virtualenv --no-download ~/equinv
source ~/equinv/bin/activate
```

3. Install library
```
pip install -r requirements.txt
```

# Downlaod dataset
```
mkdir -p semlaflow/saved
gdown --folder https://drive.google.com/drive/folders/1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT -O semlaflow/saved
```
