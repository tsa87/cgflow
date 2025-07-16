# Data Preparation

## Optimization

### 1: Building blocks extraction

#### Option A: From Enamine Catalog or Stock

Use the "Comprehensive Catalog" or "Stock" from [Enamine](https://enamine.net/building-blocks/building-blocks-catalog):

```bash
cd experiments/data

# case 1 - extract smiles from the Enamine Catalog SDF file
python scripts/a_catalog_to_smi.py -b <CATALOG_SDF> -o building_blocks/enamine_catalog.smi --cpu <CPU>
# case 2 - extract smiles from the Enamine Stock SDF file
python scripts/a_stock_to_smi.py -b <STOCK_SDF> -o building_blocks/enamine_stock.smi --cpu <CPU>
```

#### Option B: From custom SMI file

```bash
python scripts/a_refine_smi.py -b <BLOCK_SMI> -o building_blocks/custom_block.smi --cpu <CPU>
```

### 2: Drug-like filtering (optional)

```bash
python scripts/b_druglike_filter.py -b building_blocks/enamine_stock.smi -o building_blocks/enamine_stock_druglike.smi --cuda
```

### 3: Environment construction

```bash
python scripts/c_create_env.py -b building_blocks/enamine_stock.smi -o envs/enamine_stock/ --cpu <CPU>
```

---
## Multi-pocket training

Download the CrossDocked2020 dataset used in RxnFlow ([Google drive](https://drive.google.com/drive/folders/1e5pPZaTRGhvEMky3K2OKQ9-jV_NweK-a)):

```bash
cd experiments/
gdown --id 1iGr053FDC9tCYz4es4cRJ6WpkEEi3CAW -O CrossDocked2020_all.tar.gz
tar -xvzf CrossDocked2020_all.tar.gz
```

---

## Pretraining the pose prediction model

Download and prepare the preprocessed PLINDER dataset for pose prediction pretraining:

```bash
mkdir -p data/experiments/cgflow/plinder
cd data/experiments/cgflow/plinder
gdown --id 1dhH1Yfdr9L2lt-JlwylxS2Um-kU3ZZUZ -O plinder_20A.tar.gz
tar -xzf plinder_20A.tar.gz
```
