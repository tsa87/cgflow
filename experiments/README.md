# Running Experiments

## 1. Train General State Flow first (Pose Prediction)

Train the state flow model for pose prediction:

```bash
sh scripts/A_semlaflow_train_crossdocked.sh
```

You can also download the pretrained model weights from [here](https://drive.google.com/file/d/1UpdOxfMVdALdAPpzG_dXF72diP2AbHOl/view?usp=sharing)

```bash
cd weights
gdown 1UpdOxfMVdALdAPpzG_dXF72diP2AbHOl
```

Note: This script trains pose prediction on Plinder dataset rather than the CrossDocked dataset as done in the paper experiments.
Plinder is a larger dataset, and we used unbiased pocket extraction. Therefore the pose prediction performance is improved compared to reported result.

For reproducing paper results, use pretrained model weights:

```bash
mkdir weights/
curl -L -o weights/crossdocked2020_till_end.ckpt https://figshare.com/ndownloader/files/54411752
```

## 2. Pocket-specific Optimization (LIT-PCBA)

```bash
cd experiments
wandb sweep sweep/redock.yaml
wandb agent <sweep-id>
```

## 2.A Pocket-specific Optimization (LIT-PCBA) for custom pocket

Custom generation expects `<DATA_FOLDER>/<PROTEIN>` folder contain `protein.pdb` and `ligand.mol2`. Please see `./data/test/LIT-PCBA/ADRB2` for an example.

The `ligand.mol2` is the reference ligand. It is only needed for extracting the pocket from the `protein.pdb`. If you don't have a reference ligand, you can just dock any ligand to the target protein pocket to obtain it.

```bash
cd experiments
python ./scripts/exp1_redock_unidock.py redock <LOG_DIR> ./data/envs/stock/ <DATA_FOLDER> ../weights/plinder_till_end.ckpt <PROTEIN> <SEED> 50

# Example for LIT-PCBA pocket ADRB2
python ./scripts/exp1_redock_unidock.py redock ./logs/exp1-redocking/ ./data/envs/stock/ ./data/test/LIT-PCBA/ ../weights/plinder_till_end.ckpt ADRB2 0  50
```

![LIT-PCBA results](assets/lit-pcba-results.png)

## 3. Pocket-conditional Generation

### A. Download CrossDocked Dataset

1. Get `crossdocked.tar.gz` from [here](https://drive.google.com/file/d/1BKYx_H1m-TzG_75Gk-7sjPkt5ow-Acdw/view?usp=sharing)
2. Extract dataset:

```bash
cd experiments/data/
gdown 1BKYx_H1m-TzG_75Gk-7sjPkt5ow-Acdw
tar -xzvf crossdocked.tar.gz
```

### B. Use Pretrained Weights

Use `crossdocked2020_till_end.ckpt` for consistency.

### C. Docking Score Proxy Setup

Follow instructions at [PharmacoNet](https://github.com/SeonghwanSeo/PharmacoNet/tree/main/src/pmnet_appl).

### D. Run Experiment

```bash
cd experiments
python scripts/exp3A_sbdd_proxy.py
```

**Note:** Baseline methods (e.g., SynFlowNet) are provided in supplementary materials.
