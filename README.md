## 🔬 Note
> This repository contains the codebase used to reproduce the results from our [ICML 2025 submission](https://openreview.net/forum?id=4aXfSLfM0Z): **"Compositional Flows for 3D Molecule and Synthesis Pathway Co-design."**  For production-ready code or commercial use, please contact the authors directly or refer to most recent version.

# Compositional Flows for 3D Molecule and Synthesis Path Co-design

We introduce Compositional Generative Flows (CGFlow), a novel framework that extends flow matching to generate objects in compositional steps while modeling continuous states. We apply CGFlow to synthesizable drug design by jointly designing the molecule’s synthetic pathway with its 3D binding pose.

## Setup

### Install

```
# python: 3.11
conda install unidock
pip install -e . --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

# For PoseCheck
pip install -e '.[extra]' --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### Data

#### Data for 3DSynthFlow pose prediction pretraining

For convenience, we provide preprocessed data files for training pose prediction [here](https://figshare.com/s/fa64ce287f9bfc8beaba) Simply copy the `smol` folder from the CrossDock dataset and reference it when running the scripts. For example, specify `--data_path path/to/data/crossdock/smol` as an argument when executing the script.

#### Building block generation for 3DSynthFlow

The Enamine building block library is available upon request at https://enamine.net/building-blocks/building-blocks-catalog.
We used the "Comprehensive Catalog" released at 2024.06.10.

```
cd data
python scripts/a_catalog_to_smi.py -b <CATALOG_SDF> -o building_blocks/enamine_catalog.smi --cpu <CPU>
python scripts/b_create_env.py -b building_blocks/enamine_catalog.smi -o envs/catalog/ --cpu <CPU>
```

## Experiments

You can train state flow model for pose prediction using the following command:
```
sh scripts/A_semlaflow_train_crossdocked.sh
```
You can also directly download the pretrained model weights using the [same link](https://figshare.com/s/fa64ce287f9bfc8beaba).

To reproduce the results on the 15 targets from LIT-PCBA, run the following command:
```
cd experiments/
wandb sweep sweep/redock.yaml
wandb agent {sweep-id}
```

## 📖 Citation
If you use this work, please cite:

```bibtex
@inproceedings{shen2025compositional,
  title     = {Compositional Flows for 3D Molecule and Synthesis Pathway Co-design},
  author    = {Tony Shen and Seonghwan Seo and Ross Irwin and Kieran Didi and Simon Olsson and Woo Youn Kim and Martin Ester},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=4aXfSLfM0Z}
}
