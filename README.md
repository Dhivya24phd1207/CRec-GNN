# CRec-GNN: Category- and Revisit-Aware GNN for POI Recommendation

Code, per-seed results, and scripts for the paper
**"CRec-GNN: A Category- and Revisit-Aware Graph Neural Network for POI Recommendation, with a Leakage-Free Revisit Layer and a Tamil Nadu Check-in Dataset"**
(Dhivya K and Suganya R, VIT Chennai; submitted to IEEE Access).

## Contents

| Path | What it is |
|---|---|
| `notebooks/01_main_experiments_v15.ipynb` | Data loading, leakage-controlled split, all baselines, the RA variants and CRec-GNN (5 seeds, sampled + full-catalogue evaluation, per-user ranks). Writes `results/results_<DATASET>_v15.json`. |
| `notebooks/02_ablations_v17.ipynb` | Ablations of CRec-GNN (no RA layer, constant pi, no category conditioning, no decoder gate). Reuses the v15 runs; writes `results/results_<DATASET>_v17.json`. |
| `scripts/make_results.py` | Regenerates every table and figure of the paper from the result files. |
| `scripts/extract_tamil_nadu.py` | Builds the Tamil Nadu dataset from the public LBSN2Vec++ release with the GADM state boundary. |
| `results/` | Per-seed result files of all reported runs (metrics, training curves, per-user test ranks), repeat flags and training counts per test user. |
| `data/tamil_nadu_gadm.geojson` | Tamil Nadu state polygon (GADM). |

## Model names in the code vs. the paper

The notebooks grew out of an earlier model (SF-DT2Rec), so some internal keys keep older names:

| Paper | Key in the result files |
|---|---|
| CRec-GNN | `abl_v15/no_twin` |
| MF+RA, SASRec+RA, GNN+ID+RA | `main/MF+Rep`, `main/SASRec+Rep`, `main/GNN(Stable)+ID+Rep` |
| GNN, GNN+ID | `main/GNN(Stable)`, `main/GNN(Stable)+ID` |
| + state-feedback rollout (ablation) | `main/SF+ID+RepTwin` |
| Ablations | `abl_v17/no_repeat`, `abl_v17/const_pi`, `abl_v17/no_constraint`, `abl_v17/no_gate` |

## Data

The datasets are public and are not redistributed here.

| Dataset | Source | File name expected by the notebooks |
|---|---|---|
| Foursquare NYC, Tokyo | TSMC2014 (Yang et al., IEEE TSMC 2015) | `TSMC2014_NYC.csv`, `dataset_TSMC2014_TKY.csv` |
| Gowalla | SNAP (Cho et al., KDD 2011) | `Gowalla.csv` |
| Tamil Nadu | LBSN2Vec++ (Yang et al., WWW 2019; doi:10.4121/15112308, CC BY 4.0) | `TN_Foursquare_LBSN_core_u10_p5.csv` (built by the script below) |

```bash
pip install pandas shapely
python scripts/extract_tamil_nadu.py path/to/15112308.zip
# -> data/TN_Foursquare_LBSN_core_u10_p5.csv : 16,179 check-ins, 348 users, 1,246 POIs, 188 categories
```

## Reproducing the experiments

Requirements: Python 3.10+, PyTorch 2.x (GPU recommended), numpy, pandas, scipy, matplotlib.

1. Put the dataset CSV next to the notebooks (or set `CONFIG["CSV_PATH"]`).
2. In cell **2b** set `CONFIG["DATASET"]` to `"NYC"`, `"TKY"`, `"GOWALLA"` or `"TN"` (capital letters).
3. Run `01_main_experiments_v15.ipynb`, then `02_ablations_v17.ipynb`, for each dataset (Kernel > Restart & Run All).

All runs in the paper used 5 seeds, 200 epochs, and the checkpoint with the best validation NDCG@10, on one NVIDIA RTX 3050 GPU.

## Reproducing the tables and figures

```bash
pip install numpy scipy matplotlib pandas
python scripts/make_results.py        # writes paper_outputs/tab_*.tex and fig_*.pdf
```

## Citation

If you use this code or the Tamil Nadu extraction, please cite the paper (citation to be added on publication) and the LBSN2Vec++ dataset:

> D. Yang, B. Qu, J. Yang, P. Cudré-Mauroux, "Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph Embedding Approach," WWW 2019.

## License

Code: MIT (see `LICENSE`). Data remain under the licences of their original sources.
