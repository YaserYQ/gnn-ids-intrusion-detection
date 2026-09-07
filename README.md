# Quantized GNN Intrusion Detection System

A Graph Neural Network (GNN) based Network Intrusion Detection System (NIDS), built and trained on Google Colab with GPU acceleration (PyTorch + PyTorch Geometric), with post-training and quantization-aware training pipelines to produce an efficient INT8 model suitable for deployment.

This was developed as a senior capstone project (Spring 2025).

## Overview

Network traffic is modeled as a graph, where each node is a network flow/packet record and edges connect temporally-related records. A GraphSAGE-based GNN with residual connections and self-attention is trained to classify each node as `Normal` or `Attack`.

The pipeline takes a raw IDS/IoT network traffic dataset through six stages, each captured in its own notebook:

| Stage | Notebook | Description |
|---|---|---|
| 1 | `01_Data_Preprocessing` | Loads the raw IDS-IoT-2024 CSV, removes duplicate rows/columns and NaNs, and writes a cleaned CSV. |
| 2 | `02_Graph_Representation.ipynb` | Splits data by network flow (to prevent leakage across train/val/test), selects the top-20 most informative features via Random Forest importance, verifies there's no data leakage between splits, and builds graph representations (nodes = records, edges = temporal connections) for PyTorch Geometric. |
| 3 | `03_Model_Training.ipynb` | Trains the `CyberThreatDetector` model — a GraphSAGE-based GNN with residual connections, layer norm, and multi-head self-attention — using `NeighborLoader` mini-batching, label smoothing, gradient clipping, and an LR scheduler. Achieves F1 ≈ 0.997 on the held-out test set. |
| 4 | `04_Post-Training_Quantization.ipynb` | Applies dynamic INT8 quantization (`torch.quantization.quantize_dynamic`) to the trained FP32 checkpoints and compares FP32 vs. INT8 accuracy, latency, and file size across all trained seeds. |
| 5 | `05_Quantization_Aware_Training.ipynb` | Trains a simplified GNN architecture from scratch using Quantization-Aware Training (QAT) via PyTorch's `prepare_qat`/`convert` APIs, so the model learns to be robust to quantization noise during training rather than after. Achieves F1 ≈ 0.995 post-quantization. |
| 6 | `OPTUNA.ipynb` | Uses Optuna (TPE sampler + median pruner) to search the hyperparameter space (hidden dim, layers, dropout, learning rate, weight decay, loss weighting, etc.), then trains a 10-seed ensemble on the best-known configuration and evaluates it. Final ensemble result: **F1 = 0.9978, Precision = 0.9986, Recall = 0.9970** (decision threshold ≈ 0.1434), in a ~21-minute run. |

## Environment

All notebooks were run on Google Colab (GPU runtime, mostly A100). Key dependencies:

```
torch==2.6.0+cu124
torch_geometric==2.6.1
pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv  (matched to torch 2.6.0+cu124)
imbalanced-learn
optuna
tqdm
scikit-learn
pandas
numpy
```

See `requirements.txt` for a pip-installable version. The PyG extension packages (`pyg_lib`, `torch_scatter`, etc.) need to be installed from the PyG wheel index matching your exact torch + CUDA version — see the install cell at the top of each notebook.

## Data

Notebooks expect the dataset and intermediate artifacts under a Google Drive path structured as:

```
CyberThreatDetectionSystem_Project/
├── Data/
│   ├── raw/            # original IDS-IoT-2024.csv
│   └── processed/      # cleaned CSVs, train/val/test splits, graph .pkl files
├── Models/             # saved checkpoints
└── Results/            # Optuna study + results
```

Notebooks 2–6 mount Google Drive and reference this path directly, so update `DRIVE_PATH` if you use a different structure.

## Notes

- Notebooks are kept exactly as exported from Colab (including original cell outputs, execution counts, and Colab widget metadata) for reference/reproducibility.
