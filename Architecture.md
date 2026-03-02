# Architecture
This file describes the architecture of the models used. 

## GNN
### Inputs
Inputs are same as before i.e:
mean
sd
Fold Data
EWMA features etc
Volatility features
RSI
fundamental features (can be toggled)

### Layers
PATH 1: GCN(F → H//2, edge_corr)  → LayerNorm → ELU
PATH 2: GCN(2F → H//2, edge_vendor) → LayerNorm → ELU

MERGE:  concat → [N, H] → Dropout

REFINE: GCN(H → H, edge_combined) → LayerNorm → ELU → Dropout

OUTPUT: Linear(H → 1)  (regression output)

## MLP
A normal MLP with layers defined as a list in the config. For ex. [64,64] is what it implies. ELU activation here too.

### Classification objective
- Targets: `sign(return)` → binary 0/1 for BCEWithLogitsLoss
- `pos_weight` computed automatically from training label distribution
- Accuracy tracked per epoch

### Layers
Input(F) → [Linear → (LayerNorm?) → ELU → Dropout] × len(layers) → Linear(H_last, 1)

### Notes
- Input features are already RobustScaler'd in `build_folds` — no additional input normalisation needed
- Optional LayerNorm between hidden layers (off by default)
- Per-stock feedforward: each stock processed independently with shared weights




