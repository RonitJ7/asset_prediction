How to run the project:

1. Create a venv:
python3 -m venv .venv

2. Activate the venv:
source .venv/bin/activate

3. Run the code:
python3 src/main.py

Model selection (default is MLP):
python3 src/main.py model.selected_model=mlp
python3 src/main.py model.selected_model=rf
python3 src/main.py model.selected_model=xgb

Unified Optuna tuning (model selection + hyperparameters):
python3 src/tuner.py --n-trials 50

Useful tuning options:
python3 src/tuner.py --n-trials 80 --n-splits 3 --study-name all_models --storage sqlite:///outputs/optuna_all.db
python3 src/tuner.py --n-trials 40 --enable-gnn-search

You can change configs using yaml files. 