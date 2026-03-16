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

You can change configs using yaml files. 