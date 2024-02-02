local_repo_root=$(git rev-parse --show-toplevel)
cd "$local_repo_root"

module load Python/3.10.8-GCCcore-12.2.0
python -m venv venv
source venv/bin/activate
pip install -r idun/requirements.txt

wandb login
