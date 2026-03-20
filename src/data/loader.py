import pandas as pd
import yaml
from pathlib import Path


def load_data(config_path=None):
    # Cố định tìm config ở root repo: <repo>/configs/params.yaml
    repo_root = Path(__file__).resolve().parent.parent.parent
    if config_path is None:
        config_path = repo_root / 'configs' / 'params.yaml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = repo_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = repo_root / config.get('data_path', 'data/raw/bank+marketing/bank/bank-full.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, sep=';')
    return df, config