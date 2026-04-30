from pathlib import Path

import pandas as pd


TARGET = "임신 성공 여부"


def run_eda(data_dir: Path) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(f"target mean: {train[TARGET].mean():.6f}")

    summary = pd.DataFrame(
        {
            "column": train.columns,
            "train_missing": train.isna().sum().reindex(train.columns).to_numpy(),
            "train_nunique": train.nunique(dropna=False).reindex(train.columns).to_numpy(),
            "dtype": train.dtypes.astype(str).reindex(train.columns).to_numpy(),
        }
    )
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    run_eda(repo_root / "data")
