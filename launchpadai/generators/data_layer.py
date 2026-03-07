"""Generate the data layer — structured data management, versioning, and processing."""
from pathlib import Path


def generate_data_layer(config: dict, project_path: Path):
    """Generate data directory structure and processing utilities."""
    if not config.get("include_data_layer"):
        # Just ensure basic data dirs exist
        (project_path / "data" / "documents").mkdir(parents=True, exist_ok=True)
        return

    base = project_path / "data"

    # Create data directory structure
    for subdir in ["raw", "processed", "interim", "external", "documents"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    # Data README
    _write(base / "README.md", """# Data Directory

Organized following the standard data science project layout.

| Folder | Purpose |
|--------|---------|
| `raw/` | Original, immutable data. Never modify files here. |
| `interim/` | Intermediate data that has been partially transformed |
| `processed/` | Final, cleaned data ready for modeling or ingestion |
| `external/` | Data from third-party sources |
| `documents/` | Documents for RAG ingestion (PDFs, text files, etc.) |

## Data Versioning

Consider using [DVC](https://dvc.org/) for data versioning:
```bash
pip install dvc
dvc init
dvc add data/raw/your_dataset.csv
git add data/raw/your_dataset.csv.dvc
```

## Important
- Never commit large data files to git
- Add data files to `.gitignore`
- Document data sources and transformations
""")

    # .gitkeep files so empty dirs are tracked
    for subdir in ["raw", "processed", "interim", "external"]:
        _write(base / subdir / ".gitkeep", "")

    # Sample data config
    _write(base / "data_config.yaml", f"""# Data configuration
# Update these paths and settings for your project

sources:
  primary:
    path: "data/raw/"
    format: "{config.get('data_format', 'csv')}"
    description: "Primary dataset"

processing:
  output_dir: "data/processed/"
  chunk_size: 10000  # Rows per processing batch

validation:
  # Define expected schema
  required_columns: []
  # column_name: expected_type
  column_types: {{}}
""")

    # Data processing utilities
    _write(project_path / "data_processing" / "__init__.py", "")

    data_format = config.get("data_format", "csv")

    _write(project_path / "data_processing" / "loader.py", f'''"""Data loader — read and validate datasets."""
import pandas as pd
import yaml
from pathlib import Path


class DataLoader:
    """Load and validate data from various sources."""

    def __init__(self, config_path: str = "data/data_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def load_raw(self, name: str = "primary") -> pd.DataFrame:
        """Load a raw dataset by name."""
        source = self.config["sources"].get(name)
        if not source:
            raise ValueError(f"Unknown data source: {{name}}")

        path = Path(source["path"])
        fmt = source.get("format", "{data_format}")

        if fmt == "csv":
            files = list(path.glob("*.csv")) + list(path.glob("*.tsv"))
            if not files:
                raise FileNotFoundError(f"No CSV/TSV files in {{path}}")
            return pd.read_csv(files[0])
        elif fmt == "parquet":
            files = list(path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No Parquet files in {{path}}")
            return pd.read_parquet(files[0])
        elif fmt == "json":
            files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(f"No JSON files in {{path}}")
            return pd.read_json(files[0], lines=files[0].suffix == ".jsonl")
        else:
            raise ValueError(f"Unsupported format: {{fmt}}")

    def load_processed(self, name: str) -> pd.DataFrame:
        """Load a processed dataset."""
        path = Path(self.config["processing"]["output_dir"]) / f"{{name}}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        path = path.with_suffix(".csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Processed dataset '{{name}}' not found")

    def save_processed(self, df: pd.DataFrame, name: str, fmt: str = "parquet"):
        """Save a processed dataset."""
        output_dir = Path(self.config["processing"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            df.to_parquet(output_dir / f"{{name}}.parquet", index=False)
        else:
            df.to_csv(output_dir / f"{{name}}.csv", index=False)

        print(f"Saved {{len(df)}} rows to {{output_dir / name}}.{{fmt}}")

    def validate(self, df: pd.DataFrame) -> dict:
        """Validate a dataframe against the configured schema."""
        issues = []
        validation = self.config.get("validation", {{}})

        # Check required columns
        for col in validation.get("required_columns", []):
            if col not in df.columns:
                issues.append(f"Missing required column: {{col}}")

        # Check column types
        for col, expected_type in validation.get("column_types", {{}}).items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if expected_type not in actual:
                    issues.append(f"Column '{{col}}': expected {{expected_type}}, got {{actual}}")

        return {{
            "valid": len(issues) == 0,
            "issues": issues,
            "rows": len(df),
            "columns": len(df.columns),
        }}
''')

    _write(project_path / "data_processing" / "transforms.py", '''"""Data transformations — cleaning, feature engineering, preprocessing."""
import pandas as pd
import numpy as np


class DataTransformer:
    """Common data transformations for preprocessing."""

    def clean_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: strip strings, drop full-empty rows."""
        df = df.copy()

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].str.strip() if df[col].dtype == "object" else df[col]

        # Drop rows where ALL values are null
        df = df.dropna(how="all")

        return df

    def handle_missing(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """Handle missing values.

        Strategies: 'median', 'mean', 'mode', 'drop', 'zero'
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if strategy == "drop":
            return df.dropna()
        elif strategy == "zero":
            return df.fillna(0)

        for col in numeric_cols:
            if df[col].isnull().any():
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Fill categorical with mode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

        return df

    def remove_outliers(self, df: pd.DataFrame, columns: list[str] = None, method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or z-score method."""
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        mask = pd.Series(True, index=df.index)

        for col in columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (df[col] >= Q1 - factor * IQR) & (df[col] <= Q3 + factor * IQR)
            elif method == "zscore":
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask &= z < factor

        removed = (~mask).sum()
        print(f"Removed {removed} outlier rows ({removed/len(df)*100:.1f}%)")
        return df[mask].reset_index(drop=True)

    def encode_categoricals(self, df: pd.DataFrame, columns: list[str] = None, method: str = "onehot") -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if method == "onehot":
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        elif method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                df[col] = le.fit_transform(df[col].astype(str))

        return df
''')


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
