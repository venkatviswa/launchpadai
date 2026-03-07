"""Generate the notebooks layer — Jupyter notebooks for EDA and experimentation."""
from pathlib import Path


def generate_notebooks_layer(config: dict, project_path: Path):
    """Generate Jupyter notebooks with starter EDA templates."""
    if not config.get("include_notebooks"):
        return

    base = project_path / "notebooks"
    base.mkdir(parents=True, exist_ok=True)

    # README for notebooks folder
    _write(base / "README.md", """# Notebooks

Jupyter notebooks for exploratory data analysis, experimentation, and prototyping.

## Structure

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Initial EDA — schema, distributions, missing values |
| `02_data_quality.ipynb` | Data quality checks, outlier detection, cleaning |
| `03_feature_engineering.ipynb` | Feature creation, transformations, selection |
| `04_model_experiments.ipynb` | Model training experiments and comparison |
| `05_embedding_analysis.ipynb` | Embedding quality analysis for RAG pipeline |

## Usage

```bash
pip install jupyter
jupyter notebook notebooks/
```
""")

    # Data format specific imports
    data_imports = {
        "csv": "df = pd.read_csv('../data/raw/sample.csv')",
        "parquet": "df = pd.read_parquet('../data/raw/sample.parquet')",
        "json": "df = pd.read_json('../data/raw/sample.json', lines=True)",
        "sql": """from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///data/raw/sample.db'))
df = pd.read_sql('SELECT * FROM your_table LIMIT 1000', engine)""",
    }
    load_data = data_imports.get(config.get("data_format", "csv"), data_imports["csv"])

    # 01 - Data Exploration notebook
    _write(base / "01_data_exploration.ipynb", _notebook([
        _md_cell(f"# Data Exploration — {config['project_name']}\n\nInitial EDA to understand the data shape, types, distributions, and quality."),
        _code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

%matplotlib inline
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Environment ready!")"""),
        _md_cell("## 1. Load Data"),
        _code_cell(f"""# Load your dataset
# Update the path to point to your actual data file
{load_data}

print(f"Shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
df.head()"""),
        _md_cell("## 2. Basic Statistics"),
        _code_cell("""# Data types and memory usage
print("=== Data Types ===")
print(df.dtypes)
print(f"\\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Summary statistics
df.describe(include='all')"""),
        _md_cell("## 3. Missing Values"),
        _code_cell("""# Missing value analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'count': missing,
    'percent': missing_pct
}).sort_values('percent', ascending=False)

# Only show columns with missing values
missing_df = missing_df[missing_df['count'] > 0]
if len(missing_df) > 0:
    print("Columns with missing values:")
    print(missing_df)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
    missing_df['percent'].plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Values by Column')
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found!")"""),
        _md_cell("## 4. Distributions"),
        _code_cell("""# Numeric column distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], color='steelblue', edgecolor='black', alpha=0.7)
            axes[i].set_title(col)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Numeric Distributions', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()"""),
        _md_cell("## 5. Correlations"),
        _code_cell("""# Correlation matrix for numeric columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8)))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                square=True, ax=ax, linewidths=0.5)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # High correlations
    high_corr = corr.unstack().sort_values(ascending=False)
    high_corr = high_corr[(high_corr < 1.0) & (high_corr > 0.7)]
    if len(high_corr) > 0:
        print("\\nHighly correlated pairs (>0.7):")
        print(high_corr)"""),
        _md_cell("## 6. Categorical Analysis"),
        _code_cell("""# Categorical column analysis
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols[:5]:  # Limit to first 5
    print(f"\\n=== {col} ===")
    vc = df[col].value_counts()
    print(f"Unique values: {df[col].nunique()}")
    print(f"Top 10 values:")
    print(vc.head(10))
    
    if df[col].nunique() <= 20:
        fig, ax = plt.subplots(figsize=(10, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{col} — Value Counts')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()"""),
        _md_cell("## 7. Key Findings\n\n**TODO:** Summarize your findings here\n\n- Finding 1: ...\n- Finding 2: ...\n- Finding 3: ...\n\n### Next Steps\n- [ ] Clean data based on findings\n- [ ] Engineer features\n- [ ] Prototype model"),
    ]))

    # 02 - Data Quality notebook
    _write(base / "02_data_quality.ipynb", _notebook([
        _md_cell(f"# Data Quality Checks — {config['project_name']}\n\nSystematic data quality assessment: duplicates, outliers, consistency."),
        _code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.abspath('..'))

%matplotlib inline
sns.set_style('whitegrid')
print("Ready!")"""),
        _md_cell("## 1. Load Data"),
        _code_cell(f"""{load_data}
print(f"Shape: {{df.shape}}")"""),
        _md_cell("## 2. Duplicate Detection"),
        _code_cell("""# Check for duplicate rows
dups = df.duplicated()
print(f"Duplicate rows: {dups.sum()} ({dups.sum()/len(df)*100:.2f}%)")

# Check for duplicate keys (update column name)
# key_col = 'id'
# dup_keys = df[key_col].duplicated()
# print(f"Duplicate keys: {dup_keys.sum()}")"""),
        _md_cell("## 3. Outlier Detection"),
        _code_cell("""# IQR-based outlier detection for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > 0:
        outlier_summary.append({
            'column': col, 'outliers': outliers,
            'pct': round(outliers / len(df) * 100, 2),
            'lower_bound': round(lower, 2), 'upper_bound': round(upper, 2)
        })

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary).sort_values('outliers', ascending=False)
    print(outlier_df.to_string(index=False))
    
    # Box plots for top outlier columns
    top_outlier_cols = outlier_df.head(6)['column'].tolist()
    if top_outlier_cols:
        fig, axes = plt.subplots(1, len(top_outlier_cols), figsize=(4 * len(top_outlier_cols), 5))
        axes = [axes] if len(top_outlier_cols) == 1 else axes
        for ax, col in zip(axes, top_outlier_cols):
            df.boxplot(column=col, ax=ax)
            ax.set_title(col)
        plt.suptitle('Outlier Box Plots', y=1.02)
        plt.tight_layout()
        plt.show()
else:
    print("No outliers detected using IQR method.")"""),
        _md_cell("## 4. Data Type Consistency"),
        _code_cell("""# Check for mixed types and unexpected values
for col in df.columns:
    unique_types = df[col].dropna().apply(type).unique()
    if len(unique_types) > 1:
        print(f"MIXED TYPES in '{col}': {[t.__name__ for t in unique_types]}")

# Check date columns parse correctly
# date_cols = ['created_at', 'updated_at']
# for col in date_cols:
#     if col in df.columns:
#         try:
#             pd.to_datetime(df[col])
#             print(f"  {col}: Valid dates")
#         except Exception as e:
#             print(f"  {col}: INVALID — {e}")
"""),
        _md_cell("## 5. Quality Score\n\n**TODO:** Rate overall data quality and document cleaning actions needed."),
    ]))

    # 05 - Embedding analysis notebook (if RAG is enabled)
    if config.get("include_rag"):
        _write(base / "05_embedding_analysis.ipynb", _notebook([
            _md_cell(f"# Embedding Analysis — {config['project_name']}\n\nAnalyze embedding quality, cluster documents, and tune retrieval."),
            _code_cell("""import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sys, os
sys.path.insert(0, os.path.abspath('..'))

%matplotlib inline
print("Ready!")"""),
            _md_cell("## 1. Load Embeddings"),
            _code_cell("""from models.embeddings.provider import embeddings

# Sample texts to embed
sample_texts = [
    "How do I return a product?",
    "What is the refund policy?",
    "Cancel my subscription",
    "Track my order status",
    "Product specifications",
    "Billing questions",
    "Technical support needed",
    "Shipping times and costs",
]

# Generate embeddings
vectors = embeddings.embed_batch(sample_texts)
vectors = np.array(vectors)
print(f"Embedding shape: {vectors.shape}")
print(f"Dimensions: {vectors.shape[1]}")"""),
            _md_cell("## 2. Similarity Matrix"),
            _code_cell("""from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(vectors)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(len(sample_texts)))
ax.set_yticks(range(len(sample_texts)))
ax.set_xticklabels([t[:30] for t in sample_texts], rotation=45, ha='right')
ax.set_yticklabels([t[:30] for t in sample_texts])
plt.colorbar(im, label='Cosine Similarity')
ax.set_title('Document Similarity Matrix')
plt.tight_layout()
plt.show()"""),
            _md_cell("## 3. t-SNE Visualization"),
            _code_cell("""# Reduce to 2D for visualization
if len(vectors) > 5:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vectors)-1))
    vectors_2d = tsne.fit_transform(vectors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, c='steelblue', edgecolors='black')
    for i, text in enumerate(sample_texts):
        ax.annotate(text[:25], (vectors_2d[i, 0], vectors_2d[i, 1]),
                   fontsize=9, ha='center', va='bottom')
    ax.set_title('Document Embeddings (t-SNE)')
    plt.tight_layout()
    plt.show()"""),
            _md_cell("## 4. Retrieval Quality Test"),
            _code_cell("""# Test retrieval accuracy
test_queries = [
    "I want my money back",       # Should match refund/return
    "Where is my package?",        # Should match tracking/shipping
    "How to set up the product",   # Should match technical support
]

for query in test_queries:
    q_vec = np.array(embeddings.embed(query))
    similarities = cosine_similarity([q_vec], vectors)[0]
    ranked = sorted(zip(sample_texts, similarities), key=lambda x: x[1], reverse=True)
    
    print(f"\\nQuery: '{query}'")
    print("  Top 3 matches:")
    for text, score in ranked[:3]:
        print(f"    {score:.4f} — {text}")"""),
            _md_cell("## 5. Findings\n\n**TODO:** Note embedding quality issues and tuning opportunities.\n\n- Are semantically similar queries clustering together?\n- Are there surprising similarities/differences?\n- Should you fine-tune the embedding model?"),
        ]))

    # 04 - Model experiments notebook (if ML pipeline enabled)
    if config.get("include_ml_pipeline"):
        ml_framework = config.get("ml_framework", "sklearn")

        if ml_framework == "sklearn":
            model_code = """from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Prepare features and target
# UPDATE THESE for your dataset
# X = df.drop(columns=['target_column'])
# y = df['target_column']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example with dummy data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare models
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Train best model
best_name = max(results, key=lambda k: results[k]['mean'])
print(f"\\nBest model: {best_name}")
best_model = models[best_name]
best_model.fit(X_train, y_train)

# Evaluate
y_pred = best_model.predict(X_test)
print("\\n" + classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, '../ml_models/artifacts/best_model.joblib')
print("Model saved to ml_models/artifacts/best_model.joblib")"""
        elif ml_framework == "pytorch":
            model_code = """import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Example: simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)

# TODO: Replace with your actual data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train
model = SimpleNet(input_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

# Save
torch.save(model.state_dict(), '../ml_models/artifacts/model.pt')
print("Model saved!")"""
        elif ml_framework == "xgboost":
            model_code = """import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

# TODO: Replace with your actual data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with hyperparameter search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
}

model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")

# Evaluate
y_pred = grid.best_estimator_.predict(X_test)
print("\\n" + classification_report(y_test, y_pred))

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(grid.best_estimator_, max_num_features=15)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Save
joblib.dump(grid.best_estimator_, '../ml_models/artifacts/xgb_model.joblib')
print("Model saved!")"""
        else:  # transformers
            model_code = """# HuggingFace Transformers fine-tuning
# See ml_models/training/train.py for the full training script

print("For full fine-tuning, run:")
print("  python ml_models/training/train.py")
print()
print("This notebook is for quick experimentation.")
print("The training script handles checkpointing, logging, and evaluation.")"""

        _write(base / "04_model_experiments.ipynb", _notebook([
            _md_cell(f"# Model Experiments — {config['project_name']}\n\nTraining experiments, hyperparameter tuning, and model comparison.\n\nML Framework: **{ml_framework}**"),
            _code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.abspath('..'))

%matplotlib inline
print("Ready!")"""),
            _md_cell("## 1. Load Processed Data"),
            _code_cell(f"""{load_data}
print(f"Shape: {{df.shape}}")
df.head()"""),
            _md_cell("## 2. Model Training & Comparison"),
            _code_cell(model_code),
            _md_cell("## 3. Results Summary\n\n**TODO:** Document results\n\n| Model | Accuracy | F1 | Notes |\n|-------|----------|----|---------|\n| ... | ... | ... | ... |"),
        ]))


def _notebook(cells: list[dict]) -> str:
    """Create a Jupyter notebook JSON string."""
    import json
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }
    return json.dumps(nb, indent=1)


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source], "id": _cell_id()}


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None, "id": _cell_id()}


_cell_counter = 0
def _cell_id() -> str:
    global _cell_counter
    _cell_counter += 1
    return f"cell-{_cell_counter:04d}"


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
