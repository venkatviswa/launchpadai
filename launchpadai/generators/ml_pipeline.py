"""Generate the ML pipeline layer — model training, inference, and registry."""
from pathlib import Path


def generate_ml_pipeline(config: dict, project_path: Path):
    """Generate ML model training and inference pipeline."""
    if not config.get("include_ml_pipeline"):
        return

    base = project_path / "ml_models"
    ml_framework = config.get("ml_framework", "sklearn")

    # Directory structure
    for subdir in ["training", "inference", "artifacts", "configs"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    _write(base / "__init__.py", "")
    _write(base / "training" / "__init__.py", "")
    _write(base / "inference" / "__init__.py", "")

    # Artifacts README
    _write(base / "artifacts" / "README.md", """# Model Artifacts

Trained model files are saved here.

**Do not commit large model files to git.**
Add this folder to `.gitignore` and use cloud storage or model registry instead.

Suggested model registries:
- [MLflow](https://mlflow.org/) — open source, self-hosted
- [Weights & Biases](https://wandb.ai/) — managed service
- [DVC](https://dvc.org/) — version control for data and models
- AWS S3 / GCS — simple cloud storage
""")

    # Training config
    _write(base / "configs" / "training_config.yaml", f"""# Model training configuration
# Framework: {ml_framework}

experiment:
  name: "{config['project_name']}-v1"
  seed: 42
  test_size: 0.2
  cross_validation_folds: 5

model:
  framework: "{ml_framework}"
  # Update these based on your model
  hyperparameters: {{}}

training:
  epochs: 50           # For neural networks
  batch_size: 32       # For neural networks
  learning_rate: 0.001 # For neural networks
  early_stopping: true
  patience: 5

output:
  model_dir: "ml_models/artifacts"
  metrics_dir: "ml_models/artifacts/metrics"
""")

    # Training script (framework-specific)
    train_code = _get_training_script(config)
    _write(base / "training" / "train.py", train_code)

    # Inference module
    inference_code = _get_inference_module(config)
    _write(base / "inference" / "predictor.py", inference_code)

    # Model registry / versioning
    _write(base / "registry.py", '''"""Model Registry — track and load model versions.

Simple file-based registry. For production, use MLflow, W&B, or similar.
"""
import json
import os
from pathlib import Path
from datetime import datetime


REGISTRY_PATH = Path("ml_models/artifacts/model_registry.json")


def _load_registry() -> dict:
    """Load the model registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": {}}


def _save_registry(registry: dict):
    """Save the model registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    name: str,
    version: str,
    artifact_path: str,
    metrics: dict = None,
    metadata: dict = None,
):
    """Register a trained model."""
    registry = _load_registry()

    if name not in registry["models"]:
        registry["models"][name] = {"versions": {}}

    registry["models"][name]["versions"][version] = {
        "artifact_path": artifact_path,
        "metrics": metrics or {},
        "metadata": metadata or {},
        "registered_at": datetime.now().isoformat(),
    }

    # Update latest pointer
    registry["models"][name]["latest"] = version

    _save_registry(registry)
    print(f"Registered model '{name}' version '{version}'")


def get_latest_model(name: str) -> dict | None:
    """Get the latest version of a model."""
    registry = _load_registry()
    model = registry["models"].get(name)
    if not model:
        return None

    latest = model.get("latest")
    if not latest:
        return None

    return model["versions"][latest]


def list_models() -> dict:
    """List all registered models and their versions."""
    registry = _load_registry()
    summary = {}
    for name, data in registry["models"].items():
        summary[name] = {
            "latest": data.get("latest"),
            "versions": list(data["versions"].keys()),
        }
    return summary
''')

    # Agent integration — tool that uses the model
    _write(project_path / "tools" / "ml_predict.py", '''"""ML Prediction Tool — lets the agent use trained ML models.

This registers a tool that the agent can call to make predictions
using your trained models, bridging the ML pipeline with the agent.
"""
from tools.registry import Tool, registry


def predict(input_data: dict) -> dict:
    """Run prediction using the trained model.

    Args:
        input_data: Dictionary of feature values for prediction.

    Returns:
        Dictionary with prediction results.
    """
    try:
        from ml_models.inference.predictor import ModelPredictor
        predictor = ModelPredictor()
        result = predictor.predict(input_data)
        return {
            "prediction": result["prediction"],
            "confidence": result.get("confidence"),
            "model_version": result.get("model_version", "unknown"),
        }
    except FileNotFoundError:
        return {"error": "No trained model found. Run training first: python ml_models/training/train.py"}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# Register as an agent tool
registry.register(Tool(
    name="ml_predict",
    description="Make a prediction using the trained ML model. Provide input features as key-value pairs.",
    parameters={
        "type": "object",
        "properties": {
            "input_data": {
                "type": "object",
                "description": "Feature values for prediction. Keys are feature names, values are feature values.",
            }
        },
        "required": ["input_data"],
    },
    func=lambda input_data: predict(input_data),
))
''')


def _get_training_script(config: dict) -> str:
    """Generate framework-specific training script."""
    ml_framework = config.get("ml_framework", "sklearn")

    header = f'''"""Model Training Script — {config['project_name']}

Usage:
    python ml_models/training/train.py
    python ml_models/training/train.py --config ml_models/configs/training_config.yaml

Framework: {ml_framework}
"""
import sys
import os
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

'''

    if ml_framework == "sklearn":
        return header + '''import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from ml_models.registry import register_model


def train(config_path: str = "ml_models/configs/training_config.yaml"):
    """Train and evaluate models."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    test_size = config["experiment"]["test_size"]
    cv_folds = config["experiment"]["cross_validation_folds"]

    # ===== LOAD YOUR DATA =====
    # TODO: Replace with your actual data loading
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=seed)

    # If using data_processing:
    # from data_processing.loader import DataLoader
    # loader = DataLoader()
    # df = loader.load_processed("training_data")
    # X = df.drop(columns=["target"]).values
    # y = df["target"].values
    # ===========================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    # Define models to compare
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=seed
        ),
    }

    # Cross-validate and select best
    print("\\nCross-validation results:")
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Train the best model on full training set
    best_name = max(results, key=lambda k: results[k]["mean"])
    print(f"\\nBest model: {best_name}")

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\\n" + classification_report(y_test, y_pred))

    # Save model
    artifact_dir = Path(config["output"]["model_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    # Register
    register_model(
        name=config["experiment"]["name"],
        version="1.0",
        artifact_path=str(model_path),
        metrics={"accuracy": accuracy, "cv_mean": results[best_name]["mean"]},
        metadata={"model_type": best_name, "features": X_train.shape[1]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--config", default="ml_models/configs/training_config.yaml")
    args = parser.parse_args()
    train(args.config)
'''

    elif ml_framework == "pytorch":
        return header + '''import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml_models.registry import register_model


class SimpleNet(nn.Module):
    """Simple feedforward network. Replace with your architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train(config_path: str = "ml_models/configs/training_config.yaml"):
    """Train a PyTorch model."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== LOAD YOUR DATA =====
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=seed)
    # ===========================

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    # Model
    model = SimpleNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    patience_counter = 0
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        acc = accuracy_score(all_labels, all_preds)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f} — Acc: {acc:.4f}")

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "ml_models/artifacts/best_model.pt")
        else:
            patience_counter += 1
            if config["training"]["early_stopping"] and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\\nBest test accuracy: {best_acc:.4f}")
    print("Model saved to ml_models/artifacts/best_model.pt")

    register_model(
        name=config["experiment"]["name"],
        version="1.0",
        artifact_path="ml_models/artifacts/best_model.pt",
        metrics={"accuracy": best_acc},
        metadata={"framework": "pytorch", "input_dim": X_train.shape[1]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml_models/configs/training_config.yaml")
    args = parser.parse_args()
    train(args.config)
'''

    elif ml_framework == "xgboost":
        return header + '''import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from ml_models.registry import register_model


def train(config_path: str = "ml_models/configs/training_config.yaml"):
    """Train an XGBoost model with hyperparameter tuning."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]

    # ===== LOAD YOUR DATA =====
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=seed)
    # ===========================

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Hyperparameter grid
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    model = xgb.XGBClassifier(random_state=seed, eval_metric="logloss", use_label_encoder=False)

    print("Running grid search...")
    grid = GridSearchCV(
        model, param_grid, cv=config["experiment"]["cross_validation_folds"],
        scoring="accuracy", verbose=1, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"\\nBest params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Evaluate
    y_pred = grid.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\\n" + classification_report(y_test, y_pred))

    # Save
    model_path = "ml_models/artifacts/xgb_model.joblib"
    joblib.dump(grid.best_estimator_, model_path)
    print(f"Model saved to {model_path}")

    register_model(
        name=config["experiment"]["name"],
        version="1.0",
        artifact_path=model_path,
        metrics={"accuracy": accuracy, "cv_best": grid.best_score_},
        metadata={"best_params": grid.best_params_},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml_models/configs/training_config.yaml")
    args = parser.parse_args()
    train(args.config)
'''

    else:  # transformers
        return header + '''from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Dataset
from ml_models.registry import register_model
import numpy as np


def train(config_path: str = "ml_models/configs/training_config.yaml"):
    """Fine-tune a HuggingFace transformer model."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    model_name = "distilbert-base-uncased"  # TODO: Update to your model

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ===== LOAD YOUR DATA =====
    # TODO: Replace with your dataset
    # Option 1: From HuggingFace datasets
    dataset = load_dataset("imdb", split="train[:1000]")
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    # Option 2: From your data files
    # from data_processing.loader import DataLoader
    # loader = DataLoader()
    # df = loader.load_processed("training_data")
    # dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=seed)
    # ===========================

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="ml_models/artifacts/checkpoints",
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        accuracy = (predictions == eval_pred.label_ids).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    model_path = "ml_models/artifacts/fine_tuned_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    results = trainer.evaluate()
    print(f"\\nFinal evaluation: {results}")

    register_model(
        name=config["experiment"]["name"],
        version="1.0",
        artifact_path=model_path,
        metrics=results,
        metadata={"base_model": model_name, "framework": "transformers"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml_models/configs/training_config.yaml")
    args = parser.parse_args()
    train(args.config)
'''


def _get_inference_module(config: dict) -> str:
    """Generate framework-specific inference module."""
    ml_framework = config.get("ml_framework", "sklearn")

    header = '''"""Model Predictor — load trained model and run inference.

Used by the agent's ml_predict tool and directly in your application.
"""
from pathlib import Path

'''

    if ml_framework in ("sklearn", "xgboost"):
        model_file = "xgb_model.joblib" if ml_framework == "xgboost" else "best_model.joblib"
        return header + f'''import joblib
import numpy as np


class ModelPredictor:
    """Load and run inference with a trained scikit-learn/XGBoost model."""

    def __init__(self, model_path: str = "ml_models/artifacts/{model_file}"):
        self.model_path = Path(model_path)
        self._model = None

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No model found at {{self.model_path}}. "
                    "Run training first: python ml_models/training/train.py"
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, input_data: dict) -> dict:
        """Run prediction on input features.

        Args:
            input_data: Dictionary of feature_name -> value

        Returns:
            Dictionary with prediction and confidence
        """
        # Convert dict to feature array
        features = np.array([list(input_data.values())])

        prediction = self.model.predict(features)[0]

        result = {{
            "prediction": int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
            "model_version": "latest",
        }}

        # Add probabilities if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            result["confidence"] = float(max(proba))
            result["probabilities"] = [float(p) for p in proba]

        return result

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Run predictions on multiple records."""
        return [self.predict(record) for record in records]
'''

    elif ml_framework == "pytorch":
        return header + '''import torch
import torch.nn as nn
import numpy as np


class ModelPredictor:
    """Load and run inference with a trained PyTorch model."""

    def __init__(self, model_path: str = "ml_models/artifacts/best_model.pt"):
        self.model_path = Path(model_path)
        self._model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No model found at {self.model_path}. "
                    "Run training first: python ml_models/training/train.py"
                )
            # Import your model class
            from ml_models.training.train import SimpleNet
            self._model = SimpleNet(input_dim=20)  # TODO: match your architecture
            self._model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def predict(self, input_data: dict) -> dict:
        """Run prediction on input features."""
        features = torch.FloatTensor([list(input_data.values())]).to(self.device)

        with torch.no_grad():
            output = self.model(features)
            proba = torch.softmax(output, dim=1)[0]
            prediction = output.argmax(dim=1).item()

        return {
            "prediction": prediction,
            "confidence": float(proba[prediction]),
            "probabilities": [float(p) for p in proba],
            "model_version": "latest",
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Run predictions on multiple records."""
        return [self.predict(record) for record in records]
'''

    else:  # transformers
        return header + '''from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class ModelPredictor:
    """Load and run inference with a fine-tuned HuggingFace model."""

    def __init__(self, model_path: str = "ml_models/artifacts/fine_tuned_model"):
        self.model_path = Path(model_path)
        self._model = None
        self._tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No model at {self.model_path}. "
                    "Run training first: python ml_models/training/train.py"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()

    def predict(self, input_data: dict) -> dict:
        """Run prediction on text input."""
        self._load()
        text = input_data.get("text", str(input_data))

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model(**inputs)
            proba = torch.softmax(output.logits, dim=1)[0]
            prediction = output.logits.argmax(dim=1).item()

        return {
            "prediction": prediction,
            "confidence": float(proba[prediction]),
            "probabilities": [float(p) for p in proba],
            "model_version": "latest",
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        return [self.predict(record) for record in records]
'''


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
