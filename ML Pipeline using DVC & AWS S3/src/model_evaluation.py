import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dvclive import Live


# ============================================================
# Directory Setup
# ============================================================

LOG_DIR = "logs"
MODEL_PATH = "models/model.pkl"
TEST_DATA_PATH = "data/processed/test_tfidf.csv"
PARAMS_PATH = "params.yaml"
METRICS_PATH = "reports/metrics.json"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ============================================================
# Logging Configuration
# ============================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers if script is executed multiple times
if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(LOG_DIR, "model_evaluation.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Load Parameters
# ============================================================

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters loaded from %s", params_path)
        return params

    except FileNotFoundError:
        logger.error("Parameters file not found: %s", params_path)
        raise

    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading parameters: %s", e)
        raise


# ============================================================
# Load Model
# ============================================================

def load_model(file_path: str):
    """Load the trained model from a pickle file."""

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully from %s", file_path)
        return model

    except FileNotFoundError:
        logger.error("Model file not found: %s", file_path)
        raise

    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


# ============================================================
# Load Test Data
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load test data from CSV."""

    try:
        df = pd.read_csv(file_path)

        logger.info(
            "Test data loaded successfully from %s",
            file_path
        )

        logger.info(
            "Test data shape: %s",
            df.shape
        )

        return df

    except FileNotFoundError:
        logger.error("Test data file not found: %s", file_path)
        raise

    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the model and return evaluation metrics."""

    try:

        # Generate predictions
        y_pred = clf.predict(X_test)

        # Generate probability predictions
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

        else:
            y_pred_proba = None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # AUC only if probability predictions are available
        if y_pred_proba is not None:
            auc = roc_auc_score(
                y_test,
                y_pred_proba
            )
        else:
            auc = None

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc) if auc is not None else None
        }

        logger.info("Model evaluation completed")

        logger.info("Accuracy  : %.4f", accuracy)
        logger.info("Precision : %.4f", precision)
        logger.info("Recall    : %.4f", recall)
        logger.info("F1 Score  : %.4f", f1)

        if auc is not None:
            logger.info("AUC       : %.4f", auc)

        return metrics

    except Exception as e:
        logger.error(
            "Error during model evaluation: %s",
            e
        )
        raise


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(
    metrics: dict,
    file_path: str
) -> None:
    """Save evaluation metrics to JSON."""

    try:

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info(
            "Metrics saved successfully to %s",
            file_path
        )

    except Exception as e:
        logger.error(
            "Error saving metrics: %s",
            e
        )
        raise


# ============================================================
# Main Function
# ============================================================

def main():

    try:

        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 60)

        # ----------------------------------------------------
        # Load parameters
        # ----------------------------------------------------

        params = load_params(PARAMS_PATH)

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        clf = load_model(MODEL_PATH)

        # ----------------------------------------------------
        # Load test data
        # ----------------------------------------------------

        test_data = load_data(TEST_DATA_PATH)

        # ----------------------------------------------------
        # Split features and target
        # ----------------------------------------------------

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.info(
            "X_test shape: %s",
            X_test.shape
        )

        logger.info(
            "y_test shape: %s",
            y_test.shape
        )

        # ----------------------------------------------------
        # Evaluate model
        # ----------------------------------------------------

        metrics = evaluate_model(
            clf,
            X_test,
            y_test
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        save_metrics(
            metrics,
            METRICS_PATH
        )

        # ----------------------------------------------------
        # DVC Live Experiment Tracking
        # ----------------------------------------------------

        with Live(save_dvc_exp=True) as live:

            # Log actual evaluation metrics
            live.log_metric(
                "accuracy",
                metrics["accuracy"]
            )

            live.log_metric(
                "precision",
                metrics["precision"]
            )

            live.log_metric(
                "recall",
                metrics["recall"]
            )

            live.log_metric(
                "f1_score",
                metrics["f1_score"]
            )

            if metrics["auc"] is not None:
                live.log_metric(
                    "auc",
                    metrics["auc"]
                )

            # Log parameters
            if params:
                live.log_params(params)

        logger.info("=" * 60)
        logger.info("Model Evaluation Completed Successfully")
        logger.info("=" * 60)

    except Exception as e:

        logger.error(
            "Failed to complete model evaluation: %s",
            e
        )

        print(f"Error: {e}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()