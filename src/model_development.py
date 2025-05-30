import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.validation import check_is_fitted


# Function to train a single model
def train_model(model, X_train, y_train):
    """
    Melatih satu model ML Sklearn

    Args:
        model (): sklearn model object
        X_train (array atau dataframe): fitur data latih
        y_train (array atau series): target data latih

    Returns:
        model: model yang sudah dilatih
        training_time: waktu pelatihan model
    """

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


# Function to evaluate a model
def evaluate_model(model, X, y):
    """
    Mengevaluasi model yang sudah dilatih

    Args:
        model (): fitted model sklearn object
        X (array atau dataframe): fitur dari data
        y (array atau series):  target dari data

    Returns:
        hasil evaluasi: akurasi, presisi, recall, f1,
        confusion matrix dan classification report
    """

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred)
    return accuracy, precision, recall, f1, cm, cr


def plot_confusion_matrix(cm):
    """
    Plot confusion matrix menggunakn seaborn heatmap.

    Args:
        cm (array): array dengan ukuran (n_classes, n_classes).
    """

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.show()


# Function to print evaluation results
def print_evaluation_results(model_name, train_metrics, test_metrics):
    """
    Menampilkan haasil evaluasi model

    Args:
        model_name (string): nama model
        train_metrics (hasil evaluasi): hasil evaluasi dari fungsi evaluate_model()
        test_metrics (hasil evaluasi): hasil evaluasi dari fungsi evaluate_model()
    """

    print("=" * 100)
    print(f" {model_name} Classifier ".center(100, "="))
    print("=" * 100)

    print("Training")
    print(f"Train Accuracy: {train_metrics['Accuracy']}")
    print(f"Train Precision: {train_metrics['Precision']}")
    print(f"Train Recall: {train_metrics['Recall']}")
    print(f"Train F1-Score: {train_metrics['F1']}")
    print("Confusion Matrix:")
    plot_confusion_matrix(train_metrics["Confusion Matrix"])

    print("\nTesting")
    print(f"Test Accuracy: {test_metrics['Accuracy']}")
    print(f"Test Precision: {test_metrics['Precision']}")
    print(f"Test Recall: {test_metrics['Recall']}")
    print(f"Test F1-Score: {test_metrics['F1']}")
    print("Classification Report: ")
    print(test_metrics["Classification Report"])
    print("Confusion Matrix:")
    plot_confusion_matrix(test_metrics["Confusion Matrix"])


# Main function to train and evaluate multiple models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Melatih dan mengevaluasi banyak model sekaligus

    Args:
        X_train (array atau dataframe): fitur data latih
        X_test (array atau dataframe): fitur data uji
        y_train (array atau series): target data latih
        y_test (array atau series): target data uji
        models (dict): kumpulan model dalam dictionary

    Returns:
        result_df: dataframe hasil evaluasi
    """

    evaluation_results = []

    for model_name, model in models.items():
        # Train the model
        trained_model, training_time = train_model(model, X_train, y_train)

        # Evaluate on training data
        train_accuracy, train_precision, train_recall, train_f1, train_cm, train_cr = (
            evaluate_model(trained_model, X_train, y_train)
        )

        # Evaluate on testing data
        test_accuracy, test_precision, test_recall, test_f1, test_cm, test_cr = (
            evaluate_model(trained_model, X_test, y_test)
        )

        # Print results
        print_evaluation_results(
            model_name,
            {
                "Accuracy": train_accuracy,
                "Precision": train_precision,
                "Recall": train_recall,
                "F1": train_f1,
                "Confusion Matrix": train_cm,
                "Classification Report": train_cr,
            },
            {
                "Accuracy": test_accuracy,
                "Precision": test_precision,
                "Recall": test_recall,
                "F1": test_f1,
                "Confusion Matrix": test_cm,
                "Classification Report": test_cr,
            },
        )

        # Store results
        evaluation_results.append(
            {
                "Model": model_name,
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Train Precision": train_precision,
                "Test Precision": test_precision,
                "Train Recall": train_recall,
                "Test Recall": test_recall,
                "Train F1": train_f1,
                "Test F1": test_f1,
                "Training Time (s)": training_time,
            }
        )

    result_df = pd.DataFrame(evaluation_results)

    return result_df


def save_model(model, file_path):
    """
    Menyimpan model ke dalam file menggunakan joblib.

    Args:
        model (): sklearn mdoel yang sudah di-fit
        file_path (string): path untuk menyimpan model
    """

    import joblib

    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def is_model_fitted(model):
    """
    Mengecek apakah model sklearn sudah di-fit atau belum.

    Args:
        model: Estimator sklearn

    Returns:
        bool: True jika sudah di-fit, False jika belum
    """
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False


def save_params(params: dict, path: str | Path) -> None:
    """
    Menyimpan parameter model ke file JSON.

    Args:
        params (dict): Dictionary parameter model (contoh: best_params_).
        path (str or Path): Path ke file JSON tempat menyimpan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Parameter berhasil disimpan di: {path}")


def load_params(path: str | Path) -> dict:
    """
    Membaca parameter model dari file JSON.

    Args:
        path (str or Path): Path ke file JSON.

    Returns:
        dict: Dictionary parameter model.
    """
    path = Path(path)
    with open(path, "r") as f:
        params = json.load(f)
    print(f"Parameter berhasil dimuat dari: {path}")
    return params


def train_and_evaluate_models_cv(
    X, X_test, y, y_test, models, n_splits=5, scoring="accuracy", random_state=None
):
    evaluation_results = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for model_name, model in models.items():
        start_time = time.time()
        score = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        training_time_cv = time.time() - start_time

        # Train the model
        trained_model, training_time_fit = train_model(model, X, y)

        y_pred = trained_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)

        # Evaluate on testing data
        test_accuracy, test_precision, test_recall, test_f1, test_cm, test_cr = (
            evaluate_model(trained_model, X_test, y_test)
        )

        avg_score = score.mean()
        std_score = score.std()

        # Store results
        evaluation_results.append(
            {
                "Model": model_name,
                "Avg Score": avg_score,
                "Std Score": std_score,
                "Training Time CV (s)": training_time_cv,
                "Test AUC": test_auc,
                "Test Accuracy": test_accuracy,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1": test_f1,
                "Training Time Fit (s)": training_time_fit,
            }
        )

    result_df = pd.DataFrame(evaluation_results)

    return result_df.sort_values(by="Avg Score", ascending=False)
