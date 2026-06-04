from google.colab import drive
drive.mount('/content/drive')

import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score
)

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")

DATASET_FOLDER = "/content/drive/MyDrive/filesddos"
RESULTS_FOLDER = "/content/drive/MyDrive/filesddos/results"
GRAPHS_FOLDER = "/content/drive/MyDrive/filesddos/results/graphs"
MODELS_FOLDER = "/content/drive/MyDrive/filesddos/results/models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_FEATURES_COUNT = 30
CV_FOLDS = 5

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)


def log(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_dataset(folder_path):
    log("Поиск файлов датасета...")
    files = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.endswith("training.parquet"):
            files.append(full_path)
        elif file.endswith("training.csv"):
            files.append(full_path)
    if len(files) == 0:
        raise Exception("Не найдено файлов training.parquet или training.csv")
    log(f"Найдено файлов: {len(files)}")
    dataframes = []
    for file in files:
        log(f"Загрузка: {os.path.basename(file)}")
        if file.endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file, low_memory=False)
        dataframes.append(df)
    dataset = pd.concat(dataframes, ignore_index=True)
    log(f"Датасет загружен: {dataset.shape[0]} строк, {dataset.shape[1]} столбцов")
    return dataset


def explore_dataset(dataset):
    log("Разведочный анализ данных (EDA)...")

    print("\nПервые 5 строк:")
    print(dataset.head())

    print("\nИнформация о типах данных:")
    print(dataset.dtypes.value_counts())

    print("\nСтатистика по числовым признакам:")
    print(dataset.describe().T.head(20))

    print("\nПропущенные значения (топ-10):")
    missing = dataset.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing.head(10))
    else:
        print("Пропущенных значений нет")

    print("\nБесконечные значения:")
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(dataset[numeric_cols]).sum()
    inf_count = inf_count[inf_count > 0].sort_values(ascending=False)
    if len(inf_count) > 0:
        print(inf_count.head(10))
    else:
        print("Бесконечных значений нет")

    if "Label" in dataset.columns:
        print("\nРаспределение классов:")
        class_dist = dataset["Label"].value_counts()
        print(class_dist)
        print("\nДоля каждого класса (%):")
        print((class_dist / len(dataset) * 100).round(2))


def show_class_distribution(y, title="Распределение классов"):
    plt.figure(figsize=(12, 6))
    counts = y.value_counts()
    bars = plt.bar(range(len(counts)), counts.values, color=sns.color_palette("viridis", len(counts)))
    plt.xticks(range(len(counts)), counts.index, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "class_distribution.png"))
    plt.show()


def show_class_pie(y, title="Доли классов"):
    plt.figure(figsize=(10, 10))
    counts = y.value_counts()
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("viridis", len(counts)))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "class_pie.png"))
    plt.show()


def clean_data(dataset):
    log("Очистка данных...")
    df = dataset.copy()
    df.columns = [col.strip() for col in df.columns]
    initial_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_rows = df.isnull().any(axis=1).sum()
    log(f"Строк с NaN/Inf: {nan_rows}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    df = df.fillna(0)
    duplicates = df.duplicated().sum()
    log(f"Дубликатов: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        log(f"Удалено дубликатов: {duplicates}")
    log(f"Итого строк: {len(df)} (было {initial_rows})")
    return df


def normalize_labels_binary(y):
    y = y.astype(str).str.strip()
    y = y.apply(lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK")
    return y


def normalize_labels_multi(y):
    y = y.astype(str).str.strip()
    return y


def prepare_features(dataset, label_col="Label"):
    log("Подготовка признаков...")
    if label_col not in dataset.columns:
        raise Exception(f"Колонка {label_col} не найдена")
    df = dataset.copy()
    y_raw = df[label_col]
    X = df.drop(label_col, axis=1)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    low_var_cols = X.columns[X.var() < 0.001]
    if len(low_var_cols) > 0:
        log(f"Удалено признаков с низкой дисперсией: {len(low_var_cols)}")
        X = X.drop(columns=low_var_cols)
    log(f"Количество признаков: {X.shape[1]}")
    log(f"Количество объектов: {X.shape[0]}")
    return X, y_raw


def show_correlation_matrix(X, top_n=20):
    log("Построение корреляционной матрицы...")
    if X.shape[1] > top_n:
        variances = X.var().sort_values(ascending=False)
        top_cols = variances.head(top_n).index.tolist()
        X_sub = X[top_cols]
    else:
        X_sub = X
    corr = X_sub.corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(f"Корреляционная матрица (топ-{len(X_sub.columns)} признаков)")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "correlation_matrix.png"))
    plt.show()


def show_feature_distributions(X, y, features=None, top_n=6):
    log("Построение распределений признаков...")
    if features is None:
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Std": X.std()
        }).sort_values(by="Std", ascending=False)
        features = feat_imp.head(top_n)["Feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(features[:6]):
        ax = axes[i]
        col_data = X[feat].copy()
        q_low = col_data.quantile(0.02)
        q_high = col_data.quantile(0.98)
        for label in sorted(y.unique()):
            subset = col_data[y == label]
            subset = subset[(subset >= q_low) & (subset <= q_high)]
            if len(subset) > 0:
                ax.hist(subset, bins=50, alpha=0.5, label=label, density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=8)
    plt.suptitle("Распределение признаков по классам", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "feature_distributions.png"))
    plt.show()


def select_features(X, y, k=30):
    log(f"Отбор {k} лучших признаков (SelectKBest)...")
    y_encoded = LabelEncoder().fit_transform(y)
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y_encoded)
    mask = selector.get_support()
    selected_columns = X.columns[mask].tolist()
    scores = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)
    log(f"Отобрано признаков: {len(selected_columns)}")
    print("\nТоп-10 признаков по SelectKBest:")
    print(scores.head(10))
    scores.to_csv(os.path.join(RESULTS_FOLDER, "feature_selection_scores.csv"), index=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=scores.head(20), x="Score", y="Feature")
    plt.title("Top-20 признаков (SelectKBest F-statistic)")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "feature_selection_scores.png"))
    plt.show()
    return X[selected_columns], selected_columns, scores


def scale_features(X_train, X_test):
    log("Стандартизация признаков...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def train_single_model(model, model_name, X_train, X_test, y_train, y_test):
    log(f"Обучение модели: {model_name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    log(f"{model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}, train={train_time:.1f}s, predict={predict_time:.1f}s")
    return {
        "model_name": model_name,
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_time": train_time,
        "predict_time": predict_time
    }


def compare_models(X_train, X_test, y_train, y_test, mode_name="binary"):
    log(f"Сравнение моделей ({mode_name})...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    results = []
    for name, model in models.items():
        result = train_single_model(model, name, X_train, X_test, y_train, y_test)
        results.append(result)
    results_df = pd.DataFrame([{
        "Модель": r["model_name"],
        "Accuracy": round(r["accuracy"], 4),
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4),
        "F1-score": round(r["f1"], 4),
        "Время обучения (с)": round(r["train_time"], 2),
        "Время предсказания (с)": round(r["predict_time"], 2)
    } for r in results])
    print(f"\nСравнение моделей ({mode_name}):")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_FOLDER, f"model_comparison_{mode_name}.csv"), index=False)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    for i, metric in enumerate(metrics_names):
        ax = axes[i]
        bars = ax.bar(results_df["Модель"], results_df[metric], color=sns.color_palette("viridis", len(results_df)))
        ax.set_title(metric)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, str(val), ha="center", va="bottom", fontsize=8)
    plt.suptitle(f"Сравнение моделей ({mode_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, f"model_comparison_{mode_name}.png"))
    plt.show()
    return results


def cross_validate_model(model, X, y, model_name="Random Forest"):
    log(f"Кросс-валидация: {model_name} ({CV_FOLDS} фолдов)...")
    y_encoded = LabelEncoder().fit_transform(y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\nКросс-валидация {model_name}:")
    for i, score in enumerate(scores):
        print(f"  Фолд {i+1}: {score:.4f}")
    print(f"  Среднее: {scores.mean():.4f} (+/- {scores.std():.4f})")
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, CV_FOLDS+1), scores, color=sns.color_palette("viridis", CV_FOLDS))
    plt.axhline(y=scores.mean(), color="red", linestyle="--", label=f"Среднее: {scores.mean():.4f}")
    plt.xlabel("Фолд")
    plt.ylabel("Accuracy")
    plt.title(f"Кросс-валидация {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "cross_validation.png"))
    plt.show()
    return scores


def show_confusion_matrix(y_test, predictions, title="Confusion Matrix"):
    labels = sorted(list(pd.Series(y_test).astype(str).unique()))
    cm = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(max(10, len(labels)), max(8, len(labels)*0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(GRAPHS_FOLDER, f"confusion_matrix_{safe_title}.png"))
    plt.show()


def show_classification_report(y_test, predictions, title="Classification Report"):
    print(f"\n{title}:")
    print(classification_report(y_test, predictions, zero_division=0))
    report_dict = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    safe_title = title.replace(" ", "_").replace("/", "_")
    report_df.to_csv(os.path.join(RESULTS_FOLDER, f"classification_report_{safe_title}.csv"))


def show_roc_curve(model, X_test, y_test, title="ROC Curve"):
    log("Построение ROC-кривой...")
    classes = sorted(y_test.unique())
    if len(classes) == 2:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            log("Модель не поддерживает predict_proba, ROC пропущена")
            return
        y_binary = (y_test == classes[1]).astype(int)
        fpr, tpr, thresholds = roc_curve(y_binary, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_FOLDER, "roc_curve.png"))
        plt.show()
        log(f"AUC = {roc_auc:.4f}")
    else:
        log("ROC-кривая для многоклассовой задачи пропущена (>2 класса)")


def show_feature_importance(model, feature_columns, top_n=20):
    log("Анализ важности признаков...")
    if not hasattr(model, "feature_importances_"):
        log("Модель не поддерживает feature_importances_")
        return None
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    print(f"\nТоп-{top_n} важных признаков (Random Forest):")
    print(feat_imp.head(top_n).to_string(index=False))
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feat_imp.head(top_n), x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top-{top_n} важных признаков (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "feature_importance_rf.png"))
    plt.show()
    feat_imp.to_csv(os.path.join(RESULTS_FOLDER, "feature_importance.csv"), index=False)
    return feat_imp


def save_model_bundle(model, scaler, feature_columns, metrics, mode_name):
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(feature_columns),
        "mode": mode_name,
        "metrics": metrics
    }
    model_path = os.path.join(MODELS_FOLDER, f"ddos_model_{mode_name}.pkl")
    metrics_path = os.path.join(RESULTS_FOLDER, f"metrics_{mode_name}.json")
    joblib.dump(bundle, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"Модель сохранена: {model_path}")
    log(f"Метрики сохранены: {metrics_path}")
    return model_path


def predict_file(file_path, model_bundle_path):
    log(f"Предсказание для файла: {os.path.basename(file_path)}")
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path, low_memory=False)
    else:
        raise Exception("Поддерживаются только .parquet и .csv")
    original_df = df.copy()
    if "Label" in df.columns:
        df = df.drop("Label", axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    X = df.select_dtypes(include=[np.number])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    X = X.fillna(0)
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    predictions = model.predict(X)
    result_df = original_df.copy()
    result_df["Prediction"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        class_names = model.classes_
        for i, class_name in enumerate(class_names):
            result_df["Prob_" + str(class_name)] = probabilities[:, i]
    output_path = os.path.join(RESULTS_FOLDER, "predictions_output.csv")
    result_df.to_csv(output_path, index=False)
    log(f"Предсказания сохранены: {output_path}")
    print("\nРезультат (первые 10 строк):")
    print(result_df[["Prediction"]].head(10))
    pred_counts = result_df["Prediction"].value_counts()
    print("\nРаспределение предсказаний:")
    print(pred_counts)
    return result_df


def predict_folder(folder_path, model_bundle_path):
    log(f"Пакетное предсказание для папки: {folder_path}")
    files = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.endswith(".parquet") or file.endswith(".csv"):
            if "training" not in file.lower():
                files.append(full_path)
    if len(files) == 0:
        log("Нет файлов для предсказания")
        return None
    log(f"Найдено файлов: {len(files)}")
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    all_results = []
    for file in files:
        log(f"Обработка: {os.path.basename(file)}")
        if file.endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file, low_memory=False)
        original_df = df.copy()
        if "Label" in df.columns:
            df = df.drop("Label", axis=1)
        df = df.replace([np.inf, -np.inf], np.nan)
        X = df.select_dtypes(include=[np.number])
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        X = X.fillna(0)
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        predictions = model.predict(X)
        original_df["Prediction"] = predictions
        original_df["SourceFile"] = os.path.basename(file)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            class_names = model.classes_
            for i, class_name in enumerate(class_names):
                original_df["Prob_" + str(class_name)] = probabilities[:, i]
        all_results.append(original_df)
    final_df = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(RESULTS_FOLDER, "all_predictions.csv")
    final_df.to_csv(output_path, index=False)
    log(f"Все результаты сохранены: {output_path}")
    print("\nРаспределение предсказаний по всем файлам:")
    print(final_df["Prediction"].value_counts())
    return final_df


def run_experiment(X, y, mode_name):
    log(f"========== ЭКСПЕРИМЕНТ: {mode_name} ==========")

    show_class_distribution(y, title=f"Распределение классов ({mode_name})")
    show_class_pie(y, title=f"Доли классов ({mode_name})")

    X_selected, selected_columns, feature_scores = select_features(X, y, k=TOP_FEATURES_COUNT)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    comparison_results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test, mode_name)

    best_result = max(comparison_results, key=lambda x: x["f1"])
    log(f"Лучшая модель: {best_result['model_name']} (F1={best_result['f1']:.4f})")

    rf_result = next((r for r in comparison_results if r["model_name"] == "Random Forest"), best_result)
    rf_model = rf_result["model"]
    rf_predictions = rf_result["predictions"]

    show_confusion_matrix(y_test, rf_predictions, title=f"Confusion Matrix ({mode_name})")
    show_classification_report(y_test, rf_predictions, title=f"Report_{mode_name}")

    if mode_name == "binary":
        show_roc_curve(rf_model, X_test_scaled, y_test, title=f"ROC Curve ({mode_name})")

    feat_imp = show_feature_importance(rf_model, selected_columns, top_n=20)

    show_correlation_matrix(X_selected, top_n=20)
    show_feature_distributions(X_selected, y, top_n=6)

    rf_fresh = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
    cv_scores = cross_validate_model(rf_fresh, X_selected, y, model_name="Random Forest")

    metrics = {
        "mode": mode_name,
        "accuracy": float(rf_result["accuracy"]),
        "precision_weighted": float(rf_result["precision"]),
        "recall_weighted": float(rf_result["recall"]),
        "f1_weighted": float(rf_result["f1"]),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "train_time": float(rf_result["train_time"]),
        "num_features": len(selected_columns),
        "num_samples_train": len(X_train),
        "num_samples_test": len(X_test)
    }

    model_path = save_model_bundle(rf_model, scaler, selected_columns, metrics, mode_name)

    return model_path, metrics


def main():
    log("===== ИНТЕЛЛЕКТУАЛЬНЫЙ МОДУЛЬ АНАЛИЗА ТРАФИКА =====")
    log("===== Обнаружение DDoS-атак (CIC-DDoS2019) =====")

    dataset = load_dataset(DATASET_FOLDER)
    explore_dataset(dataset)
    dataset = clean_data(dataset)
    X, y_raw = prepare_features(dataset)

    log("--- Бинарная классификация ---")
    y_binary = normalize_labels_binary(y_raw)
    binary_model_path, binary_metrics = run_experiment(X, y_binary, "binary")

    log("--- Многоклассовая классификация ---")
    y_multi = normalize_labels_multi(y_raw)
    multi_model_path, multi_metrics = run_experiment(X, y_multi, "multiclass")

    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ РЕЖИМОВ")
    print("="*60)
    comparison = pd.DataFrame([binary_metrics, multi_metrics])
    print(comparison[["mode", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cv_mean_accuracy"]].to_string(index=False))
    comparison.to_csv(os.path.join(RESULTS_FOLDER, "final_comparison.csv"), index=False)

    log("===== МОДУЛЬ ЗАВЕРШИЛ РАБОТУ =====")
    log(f"Результаты: {RESULTS_FOLDER}")
    log(f"Графики: {GRAPHS_FOLDER}")
    log(f"Модели: {MODELS_FOLDER}")

    return binary_model_path, multi_model_path


binary_model_path, multi_model_path = main()
