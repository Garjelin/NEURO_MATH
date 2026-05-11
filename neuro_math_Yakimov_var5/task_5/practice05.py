"""
Практическая работа №5, вариант 5.

Задание 1 — таблица 5.16: классификация «требуется ремонт двигателя»
по температуре подшипника и уровню вибрации (НС — `MLPClassifier`).

Задание 2 — таблица 5.21, вариант 5: самостоятельная постановка задачи классификации:
прогноз метеорологических условий (наличие осадков на следующий шаг временного ряда)
на синтетических исторических признаках: температура, влажность, давление.

Разбиение задания 1 — stratified train/test 80%/20% на общих 10 наблюдениях из методички.
Разбиение задания 2 — хронологическое (без shuffle), чтобы не смешивать будущее в прошлое.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- Задание 1: таблица 5.16 (температура, °C; вибрация, мм/с; класс: да/нет) -------------
TABLE_516 = np.array(
    [
        [125.0, 1.80],
        [100.0, 1.00],
        [70.0, 0.90],
        [80.0, 0.76],
        [105.0, 1.10],
        [85.0, 1.90],
        [60.0, 0.60],
        [65.0, 0.50],
        [90.0, 1.50],
        [86.0, 1.70],
    ],
    dtype=float,
)

LABEL_YES_NO = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
LABEL_NAMES = {0: "нет", 1: "да"}


def run_task1_table_classification(
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """
    Обучение `MLPClassifier` по всей таблице 5.16; контроль — stratified hold-out.
    Класс 1 — «да», класс 0 — «нет».
    """
    X = TABLE_516.copy()
    y = LABEL_YES_NO.copy()
    idx_all = np.arange(len(y))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        idx_all,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="adam",
        max_iter=8000,
        random_state=random_state,
        alpha=1e-4,
        tol=1e-7,
    )
    clf.fit(X_tr, y_train)

    y_pred_train = clf.predict(X_tr)
    y_pred_test = clf.predict(X_te)

    def _metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }

    train_rows = tuple(int(i) + 1 for i in sorted(idx_train))
    test_rows = tuple(int(i) + 1 for i in sorted(idx_test))

    return {
        "table_rows_total": len(y),
        "train_rows_1based": train_rows,
        "test_rows_1based": test_rows,
        "scaler": scaler,
        "model": clf,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "metrics_train": _metrics(y_train, y_pred_train),
        "metrics_test": _metrics(y_test, y_pred_test),
        "confusion_train": confusion_matrix(y_train, y_pred_train, labels=[0, 1]),
        "confusion_test": confusion_matrix(y_test, y_pred_test, labels=[0, 1]),
        "report_test": classification_report(
            y_test,
            y_pred_test,
            labels=[0, 1],
            target_names=["нет", "да"],
            zero_division=0,
        ),
        "random_state": random_state,
        "test_size": test_size,
    }


def _save_task1_boundary(path: Path, task1: dict) -> None:
    clf: MLPClassifier = task1["model"]
    scaler: StandardScaler = task1["scaler"]
    X = np.vstack([task1["X_train"], task1["X_test"]])
    y = np.concatenate([task1["y_train"], task1["y_test"]])

    t_min, t_max = float(X[:, 0].min()) - 8.0, float(X[:, 0].max()) + 8.0
    v_min, v_max = float(X[:, 1].min()) - 0.15, float(X[:, 1].max()) + 0.35

    tt, vv = np.meshgrid(
        np.linspace(t_min, t_max, 250),
        np.linspace(v_min, v_max, 250),
        indexing="xy",
    )
    grid = np.column_stack([tt.ravel(), vv.ravel()])
    Z = clf.predict_proba(scaler.transform(grid))[:, 1].reshape(tt.shape)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cf = ax.contourf(tt, vv, Z, levels=25, cmap="RdYlBu_r", alpha=0.75)
    ax.contour(tt, vv, Z, levels=[0.5], colors="black", linewidths=1.6)
    for yi in (0, 1):
        mask = y == yi
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            label=f'Факт: «{LABEL_NAMES[yi]}»',
            s=120,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
    ax.set_xlabel("Температура подшипника, °C")
    ax.set_ylabel("Вибрация, мм/с")
    ax.set_title("Задание 1: вероятность класса «да» и разделяющая кривая p=0.5")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="P(ремонт)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_confusion(path: Path, matrix: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["нет", "да"])
    ax.set_yticklabels(["нет", "да"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Факт")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_meteo_series_classification(
    *,
    n_days: int = 520,
    train_ratio: float = 0.78,
    seed: int = 2026,
) -> dict:
    """
    Синтетический «метеорологический» ряд с признаками на шаге t и меткой осадков на шаге t+1.

    Признаки на момент t:
      — температура воздуха (условные °C),
      — относительная влажность (%),
      — давление (условные гПа).
    Целевая переменная y[t]: наличие осадков на день t+1 (бинарная классификация).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days, dtype=float)

    seasonal_temp = 6.0 + 14.0 * np.sin(2 * np.pi * t / 365.25)
    temp = seasonal_temp + rng.normal(0.0, 2.2, n_days)

    humidity = np.clip(35.0 + 0.55 * temp + rng.normal(0.0, 11.0, n_days), 8.0, 98.0)
    pressure = 1013.0 + rng.normal(0.0, 10.0, n_days) - 0.07 * humidity + 0.02 * temp

    dp = np.diff(pressure, prepend=pressure[0])

    score = (
        0.22 * humidity[:-1]
        + 0.055 * temp[:-1]
        - 0.11 * dp[:-1]
        + 0.004 * (pressure[:-1] - 1013.0)
        + rng.normal(0.0, 6.5, n_days - 1)
    )
    # Порог по медиане даёт около 50% «осадков» и смесь классов на типичном окне времени
    thresh = float(np.median(score))
    rain_next = (score > thresh).astype(int)

    X = np.column_stack([temp[:-1], humidity[:-1], pressure[:-1]])
    y = rain_next

    n_eff = len(y)
    # Подбираем долю обучения так, чтобы на хронологическом тесте были оба класса
    n_train = max(2, int(round(train_ratio * n_eff)))
    found = len(np.unique(y[n_train:])) == 2 and len(np.unique(y[:n_train])) == 2
    if not found:
        for frac in np.linspace(0.55, 0.92, 45):
            nt = max(2, min(n_eff - 2, int(round(frac * n_eff))))
            if len(np.unique(y[nt:])) == 2 and len(np.unique(y[:nt])) == 2:
                n_train = nt
                found = True
                break
    if not found:
        n_train = max(2, int(round(train_ratio * n_eff)))

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(36, 18),
        activation="tanh",
        solver="adam",
        max_iter=12000,
        random_state=seed,
        alpha=1e-4,
        tol=1e-7,
    )
    clf.fit(X_tr, y_train)

    y_pred_train = clf.predict(X_tr)
    y_pred_test = clf.predict(X_te)

    def _metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }

    return {
        "n_days_source": n_days,
        "n_supervised": n_eff,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_ratio_effective": float(n_train / max(1, n_eff)),
        "train_ratio_config": train_ratio,
        "seed": seed,
        "feature_names": ["T(t)", "RH(t)", "P(t)"],
        "target_name": "осадки на день t+1",
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "metrics_train": _metrics(y_train, y_pred_train),
        "metrics_test": _metrics(y_test, y_pred_test),
        "confusion_train": confusion_matrix(y_train, y_pred_train, labels=[0, 1]),
        "confusion_test": confusion_matrix(y_test, y_pred_test, labels=[0, 1]),
        "report_test": classification_report(
            y_test,
            y_pred_test,
            labels=[0, 1],
            target_names=["нет осадков", "есть осадки"],
            zero_division=0,
        ),
        # humidity[:-1] и rain_next одинаковой длины n_days-1 (метка на t→t+1)
        "plot_tail_humidity": humidity[:-1][-140:],
        "plot_tail_rain": rain_next[-140:],
        "model": clf,
        "scaler": scaler,
    }


def _save_task2_plot(path: Path, task2: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hum = task2["plot_tail_humidity"]
    rain_aligned = task2["plot_tail_rain"]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)

    axes[0].plot(hum, color="steelblue", lw=1.2)
    axes[0].set_ylabel("Влажность, %")
    axes[0].set_title("Задание 2 (вариант 5): фрагмент синтетического ряда")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(np.arange(len(rain_aligned)), rain_aligned, where="mid", color="darkgreen", lw=1.4)
    axes[1].set_ylabel("Осадки t→t+1")
    axes[1].set_xlabel("Шаг (хвост ряда)")
    axes[1].set_yticks([0.0, 1.0])
    axes[1].set_yticklabels(["нет", "да"])
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_task2_meteo_classification() -> dict:
    return build_meteo_series_classification()


def run_practice05(out_dir: Path | str | None = None, save_png: bool = True) -> dict:
    """
    Полный прогон практики №5. Консольный вывод не выполняется — см. `print_results`.
    """
    out = Path(out_dir) if out_dir is not None else Path("outputs") / "practice05"
    out.mkdir(parents=True, exist_ok=True)

    task1 = run_task1_table_classification()
    task2 = run_task2_meteo_classification()

    p_boundary = out / "practice05_task1_boundary.png"
    p_cm1 = out / "practice05_task1_confusion_test.png"
    p_cm2 = out / "practice05_task2_confusion_test.png"
    p_series = out / "practice05_task2_series_tail.png"

    if save_png:
        _save_task1_boundary(p_boundary, task1)
        _save_confusion(p_cm1, task1["confusion_test"], "Задание 1 — матрица ошибок (test)")
        _save_confusion(p_cm2, task2["confusion_test"], "Задание 2 — матрица ошибок (test)")
        _save_task2_plot(p_series, task2)

    return {
        "out_dir": str(out.resolve()),
        "task1": task1,
        "task2": task2,
        "plots": {
            "task1_boundary": str(p_boundary) if save_png else None,
            "task1_confusion_test": str(p_cm1) if save_png else None,
            "task2_confusion_test": str(p_cm2) if save_png else None,
            "task2_series": str(p_series) if save_png else None,
        },
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    bar = "=" * 72
    if verbose:
        print(bar)
        print("Практическая работа №5, вариант 5")
        print(bar)

    t1 = data["task1"]
    print(f"\n{bar}\n[1/2] Задание 1 — таблица 5.16 (ремонт двигателя: да/нет)\n{bar}")
    print(f"Строки таблицы в обучении (№ по методичке): {t1['train_rows_1based']}")
    print(f"Строки таблицы в тесте:                      {t1['test_rows_1based']}")
    print("Метрики на обучении:", t1["metrics_train"])
    print("Метрики на тесте:   ", t1["metrics_test"])
    print("Матрица ошибок (test), строки=факт, столбцы=прогноз [нет, да]:")
    print(t1["confusion_test"])
    print("classification_report (test):\n" + t1["report_test"])

    t2 = data["task2"]
    print(f"\n{bar}\n[2/2] Задание 2 — таблица 5.21, тема «метеоданные» (осадки t→t+1)\n{bar}")
    print(
        f"Supervised-выборка: {t2['n_supervised']} точек; train/test по времени: {t2['n_train']} / {t2['n_test']} "
        f"(эффективная доля обучения ≈ {t2['train_ratio_effective']:.0%})"
    )
    print(f"Признаки на шаге t: {t2['feature_names']} → цель: {t2['target_name']}")
    print("Метрики на обучении:", t2["metrics_train"])
    print("Метрики на тесте:   ", t2["metrics_test"])
    print("Матрица ошибок (test), строки=факт, столбцы=прогноз:")
    print(t2["confusion_test"])
    print("classification_report (test):\n" + t2["report_test"])

    print("\n[файлы]")
    print(f"      {data['out_dir']}")
    for key, p in data["plots"].items():
        if p:
            print(f"      • {key}: {p}")


def main() -> None:
    data = run_practice05()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
