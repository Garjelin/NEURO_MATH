"""
Практическая работа №3, вариант 5.

Тема из таблицы 3.11: «Прогноз стоимости акций».
Требование: выполнить нормализацию данных ДВУМЯ способами и сравнить результат.

Рабочий скрипт:
- формирует тренировочную/тестовую выборку (синтетический ряд цен);
- обучает две одинаковые НС (различается только метод нормализации);
- печатает подробные метрики в консоль;
- сохраняет графики.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def build_synthetic_stock_series(n: int = 720, seed: int = 42) -> dict:
    """Генерирует воспроизводимый «биржевой» ряд цен/объёма и вспомогательные признаки."""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.00035, 0.017, n)
    close = 120.0 * np.exp(np.cumsum(daily_ret))
    volume = rng.lognormal(mean=12.0, sigma=0.35, size=n)

    log_ret = np.zeros(n, dtype=float)
    log_ret[1:] = np.diff(np.log(close))

    # Скользящая волатильность доходностей (std по окну)
    win = 7
    vol_roll = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - win + 1)
        vol_roll[i] = float(np.std(log_ret[lo : i + 1])) if i >= 1 else 0.0

    return {"close": close, "volume": volume, "log_ret": log_ret, "vol_roll": vol_roll}


def build_features_targets(series: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Формирует supervised-датасет:
    признаки на момент t -> цель close[t+1].
    """
    close = series["close"]
    volume = series["volume"]
    log_ret = series["log_ret"]
    vol_roll = series["vol_roll"]
    n = len(close)

    X, y = [], []
    for t in range(3, n - 1):
        X.append([log_ret[t], log_ret[t - 1], log_ret[t - 2], volume[t], vol_roll[t], close[t]])
        y.append(close[t + 1])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def split_time_series(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> dict:
    """Разделение по времени без перемешивания."""
    n_train = max(60, int(len(y) * train_ratio))
    return {
        "X_train": X[:n_train],
        "X_test": X[n_train:],
        "y_train": y[:n_train],
        "y_test": y[n_train:],
        "n_train": n_train,
        "n_total": len(y),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_with_scalers(
    split: dict,
    x_scaler,
    y_scaler,
    random_state: int = 42,
) -> dict:
    """
    Обучает одну и ту же архитектуру MLPRegressor при заданных скейлерах.
    Это позволяет честно сравнить только влияние нормализации.
    """
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    X_tr = x_scaler.fit_transform(X_train)
    X_te = x_scaler.transform(X_test)
    y_tr = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        activation="tanh",
        solver="lbfgs",
        max_iter=7000,
        alpha=5e-3,
        tol=1e-5,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)

    y_pred_train = y_scaler.inverse_transform(model.predict(X_tr).reshape(-1, 1)).ravel()
    y_pred_test = y_scaler.inverse_transform(model.predict(X_te).reshape(-1, 1)).ravel()

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "metrics_train": _metrics(y_train, y_pred_train),
        "metrics_test": _metrics(y_test, y_pred_test),
    }


def save_comparison_plot(path: Path, split: dict, z_res: dict, mm_res: dict) -> None:
    """Сравнение прогноза на тесте: факт vs Z-score vs Min-Max."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y_test = split["y_test"]
    idx = np.arange(len(y_test))

    plt.figure(figsize=(10, 5))
    plt.plot(idx, y_test, "k-", linewidth=2, label="Факт (тест)")
    plt.plot(idx, z_res["y_pred_test"], "b--", linewidth=1.5, label="Прогноз (Z-score)")
    plt.plot(idx, mm_res["y_pred_test"], "r-.", linewidth=1.5, label="Прогноз (Min-Max)")
    plt.title("Задание 3 (вар. 5): сравнение нормализаций на тестовом интервале")
    plt.xlabel("Номер точки теста")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_scaling_demo_plot(path: Path, split: dict) -> None:
    """Показывает эффект двух нормализаций на примере признака 'close[t]'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x_close = split["X_train"][:, -1].reshape(-1, 1)
    z = StandardScaler().fit_transform(x_close).ravel()
    mm = MinMaxScaler().fit_transform(x_close).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(x_close.ravel(), bins=25, color="gray")
    axes[0].set_title("Исходный close[t]")
    axes[1].hist(z, bins=25, color="steelblue")
    axes[1].set_title("После Z-score")
    axes[2].hist(mm, bins=25, color="tomato")
    axes[2].set_title("После Min-Max")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_practice03(out_dir: Path | None = None, save_png: bool = True) -> dict:
    """Полный прогон: одна задача прогноза + два способа нормализации."""
    if out_dir is None:
        out_dir = Path("outputs") / "practice03"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = build_synthetic_stock_series()
    X, y = build_features_targets(series)
    split = split_time_series(X, y, train_ratio=0.8)

    # Способ 1: Z-score (стандартизация)
    z_res = train_with_scalers(split, StandardScaler(), StandardScaler())
    # Способ 2: Min-Max [0,1]
    mm_res = train_with_scalers(split, MinMaxScaler(), MinMaxScaler())

    paths: dict[str, Path] = {}
    if save_png:
        p1 = out_dir / "practice03_scaling_demo.png"
        p2 = out_dir / "practice03_test_comparison.png"
        save_scaling_demo_plot(p1, split)
        save_comparison_plot(p2, split, z_res, mm_res)
        paths["scaling_demo"] = p1.resolve()
        paths["test_comparison"] = p2.resolve()

    return {
        "out_dir": out_dir.resolve(),
        "split": split,
        "zscore": z_res,
        "minmax": mm_res,
        "paths": paths,
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    sep = "=" * 72
    split = data["split"]
    z = data["zscore"]
    mm = data["minmax"]

    if verbose:
        print(sep)
        print("Практическая работа №3, вариант 5")
        print("Тема: прогноз стоимости акций; сравнение 2 способов нормализации")
        print(sep)

    print("\n[1/4] Подготовка выборки")
    print(
        f"      Всего supervised-точек: {split['n_total']}, "
        f"train/test: {split['n_train']} / {split['n_total'] - split['n_train']}"
    )
    print(f"      Размер X_train: {split['X_train'].shape}, X_test: {split['X_test'].shape}")

    print("\n[2/4] Нормализация №1: Z-score (StandardScaler)")
    print("      Метрики на обучении:", z["metrics_train"])
    print("      Метрики на тесте:   ", z["metrics_test"])

    print("\n[3/4] Нормализация №2: Min-Max (MinMaxScaler)")
    print("      Метрики на обучении:", mm["metrics_train"])
    print("      Метрики на тесте:   ", mm["metrics_test"])

    print("\n[4/4] Сравнение на тесте (ниже — чем меньше MSE/MAE, тем лучше; R² — выше лучше)")
    dz = z["metrics_test"]
    dm = mm["metrics_test"]
    print(
        f"      Z-score: MSE={dz['mse']:.6f}, MAE={dz['mae']:.6f}, R2={dz['r2']:.6f}\n"
        f"      Min-Max: MSE={dm['mse']:.6f}, MAE={dm['mae']:.6f}, R2={dm['r2']:.6f}"
    )

    print("\n      Файлы результатов:")
    print(f"      {data['out_dir']}")
    for key, p in data["paths"].items():
        print(f"      • {key}: {p}")


def main() -> None:
    data = run_practice03()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
