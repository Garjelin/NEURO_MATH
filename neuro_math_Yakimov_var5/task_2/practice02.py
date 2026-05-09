"""
Практическая работа №2, вариант 5.

Задание 1 (табл. 2.7, 2.8): НС по тренировочной выборке — для варианта 5 строки 7, 8, 9, 10.
Задание 2 (табл. 2.9): прогнозирование цен акций — временной ряд и признаки в стиле OHLCV.

Примечание по заданию 1: в строках 9 и 10 одно и то же x = 2 при разных Y (3 и −2);
однозначной функции y = f(x) не существует — НС даёт регрессионное усреднение в смысле MSE.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- Задание 1: таблица 2.7 (полная), вариант 5 → номера строк 7, 8, 9, 10 -----------------
TABLE_27_FULL = np.array(
    [
        [-1.0, 1.0],  # 1
        [0.0, 0.0],  # 2
        [1.0, 1.0],  # 3
        [2.0, 4.0],  # 4
        [-1.0, 0.0],  # 5
        [1.0, 1.0],  # 6
        [0.0, 1.0],  # 7  — тренировка вар. 5
        [1.0, 0.0],  # 8
        [2.0, 3.0],  # 9
        [2.0, -2.0],  # 10
    ],
    dtype=float,
)

# Индексы строк (1-based как в задании) для варианта 5
VARIANT_5_TRAIN_ROWS_1BASE = (7, 8, 9, 10)
VARIANT_5_TEST_ROWS_1BASE = tuple(i for i in range(1, 11) if i not in VARIANT_5_TRAIN_ROWS_1BASE)


def _rows_by_indices(rows_1based: tuple[int, ...]) -> np.ndarray:
    """Выборка строк табл. 2.7 по номерам с 1."""
    idx0 = [i - 1 for i in rows_1based]
    return TABLE_27_FULL[idx0].copy()


def run_task1_logical_sample(
    random_state: int = 42,
) -> dict:
    """
    Обучение MLPRegressor на 4 точках варианта 5; тест — остальные строки табл. 2.7.

    Масштабирование признака x и цели y через StandardScaler — иначе на малых данных
    MLP плохо сходится.
    """
    train_xy = _rows_by_indices(VARIANT_5_TRAIN_ROWS_1BASE)
    test_xy = _rows_by_indices(VARIANT_5_TEST_ROWS_1BASE)

    X_train = train_xy[:, :1]
    y_train = train_xy[:, 1]
    X_test = test_xy[:, :1]
    y_test = test_xy[:, 1]

    sx = StandardScaler()
    sy = StandardScaler()
    X_tr = sx.fit_transform(X_train)
    y_tr = sy.fit_transform(y_train.reshape(-1, 1)).ravel()

    # На 4 точках Adam часто «не находит» минимум; L-BFGS хорош для крошечных табличных задач.
    mlp = MLPRegressor(
        hidden_layer_sizes=(12, 12),
        activation="tanh",
        solver="lbfgs",
        max_iter=5000,
        random_state=random_state,
        alpha=1e-4,
        tol=1e-7,
    )
    mlp.fit(X_tr, y_tr)

    def predict_raw(x_col: np.ndarray) -> np.ndarray:
        """Предсказание в исходном масштабе Y."""
        xs = sx.transform(x_col.reshape(-1, 1))
        return sy.inverse_transform(mlp.predict(xs).reshape(-1, 1)).ravel()

    y_pred_train = predict_raw(X_train.ravel())
    y_pred_test = predict_raw(X_test.ravel())

    metrics_train = {
        "mse": mean_squared_error(y_train, y_pred_train),
        "mae": mean_absolute_error(y_train, y_pred_train),
        "r2": r2_score(y_train, y_pred_train),
    }
    metrics_test = {
        "mse": mean_squared_error(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "r2": r2_score(y_test, y_pred_test),
    }

    return {
        "mlp": mlp,
        "scaler_x": sx,
        "scaler_y": sy,
        "predict_raw": predict_raw,
        "train_xy": train_xy,
        "test_xy": test_xy,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }


def save_task1_plot(path: Path, t1: dict) -> None:
    """График: точки обучения/теста и непрерывная кривая предсказания НС."""
    path.parent.mkdir(parents=True, exist_ok=True)
    predict_raw = t1["predict_raw"]
    train_xy = t1["train_xy"]
    test_xy = t1["test_xy"]

    x_min = min(train_xy[:, 0].min(), test_xy[:, 0].min()) - 0.5
    x_max = max(train_xy[:, 0].max(), test_xy[:, 0].max()) + 0.5
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = predict_raw(x_grid)

    plt.figure(figsize=(9, 5))
    plt.plot(x_grid, y_grid, "b-", linewidth=2, label="Аппроксимация НС")
    plt.scatter(train_xy[:, 0], train_xy[:, 1], s=120, c="green", marker="o", zorder=5, label="Обучение")
    plt.scatter(test_xy[:, 0], test_xy[:, 1], s=120, c="red", marker="s", zorder=5, label="Тест")
    plt.xlabel("x")
    plt.ylabel("Y")
    plt.title("Задание 1 (вар. 5): зависимость Y(x), обучение на строках 7–10 табл. 2.7")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- Задание 2: прогноз «цен акций» (воспроизводимый синтетический ряд OHLCV-стиля) ------


def build_synthetic_stock_series(n: int = 600, seed: int = 42) -> dict:
    """
    Демонстрационный ряд: цена закрытия (random walk в лог-масштабе) и объём.

    Для реальных котировок можно подставить данные yfinance / CSV — структура признаков та же.
    """
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0004, 0.018, n)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    volume = rng.lognormal(12.0, 0.35, n)

    log_ret = np.zeros(n, dtype=float)
    log_ret[1:] = np.diff(np.log(close))

    win = 5
    vol_roll = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - win + 1)
        vol_roll[i] = float(np.std(log_ret[lo : i + 1])) if i >= 1 else 0.0

    return {"close": close, "volume": volume, "vol_roll": vol_roll, "log_ret": log_ret}


def run_task2_stock_forecast(
    n: int = 600,
    train_ratio: float = 0.82,
    seed: int = 42,
    random_state: int = 42,
) -> dict:
    """
    Прогноз цены на один шаг вперёд: признаки на день t предсказывают close[t+1].

    Признаки — лаги лог-доходности, объём, скользящая вола, текущий уровень цены
    (без уровня модель плохо экстраполирует масштаб на тесте).
    Разбиение по времени (без перемешивания): первые train_ratio точек — обучение.
    """
    s = build_synthetic_stock_series(n=n, seed=seed)
    close = s["close"]
    volume = s["volume"]
    vol_roll = s["vol_roll"]
    log_ret = s["log_ret"]

    feats = []
    targets = []
    # На момент t используем только информацию ≤ t; предсказываем close[t+1]
    for t in range(3, n - 1):
        r1, r2, r3 = log_ret[t], log_ret[t - 1], log_ret[t - 2]
        feats.append([r1, r2, r3, volume[t], vol_roll[t], close[t]])
        targets.append(close[t + 1])

    X = np.asarray(feats, dtype=float)
    y = np.asarray(targets, dtype=float)
    n_eff = len(y)

    n_train = max(50, int(n_eff * train_ratio))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_train)
    X_te = sx.transform(X_test)

    # Цель — цена в «сыром» масштабе (~80–150); для L-BFGS удобнее учить сеть на стандартизованном y.
    # Иначе sklearn часто выводит ConvergenceWarning: не хватило итераций до внутреннего tol.
    sy = StandardScaler()
    y_tr_scaled = sy.fit_transform(y_train.reshape(-1, 1)).ravel()

    # L-BFGS — итерационный решатель; предупреждение означает только «упёрлись в max_iter»,
    # а не ошибку программы. Масштаб y + запас по итерациям обычно убирает предупреждение.
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        solver="lbfgs",
        max_iter=8000,
        random_state=random_state,
        alpha=5e-3,
        tol=1e-5,
    )
    mlp.fit(X_tr, y_tr_scaled)

    y_pred_train = sy.inverse_transform(mlp.predict(X_tr).reshape(-1, 1)).ravel()
    y_pred_test = sy.inverse_transform(mlp.predict(X_te).reshape(-1, 1)).ravel()

    metrics_train = {
        "mse": mean_squared_error(y_train, y_pred_train),
        "mae": mean_absolute_error(y_train, y_pred_train),
        "r2": r2_score(y_train, y_pred_train),
    }
    metrics_test = {
        "mse": mean_squared_error(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "r2": r2_score(y_test, y_pred_test),
    }

    # Индекс на шкале «дней» close: последний обучающий таргет close[3 + n_train].
    split_close_idx = 3 + n_train

    return {
        "mlp": mlp,
        "scaler_x": sx,
        "scaler_y": sy,
        "close": close,
        "n_train_samples": n_train,
        "n_eff": n_eff,
        "split_close_idx": split_close_idx,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "train_ratio": train_ratio,
    }


def save_task2_plots(path: Path, t2: dict) -> None:
    """Два блока: весь ряд close с границей train/test; фрагмент теста — факт vs прогноз."""
    path.parent.mkdir(parents=True, exist_ok=True)
    close = t2["close"]
    split_idx = int(t2["split_close_idx"])
    y_test = t2["y_test"]
    y_pred_test = t2["y_pred_test"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax1 = axes[0]
    ax1.plot(close, color="steelblue", linewidth=1.0, label="Цена закрытия (синт.)")
    ax1.axvline(split_idx, color="gray", linestyle="--", label="Граница train / test (по задаче)")
    ax1.set_title("Задание 2 (вар. 5): ряд цен и разбиение выборки")
    ax1.set_xlabel("День")
    ax1.set_ylabel("Цена")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    idx = np.arange(len(y_test))
    ax2.plot(idx, y_test, "ko-", markersize=3, label="Факт (тест)")
    ax2.plot(idx, y_pred_test, "r^-", markersize=3, label="Прогноз НС")
    ax2.set_title("Тестовый интервал: фактическая цена vs предсказание")
    ax2.set_xlabel("Номер точки в тесте")
    ax2.set_ylabel("Цена")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_practice02(out_dir: Path | None = None, save_png: bool = True) -> dict:
    """Полный прогон практики №2: оба задания, метрики, при необходимости PNG."""
    if out_dir is None:
        out_dir = Path("outputs") / "practice02"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task1 = run_task1_logical_sample()
    task2 = run_task2_stock_forecast()

    paths: dict[str, Path] = {}
    if save_png:
        p1 = out_dir / "practice02_task1_fit.png"
        p2 = out_dir / "practice02_task2_forecast.png"
        save_task1_plot(p1, task1)
        save_task2_plots(p2, task2)
        paths["task1_plot"] = p1.resolve()
        paths["task2_plot"] = p2.resolve()

    return {
        "out_dir": out_dir.resolve(),
        "task1": task1,
        "task2": task2,
        "paths": paths,
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    sep = "=" * 72
    if verbose:
        print(sep)
        print("Практическая работа №2, вариант 5")
        print(sep)

    t1 = data["task1"]
    print("\n[1/3] Задание 1 — НС по табл. 2.7 (обучение: строки 7, 8, 9, 10)")
    print(f"      Тренировочные точки (x, Y):\n{t1['train_xy']}")
    print(f"      Тестовые точки — остальные строки таблицы:\n{t1['test_xy']}")
    print("      Метрики на обучении: ", t1["metrics_train"])
    print("      Метрики на тесте:    ", t1["metrics_test"])
    print(
        "      Примечание: при x=2 в обучении два разных Y — НС аппроксимирует в смысле MSE."
    )

    t2 = data["task2"]
    print("\n[2/3] Задание 2 — прогноз цен акций (синтетический OHLCV-подобный ряд)")
    frac = t2["n_train_samples"] / max(1, t2["n_eff"])
    print(
        f"      Длина ряда close: {len(t2['close'])}, образцов признаков: {t2['n_eff']}, "
        f"train/test по точкам ≈ {frac:.0%} / {1 - frac:.0%}"
    )
    print("      Метрики на обучении: ", t2["metrics_train"])
    print("      Метрики на тесте:    ", t2["metrics_test"])

    print("\n[3/3] Файлы результатов:")
    print(f"      {data['out_dir']}")
    for key, p in data["paths"].items():
        print(f"      • {key}: {p}")


def main() -> None:
    data = run_practice02()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
