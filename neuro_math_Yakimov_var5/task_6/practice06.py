"""
Практическая работа №6.

Используется та же постановка и те же признаки/разбиение, что в практике №2, задание 2
(прогноз цены close[t+1] по признакам на день t). В `MLPRegressor` из scikit-learn
нет dropout, поэтому здесь реализована полносвязная сеть с теми же размерами скрытых
слоёв (32, 16), активацией tanh и **inverted dropout** после каждого скрытого слоя.

Обучение — мини-батч Adam + L2 (weight decay) на **NumPy**

Сравниваются вероятности dropout: 0.1, 0.2 и 0.5.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from neuro_math_Yakimov_var5.task_2.practice02 import build_synthetic_stock_series
except ModuleNotFoundError:
    # Запуск из каталога neuro_math_Yakimov_var5: python -m task_6.practice06
    from task_2.practice02 import build_synthetic_stock_series


def _build_stock_xy_split(
    *,
    n: int = 600,
    train_ratio: float = 0.82,
    seed: int = 42,
) -> dict:
    """Тот же конвейер признаков и хронологическое разбиение, что в `run_task2_stock_forecast`."""
    s = build_synthetic_stock_series(n=n, seed=seed)
    close = s["close"]
    volume = s["volume"]
    vol_roll = s["vol_roll"]
    log_ret = s["log_ret"]

    feats = []
    targets = []
    for t in range(3, n - 1):
        r1, r2, r3 = log_ret[t], log_ret[t - 1], log_ret[t - 2]
        feats.append([r1, r2, r3, volume[t], vol_roll[t], close[t]])
        targets.append(close[t + 1])

    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    n_eff = len(y)
    n_train = max(50, int(n_eff * train_ratio))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_train).astype(np.float64)
    X_te = sx.transform(X_test).astype(np.float64)

    sy = StandardScaler()
    y_tr_scaled = sy.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float64)

    split_close_idx = 3 + n_train

    return {
        "close": close,
        "n_train": n_train,
        "n_eff": n_eff,
        "split_close_idx": split_close_idx,
        "X_train": X_tr,
        "X_test": X_te,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "y_train_scaled": y_tr_scaled,
        "scaler_y": sy,
    }


def _xavier_std(fan_in: int, fan_out: int) -> float:
    return float(np.sqrt(2.0 / (fan_in + fan_out)))


def _init_params(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "W1": rng.normal(0.0, _xavier_std(6, 32), (6, 32)),
        "b1": np.zeros(32, dtype=np.float64),
        "W2": rng.normal(0.0, _xavier_std(32, 16), (32, 16)),
        "b2": np.zeros(16, dtype=np.float64),
        "W3": rng.normal(0.0, _xavier_std(16, 1), (16, 1)),
        "b3": np.zeros(1, dtype=np.float64),
    }


def _forward_eval(X: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """Инференс без dropout."""
    z1 = X @ p["W1"] + p["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ p["W2"] + p["b2"]
    h2 = np.tanh(z2)
    return (h2 @ p["W3"] + p["b3"]).ravel()


def _forward_train_batch(
    Xb: np.ndarray,
    yb: np.ndarray,
    p: dict[str, np.ndarray],
    dropout_p: float,
    rng: np.random.Generator,
    weight_decay: float,
) -> tuple[float, dict]:
    """Один мини-батч: прямой проход + MSE + кэш для обратного прохода."""
    B = Xb.shape[0]
    keep_p = 1.0 - dropout_p
    if keep_p <= 0.0:
        raise ValueError("dropout_p должен быть < 1")

    z1 = Xb @ p["W1"] + p["b1"]
    h1 = np.tanh(z1)
    mask1 = (rng.random(h1.shape) < keep_p).astype(np.float64)
    h1d = h1 * mask1 / keep_p

    z2 = h1d @ p["W2"] + p["b2"]
    h2 = np.tanh(z2)
    mask2 = (rng.random(h2.shape) < keep_p).astype(np.float64)
    h2d = h2 * mask2 / keep_p

    pred = (h2d @ p["W3"] + p["b3"]).ravel()

    diff = pred - yb
    mse = float(np.mean(diff**2))
    reg = 0.5 * weight_decay * (
        np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2) + np.sum(p["W3"] ** 2)
    )
    loss = mse + reg

    cache = {
        "Xb": Xb,
        "z1": z1,
        "h1": h1,
        "mask1": mask1,
        "h1d": h1d,
        "z2": z2,
        "h2": h2,
        "mask2": mask2,
        "h2d": h2d,
        "pred": pred,
        "yb": yb,
        "B": B,
        "keep_p": keep_p,
    }
    return loss, cache


def _backward_batch(cache: dict, p: dict[str, np.ndarray], weight_decay: float) -> dict[str, np.ndarray]:
    """Градиенты усреднённого MSE + L2 по весам."""
    B = cache["B"]
    d_pred = (2.0 / B) * (cache["pred"] - cache["yb"])

    d_b3 = np.array([np.sum(d_pred)], dtype=np.float64)
    d_W3 = cache["h2d"].T @ d_pred.reshape(B, 1)

    d_h2d = d_pred.reshape(B, 1) * p["W3"].reshape(1, -1)
    d_h2 = d_h2d * cache["mask2"] / cache["keep_p"] * (1.0 - cache["h2"] ** 2)

    d_W2 = cache["h1d"].T @ d_h2
    d_b2 = np.sum(d_h2, axis=0)

    d_h1d = d_h2 @ p["W2"].T
    d_h1 = d_h1d * cache["mask1"] / cache["keep_p"] * (1.0 - cache["h1"] ** 2)

    d_W1 = cache["Xb"].T @ d_h1
    d_b1 = np.sum(d_h1, axis=0)

    grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2, "W3": d_W3, "b3": d_b3}

    if weight_decay > 0.0:
        grads["W1"] += weight_decay * p["W1"]
        grads["W2"] += weight_decay * p["W2"]
        grads["W3"] += weight_decay * p["W3"]

    return grads


def _adam_step(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    m: dict[str, np.ndarray],
    v: dict[str, np.ndarray],
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for k in params:
        g = grads[k]
        m[k] = beta1 * m[k] + (1.0 - beta1) * g
        v[k] = beta2 * v[k] + (1.0 - beta2) * (g * g)
        m_hat = m[k] / (1.0 - beta1**t)
        v_hat = v[k] / (1.0 - beta2**t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _train_one_dropout(
    *,
    split: dict,
    dropout_p: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    X_train = split["X_train"]
    y_train = split["y_train_scaled"]
    X_test = split["X_test"]
    y_test_raw = split["y_test_raw"]
    y_train_raw = split["y_train_raw"]
    sy: StandardScaler = split["scaler_y"]

    params = _init_params(rng)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}

    n_train = X_train.shape[0]
    global_step = 0

    for _ in range(epochs):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, batch_size):
            idx = perm[s : s + batch_size]
            Xb = X_train[idx]
            yb = y_train[idx]
            _, cache = _forward_train_batch(
                Xb, yb, params, dropout_p, rng, weight_decay
            )
            global_step += 1
            grads = _backward_batch(cache, params, weight_decay)
            _adam_step(params, grads, m, v, global_step, lr)

    y_pred_tr_s = _forward_eval(X_train, params)
    y_pred_te_s = _forward_eval(X_test, params)

    y_pred_train = sy.inverse_transform(y_pred_tr_s.reshape(-1, 1)).ravel()
    y_pred_test = sy.inverse_transform(y_pred_te_s.reshape(-1, 1)).ravel()

    def _met(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        return {
            "mse": float(mean_squared_error(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }

    return {
        "dropout_p": float(dropout_p),
        "metrics_train": _met(y_train_raw, y_pred_train),
        "metrics_test": _met(y_test_raw, y_pred_test),
        "y_test_raw": y_test_raw,
        "y_pred_test": y_pred_test,
    }


def run_dropout_experiment(
    dropout_rates: tuple[float, ...] = (0.1, 0.2, 0.5),
    *,
    epochs: int = 1200,
    batch_size: int = 64,
    lr: float = 4e-3,
    weight_decay: float = 5e-3,
    base_seed: int = 42,
) -> dict:
    split = _build_stock_xy_split()

    runs = []
    for i, p in enumerate(dropout_rates):
        seed_m = base_seed + 17 * i
        runs.append(
            _train_one_dropout(
                split=split,
                dropout_p=p,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed_m,
            )
        )

    return {
        "split": split,
        "runs": runs,
        "training_config": {
            "backend": "numpy",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    }


def _save_comparison_plot(path: Path, experiment: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_test = experiment["split"]["y_test_raw"]
    idx = np.arange(len(y_test))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(idx, y_test, "ko-", markersize=4, linewidth=1.0, label="Факт (тест)")
    colors = ["tab:red", "tab:blue", "tab:green"]
    for run, c in zip(experiment["runs"], colors, strict=True):
        p = run["dropout_p"]
        ax.plot(idx, run["y_pred_test"], "^-", markersize=3, linewidth=1.0, color=c, label=f"Dropout p={p:g}")

    ax.set_title("Практика №6: прогноз цены на тесте при разных dropout (та же постановка, что пр. 2 зад. 2)")
    ax.set_xlabel("Номер точки на тесте")
    ax.set_ylabel("Цена")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_metrics_bar(path: Path, experiment: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = experiment["runs"]
    ps = [r["dropout_p"] for r in runs]
    mse = [r["metrics_test"]["mse"] for r in runs]
    labels = [f"p={p:g}" for p in ps]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, mse, color=["steelblue", "coral", "seagreen"])
    ax.set_ylabel("MSE на тесте")
    ax.set_title("Сравнение качества (MSE) на отложенном тесте")
    for b, v in zip(bars, mse, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_practice06(
    out_dir: Path | str | None = None,
    save_png: bool = True,
    *,
    epochs: int = 1200,
) -> dict:
    """
    Полный прогон практики №6. Консольный вывод не выполняется — см. `print_results`.
    """
    out = Path(out_dir) if out_dir is not None else Path("outputs") / "practice06"
    out.mkdir(parents=True, exist_ok=True)

    experiment = run_dropout_experiment(epochs=epochs)

    p_compare = out / "practice06_dropout_test_forecasts.png"
    p_bar = out / "practice06_dropout_test_mse_bar.png"

    if save_png:
        _save_comparison_plot(p_compare, experiment)
        _save_metrics_bar(p_bar, experiment)

    return {
        "out_dir": str(out.resolve()),
        "experiment": experiment,
        "plots": {
            "test_forecasts": str(p_compare) if save_png else None,
            "test_mse_bar": str(p_bar) if save_png else None,
        },
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    bar = "=" * 72
    if verbose:
        print(bar)
        print("Практическая работа №6 — dropout в НС (постановка как практика 2, задание 2)")
        print(bar)

    exp = data["experiment"]["split"]
    print("\nДанные и разбиение (как в task_2.practice02.run_task2_stock_forecast):")
    print(f"  n_eff={exp['n_eff']}, n_train={exp['n_train']}, n_test={exp['n_eff'] - exp['n_train']}")
    print(f"  Граница по индексу close (для графиков): split_close_idx={exp['split_close_idx']}")

    cfg = data["experiment"]["training_config"]
    print(
        f"\nОбучение (NumPy): backend={cfg['backend']}, epochs={cfg['epochs']}, "
        f"batch_size={cfg['batch_size']}, lr={cfg['lr']}, weight_decay={cfg['weight_decay']}"
    )
    print("Архитектура: Linear(6→32) → tanh → Dropout(p) → Linear(32→16) → tanh → Dropout(p) → Linear(16→1)")

    print(f"\n{bar}\nМетрики на тесте для заданных вероятностей dropout\n{bar}")
    for run in data["experiment"]["runs"]:
        p = run["dropout_p"]
        mt = run["metrics_test"]
        print(f"  dropout p={p:g}: MSE={mt['mse']:.6g}, MAE={mt['mae']:.6g}, R²={mt['r2']:.6g}")

    print("\n[файлы]")
    print(f"      {data['out_dir']}")
    for key, pth in data["plots"].items():
        if pth:
            print(f"      • {key}: {pth}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Практика №6 — dropout (NumPy)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1200,
        help="Число эпох обучения на каждое значение dropout (по умолчанию 1200)",
    )
    args = parser.parse_args()
    data = run_practice06(epochs=args.epochs)
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
