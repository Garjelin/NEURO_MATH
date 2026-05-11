"""
Практическая работа №7, вариант 5.

Задание 1 — таблица 7.7: регрессия величины Y по бинарным изображениям (в методичке
полная таблица приведена в основном пособии; в выданном PDF фрагмент только ссылка).
Здесь: обучение MLPRegressor на фиксированном наборе образцов 3×3 и прогноз Y для
объекта «вариант 5» (5-й образец), оценка по удержанному эталону.

Задание 2 — таблица 7.8 (вариант 5): изображение 4×4, фильтр 3×3, stride=1, padding=0,
деконволюция (транспонированная свёртка) с выходом 6×6.

Задание 3 — рис. 7.42, таблица 7.9: прямой проход embedding-слоя (матрица E) и
агрегирование эмбеддингов токенов для объектов из учебной таблицы (минимальный пример;
при расхождении с методичкой замените константы в коде).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- Задание 2: таблица 7.8, вариант 5 (из PDF «Задание_7_Вар_5», -layout) -----------------
IMAGE_78_VAR5 = np.array(
    [
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
    ],
    dtype=np.float64,
)

FILTER_78_VAR5 = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


def conv_transpose2d_stride1_pad0(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Транспонированная свёртка (как nn.ConvTranspose2d в PyTorch при stride=1, padding=0,
    output_padding=0, dilation=1, groups=1): каждый элемент входа «размазывает» ядро.
    Выход: (Hi + Hk - 1) × (Wi + Wk - 1).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    Ho, Wo = Hi + Hk - 1, Wi + Wk - 1
    out = np.zeros((Ho, Wo), dtype=np.float64)
    for i in range(Hi):
        for j in range(Wi):
            v = image[i, j]
            if v != 0.0:
                out[i : i + Hk, j : j + Wk] += v * kernel
    return out


def run_task2_deconvolution() -> dict:
    out = conv_transpose2d_stride1_pad0(IMAGE_78_VAR5, FILTER_78_VAR5)
    assert out.shape == (6, 6), out.shape
    return {
        "image": IMAGE_78_VAR5,
        "kernel": FILTER_78_VAR5,
        "output_6x6": out,
    }


# --- Задание 1: учебный набор 3×3 (замените при необходимости по полной табл. 7.7) -------
def _flatten_img(img: np.ndarray) -> np.ndarray:
    return img.reshape(-1).astype(np.float64)


def run_task1_image_regression(random_state: int = 42) -> dict:
    """
    6 обучающих бинарных изображений 3×3 и известные Y; 5-й объект (индекс 4) —
    «вариант 5»: обучение без него и прогноз Y, сравнение с зашитым эталоном.
    Эталон Y для демонстрации: Y = 0.12 * sum(pixels) + 0.05 * (верхний левый пиксель).
    """
    rng = np.random.default_rng(7)
    imgs: list[np.ndarray] = []
    for _ in range(6):
        m = (rng.random((3, 3)) > 0.45).astype(np.float64)
        imgs.append(m)

    imgs[4] = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float64)

    y_true = np.array([0.12 * float(m.sum()) + 0.05 * m[0, 0] for m in imgs], dtype=np.float64)

    hold_idx = 4
    train_idx = [i for i in range(6) if i != hold_idx]

    X_train = np.stack([_flatten_img(imgs[i]) for i in train_idx])
    y_train = y_true[train_idx]
    X_hold = _flatten_img(imgs[hold_idx]).reshape(1, -1)
    y_hold_true = float(y_true[hold_idx])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_hold)

    mlp = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=8000,
        random_state=random_state,
        alpha=1e-4,
        tol=1e-6,
    )
    mlp.fit(X_tr, y_train)
    y_pred_hold = float(mlp.predict(X_ho)[0])
    y_pred_train = mlp.predict(X_tr)

    return {
        "description": "Учебный набор 3×3; при наличии полной табл. 7.7 замените imgs/y_true в коде.",
        "hold_index_1based": hold_idx + 1,
        "y_hold_true": y_hold_true,
        "y_hold_pred": y_pred_hold,
        "abs_err_hold": abs(y_pred_hold - y_hold_true),
        "metrics_train": {
            "mse": float(mean_squared_error(y_train, y_pred_train)),
            "mae": float(mean_absolute_error(y_train, y_pred_train)),
            "r2": float(r2_score(y_train, y_pred_train)),
        },
        "images": imgs,
        "y_all": y_true,
    }


# --- Задание 3: embedding (рис. 7.42) + учебная таблица 7.9 (минимальный пример) ----------
def run_task3_embedding_demo() -> dict:
    """
    Словарь из V слов, каждому — вектор размерности d (матрица E размера V×d).
    Для объекта: берём последовательность индексов токенов, суммируем их эмбеддинги,
    скаляр Y = w^T sum_emb + b (один выход). Прямой проход + подбор w,b МНК по таблице.
    """
    vocab_size = 5
    embed_dim = 3
    rng = np.random.default_rng(2027)
    E = rng.normal(0.0, 0.4, (vocab_size, embed_dim))

    # Объекты: список токенов (индексы) и целевое Y (замените по табл. 7.9 из методички)
    sequences = [
        [0, 1, 2],
        [2, 3],
        [1, 4, 0, 2],
        [3, 1],
    ]
    y_targets = np.array([0.5, -0.2, 0.8, 0.1], dtype=np.float64)

    sums = []
    for seq in sequences:
        s = np.zeros(embed_dim, dtype=np.float64)
        for t in seq:
            s += E[t]
        sums.append(s)
    S = np.stack(sums)
    A = np.column_stack([S, np.ones(len(sequences))])
    w_b, *_ = np.linalg.lstsq(A, y_targets, rcond=None)
    w = w_b[:-1]
    b = float(w_b[-1])
    y_hat = S @ w + b

    return {
        "description": "Минимальный пример; замените sequences/y_targets по табл. 7.9.",
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "E": E,
        "sequences": sequences,
        "y_targets": y_targets,
        "y_pred": y_hat,
        "w": w,
        "b": b,
        "mse": float(mean_squared_error(y_targets, y_hat)),
    }


def _heatmap(path: Path, mat: np.ndarray, title: str, cbar_label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", color="w", fontsize=9)
    if cbar_label:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    else:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_practice07(out_dir: Path | str | None = None, save_png: bool = True) -> dict:
    out = Path(out_dir) if out_dir is not None else Path("outputs") / "practice07"
    out.mkdir(parents=True, exist_ok=True)

    t1 = run_task1_image_regression()
    t2 = run_task2_deconvolution()
    t3 = run_task3_embedding_demo()

    p_img = out / "practice07_task2_image.png"
    p_ker = out / "practice07_task2_kernel.png"
    p_out = out / "practice07_task2_deconv_6x6.png"
    p_emb = out / "practice07_task3_embedding_matrix.png"

    if save_png:
        _heatmap(p_img, t2["image"], "Задание 2: изображение 4×4 (табл. 7.8, вар. 5)")
        _heatmap(p_ker, t2["kernel"], "Задание 2: фильтр 3×3")
        _heatmap(p_out, t2["output_6x6"], "Задание 2: результат деконволюции 6×6")
        _heatmap(p_emb, t3["E"], "Задание 3: матрица embedding E (V×d)")

    return {
        "out_dir": str(out.resolve()),
        "task1": t1,
        "task2": t2,
        "task3": t3,
        "plots": {
            "task2_image": str(p_img) if save_png else None,
            "task2_kernel": str(p_ker) if save_png else None,
            "task2_output": str(p_out) if save_png else None,
            "task3_E": str(p_emb) if save_png else None,
        },
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    bar = "=" * 72
    if verbose:
        print(bar)
        print("Практическая работа №7, вариант 5")
        print(bar)

    t1 = data["task1"]
    print(f"\n{bar}\n[1/3] Задание 1 — регрессия Y по изображениям (табл. 7.7, учебный набор)\n{bar}")
    print(t1["description"])
    print(f"Удержан для прогноза объект №{t1['hold_index_1based']} (вариант 5 в смысле номера образца).")
    print(f"Эталонное Y: {t1['y_hold_true']:.6g}, прогноз: {t1['y_hold_pred']:.6g}, |ошибка|: {t1['abs_err_hold']:.6g}")
    print("Метрики на обучении (без удержанного):", t1["metrics_train"])

    t2 = data["task2"]
    print(f"\n{bar}\n[2/3] Задание 2 — деконволюция 6×6 (табл. 7.8)\n{bar}")
    print("Изображение 4×4:")
    print(t2["image"])
    print("Фильтр 3×3:")
    print(t2["kernel"])
    print("Выход 6×6:")
    print(np.round(t2["output_6x6"], 6))

    t3 = data["task3"]
    print(f"\n{bar}\n[3/3] Задание 3 — embedding (рис. 7.42) и табл. 7.9 (пример)\n{bar}")
    print(t3["description"])
    print("Последовательности токенов:", t3["sequences"])
    print("Y из таблицы:", t3["y_targets"])
    print("Y после подбора w,b (МНК):", np.round(t3["y_pred"], 6))
    print("MSE подгонки:", t3["mse"])

    print("\n[файлы]")
    print(f"      {data['out_dir']}")
    for k, p in data["plots"].items():
        if p:
            print(f"      • {k}: {p}")


def main() -> None:
    data = run_practice07()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
