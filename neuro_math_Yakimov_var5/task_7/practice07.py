"""
Практическая работа №7, вариант 5.

Задание 1 — таблица 7.7, вариант 5: три бинарных образа P1, P2, P3 размера 2×3 и значения Y.
Задание 2 — таблица 7.8, вариант 5: изображение 4×4, фильтр 3×3, stride=1, padding=0, деконволюция 6×6.

Задание 3 — таблица 7.9, вариант 5: для всех допустимых
троек (Anchor, Positive, Negative) выписать triplet loss, суммировать — итоговая функция
потерь (минимизируется при обучении embedding).
сеть моделируется как четыре обучаемых вектора вложения объектов A1, A2, B1, B2 ∈ R^d.
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

# --- Задание 1: таблица 7.7, вариант 5 (каждый P — матрица 2×3, построчно из методички) ---
# P1: Y=0   010 / 000
# P2: Y=2   110 / 011
# P3: Y=1   101 / 010
TASK1_PATTERNS_VAR5 = [
    (np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float64), 0.0, "P1"),
    (np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64), 2.0, "P2"),
    (np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float64), 1.0, "P3"),
]

# --- Задание 2: таблица 7.8, вариант 5 ---------------------------------------------------
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


def _flatten_img(img: np.ndarray) -> np.ndarray:
    return img.reshape(-1).astype(np.float64)


def run_task1_table77_variant5(random_state: int = 42) -> dict:
    """
    Три объекта из табл. 7.7 (вар. 5). Регрессия Y по развёртке 2×3 → вектор длины 6.

    Демонстрация: leave-one-out — по двум образцам обучается MLPRegressor, прогноз на третьем.
    """
    imgs = [t[0] for t in TASK1_PATTERNS_VAR5]
    names = [t[2] for t in TASK1_PATTERNS_VAR5]
    y_all = np.array([t[1] for t in TASK1_PATTERNS_VAR5], dtype=np.float64)

    loocv_preds: list[float] = []
    loocv_true: list[float] = []
    for hold in range(3):
        train = [i for i in range(3) if i != hold]
        X_train = np.stack([_flatten_img(imgs[i]) for i in train])
        y_train = y_all[train]
        X_hold = _flatten_img(imgs[hold]).reshape(1, -1)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_ho = scaler.transform(X_hold)

        mlp = MLPRegressor(
            hidden_layer_sizes=(12, 6),
            activation="relu",
            solver="adam",
            max_iter=12000,
            random_state=random_state + hold,
            alpha=1e-4,
            tol=1e-6,
        )
        mlp.fit(X_tr, y_train)
        loocv_preds.append(float(mlp.predict(X_ho)[0]))
        loocv_true.append(float(y_all[hold]))

    y_pred_all = np.array(loocv_preds)
    y_true_all = np.array(loocv_true)

    X_full = np.stack([_flatten_img(im) for im in imgs])
    scaler_f = StandardScaler()
    mlp_f = MLPRegressor(
        hidden_layer_sizes=(12, 6),
        activation="relu",
        solver="adam",
        max_iter=12000,
        random_state=random_state,
        alpha=1e-4,
        tol=1e-6,
    )
    mlp_f.fit(scaler_f.fit_transform(X_full), y_all)
    y_fit_train = mlp_f.predict(scaler_f.transform(X_full))

    return {
        "source": "Таблица 7.7, вариант 5",
        "pattern_names": names,
        "images_2x3": imgs,
        "y_table": y_all,
        "loocv_pred": y_pred_all,
        "loocv_true": y_true_all,
        "loocv_mae": float(mean_absolute_error(y_true_all, y_pred_all)),
        "loocv_mse": float(mean_squared_error(y_true_all, y_pred_all)),
        "fit_all_mae": float(mean_absolute_error(y_all, y_fit_train)),
        "fit_all_mse": float(mean_squared_error(y_all, y_fit_train)),
        "fit_all_r2": float(r2_score(y_all, y_fit_train)),
    }


# --- Задание 3: таблица 7.9 вар. 5 + суммарный triplet loss --------------------------------
OBJECT_NAMES = ["A1", "A2", "B1", "B2"]
# Таблица 7.9, вариант 5
TABLE_79_VAR5_X = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64)
TABLE_79_VAR5_Y = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)


def _triplets_anchor_positive_negative() -> list[tuple[int, int, int]]:
    """
    Класс A: A1, A2 (индексы 0,1); класс B: B1, B2 (2,3).
    Тройки (anchor, positive, negative): A и P в одном классе, N в другом, P ≠ A.
    """
    A = (0, 1)
    B = (2, 3)
    triplets: list[tuple[int, int, int]] = []
    for a in A:
        for p in A:
            if p == a:
                continue
            for n in B:
                triplets.append((a, p, n))
    for a in B:
        for p in B:
            if p == a:
                continue
            for n in A:
                triplets.append((a, p, n))
    return triplets


def _triplet_loss_term(
    E: np.ndarray,
    a: int,
    p: int,
    n: int,
    margin: float,
) -> tuple[float, bool]:
    """L = max(0, ||e_a-e_p||^2 - ||e_a-e_n||^2 + margin). Возвращает (значение, активен ли шарнир)."""
    d_ap = float(np.sum((E[a] - E[p]) ** 2))
    d_an = float(np.sum((E[a] - E[n]) ** 2))
    raw = d_ap - d_an + margin
    active = raw > 0.0
    return (float(max(0.0, raw)), active)


def _loss_grad(E: np.ndarray, triplets: list[tuple[int, int, int]], margin: float) -> tuple[float, np.ndarray]:
    total = 0.0
    G = np.zeros_like(E)
    for a, p, n in triplets:
        d_ap = np.sum((E[a] - E[p]) ** 2)
        d_an = np.sum((E[a] - E[n]) ** 2)
        raw = d_ap - d_an + margin
        if raw <= 0.0:
            continue
        total += raw
        G[a] += 2.0 * (E[n] - E[p])
        G[p] += -2.0 * (E[a] - E[p])
        G[n] += 2.0 * (E[a] - E[n])
    return total, G


def run_task3_triplet_embedding(
    *,
    embed_dim: int = 2,
    margin: float = 0.25,
    steps: int = 4000,
    lr: float = 0.08,
    seed: int = 11,
) -> dict:
    """
    Четыре вектора в R^{embed_dim} — вложения объектов A1..B2.
    Суммарный triplet loss по всем тройкам из _triplets_anchor_positive_negative().
    Градиентный спуск по сумме (как численная иллюстрация минимизации из методички).
    """
    rng = np.random.default_rng(seed)
    E = 0.15 * rng.standard_normal((4, embed_dim))
    triplets = _triplets_anchor_positive_negative()

    per_triplet_initial = []
    for a, p, n in triplets:
        lt, act = _triplet_loss_term(E, a, p, n, margin)
        per_triplet_initial.append(
            {
                "anchor": OBJECT_NAMES[a],
                "positive": OBJECT_NAMES[p],
                "negative": OBJECT_NAMES[n],
                "loss": lt,
                "active": act,
            }
        )
    L0, _ = _loss_grad(E, triplets, margin)

    for _ in range(steps):
        L, G = _loss_grad(E, triplets, margin)
        if L <= 1e-12:
            break
        E -= lr * G

    per_triplet_final = []
    for a, p, n in triplets:
        lt, act = _triplet_loss_term(E, a, p, n, margin)
        per_triplet_final.append(
            {
                "anchor": OBJECT_NAMES[a],
                "positive": OBJECT_NAMES[p],
                "negative": OBJECT_NAMES[n],
                "loss": lt,
                "active": act,
            }
        )
    L1, _ = _loss_grad(E, triplets, margin)

    return {
        "source_table": "Таблица 7.9, вариант 5",
        "object_names": OBJECT_NAMES,
        "table_X": TABLE_79_VAR5_X.copy(),
        "table_Y": TABLE_79_VAR5_Y.copy(),
        "embed_dim": embed_dim,
        "margin": margin,
        "triplets_count": len(triplets),
        "triplets": [(OBJECT_NAMES[a], OBJECT_NAMES[p], OBJECT_NAMES[n]) for a, p, n in triplets],
        "loss_sum_initial": L0,
        "loss_sum_final": L1,
        "per_triplet_initial": per_triplet_initial,
        "per_triplet_final": per_triplet_final,
        "E_final": E.copy(),
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


def _save_task1_montage(path: Path, task1: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = task1["images_2x3"]
    names = task1["pattern_names"]
    yv = task1["y_table"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, im, nm, y in zip(axes, imgs, names, yv, strict=True):
        ax.imshow(im, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{nm}, Y={y:g}")
        ax.set_xticks(range(3))
        ax.set_yticks(range(2))
    fig.suptitle("Задание 1: табл. 7.7, вар. 5 — образы 2×3")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_task3_scatter(path: Path, task3: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    E = task3["E_final"]
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange"]
    for i, name in enumerate(OBJECT_NAMES):
        ax.scatter(E[i, 0], E[i, 1], s=120, c=colors[i], edgecolors="black", zorder=5)
        ax.annotate(name, (E[i, 0], E[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=11)
    ax.set_title("Задание 3: эмбеддинги объектов (после минимизации triplet loss)")
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_practice07(out_dir: Path | str | None = None, save_png: bool = True) -> dict:
    out = Path(out_dir) if out_dir is not None else Path("outputs") / "practice07"
    out.mkdir(parents=True, exist_ok=True)

    t1 = run_task1_table77_variant5()
    t2 = run_task2_deconvolution()
    t3 = run_task3_triplet_embedding()

    p_t1 = out / "practice07_task1_patterns.png"
    p_img = out / "practice07_task2_image.png"
    p_ker = out / "practice07_task2_kernel.png"
    p_out = out / "practice07_task2_deconv_6x6.png"
    p_emb = out / "practice07_task3_embeddings_heatmap.png"
    p_sc = out / "practice07_task3_embeddings_scatter.png"

    if save_png:
        _save_task1_montage(p_t1, t1)
        _heatmap(p_img, t2["image"], "Задание 2: изображение 4×4 (табл. 7.8, вар. 5)")
        _heatmap(p_ker, t2["kernel"], "Задание 2: фильтр 3×3")
        _heatmap(p_out, t2["output_6x6"], "Задание 2: результат деконволюции 6×6")
        _heatmap(p_emb, t3["E_final"], "Задание 3: матрица эмбеддингов 4×d (после обучения)")
        _save_task3_scatter(p_sc, t3)

    return {
        "out_dir": str(out.resolve()),
        "task1": t1,
        "task2": t2,
        "task3": t3,
        "plots": {
            "task1_patterns": str(p_t1) if save_png else None,
            "task2_image": str(p_img) if save_png else None,
            "task2_kernel": str(p_ker) if save_png else None,
            "task2_output": str(p_out) if save_png else None,
            "task3_E": str(p_emb) if save_png else None,
            "task3_scatter": str(p_sc) if save_png else None,
        },
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    bar = "=" * 72
    if verbose:
        print(bar)
        print("Практическая работа №7, вариант 5")
        print(bar)

    t1 = data["task1"]
    print(f"\n{bar}\n[1/3] Задание 1 — таблица 7.7, вариант 5\n{bar}")
    print(t1["source"])
    for nm, im, y in zip(t1["pattern_names"], t1["images_2x3"], t1["y_table"], strict=True):
        print(f"  {nm}, Y={y:g}:\n{im}")
    print("Leave-one-out (обучение на двух образцах, прогноз на третьем):")
    print(f"  y_true: {t1['loocv_true']}")
    print(f"  y_pred: {t1['loocv_pred']}")
    print(f"  LOOCV MSE={t1['loocv_mse']:.6g}, MAE={t1['loocv_mae']:.6g}")
    print(f"Подгонка на всех трёх сразу: MSE={t1['fit_all_mse']:.6g}, MAE={t1['fit_all_mae']:.6g}, R²={t1['fit_all_r2']:.6g}")

    t2 = data["task2"]
    print(f"\n{bar}\n[2/3] Задание 2 — деконволюция 6×6 (табл. 7.8, вар. 5)\n{bar}")
    print("Изображение 4×4:")
    print(t2["image"])
    print("Фильтр 3×3:")
    print(t2["kernel"])
    print("Выход 6×6:")
    print(np.round(t2["output_6x6"], 6))

    t3 = data["task3"]
    print(f"\n{bar}\n[3/3] Задание 3 — табл. 7.9 + triplet loss\n{bar}")
    print(t3["source_table"])
    print("Объекты и столбцы X, Y (табл. 7.9, вар. 5):")
    for i, name in enumerate(t3["object_names"]):
        print(f"  {name}: X={t3['table_X'][i]:g}, Y={t3['table_Y'][i]:g}")
    print(f"Число троек (A,P,N): {t3['triplets_count']}, margin={t3['margin']}, d={t3['embed_dim']}")
    print(f"Сумма triplet loss до SGD: {t3['loss_sum_initial']:.6g}")
    print(f"Сумма triplet loss после SGD: {t3['loss_sum_final']:.6g}")
    print("Итоговые эмбеддинги (строки A1..B2):\n", np.round(t3["E_final"], 6))

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
