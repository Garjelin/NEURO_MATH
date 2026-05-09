"""
Практическая работа №4, вариант 5.

Задание 1 (табл. 4.3 / 4.4): дерево решений по строкам 7–10; метрика на логическом тесте — MSE.
Задание 2 (табл. 4.5 / 4.6): два бинарных признака x, z; обучение по строкам 2,3,5,8,9,10;
на оставшихся строках — MAE.
Задание 3 (табл. 4.7): граф вычислений и прямой/обратный проход для
    f(x,y) = exp(x**y) - cos(x/y) + 1/sin(x+y).
Область определения: x > 0 (для степени x^y и ∂/∂y через ln(x)), y ≠ 0, sin(x+y) ≠ 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree

# --- Задание 1: таблица 4.3 (аналог табл. 2.7), вариант 5 → строки 7–10 ---------------
TABLE_43_FULL = np.array(
    [
        [-1.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 4.0],
        [-1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],  # 7 train var 5
        [1.0, 0.0],
        [2.0, 3.0],
        [2.0, -2.0],
    ],
    dtype=float,
)

VARIANT_5_TASK1_TRAIN_ROWS = (7, 8, 9, 10)
VARIANT_5_TASK1_TEST_ROWS = tuple(i for i in range(1, 11) if i not in VARIANT_5_TASK1_TRAIN_ROWS)

# --- Задание 2: таблица 4.5 (номер строки, x, z, Y) ------------------------------------
# № | x | z | Y
TABLE_45 = np.array(
    [
        [0.0, 0.0, 0.0],  # 1
        [0.0, 0.0, 1.0],  # 2
        [1.0, 1.0, 2.0],  # 3
        [1.0, 1.0, 1.0],  # 4
        [1.0, 1.0, 0.0],  # 5
        [0.0, 1.0, 4.0],  # 6
        [1.0, 0.0, 4.0],  # 7
        [1.0, 1.0, 6.0],  # 8
        [1.0, 1.0, 8.0],  # 9
        [1.0, 0.0, 10.0],  # 10
    ],
    dtype=float,
)

VARIANT_5_TASK2_TRAIN_ROWS = (2, 3, 5, 8, 9, 10)
VARIANT_5_TASK2_TEST_ROWS = tuple(i for i in range(1, 11) if i not in VARIANT_5_TASK2_TRAIN_ROWS)


def _rows_table43(rows_1based: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    idx0 = [i - 1 for i in rows_1based]
    xy = TABLE_43_FULL[idx0]
    return xy[:, :1], xy[:, 1]


def _rows_table45(rows_1based: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    idx0 = [i - 1 for i in rows_1based]
    block = TABLE_45[idx0]
    return block[:, :2], block[:, 2]


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def run_task1_decision_tree(random_state: int = 42) -> dict:
    """Дерево решений (MSE-критерий sklearn по умолчанию для регрессии)."""
    X_train, y_train = _rows_table43(VARIANT_5_TASK1_TRAIN_ROWS)
    X_test, y_test = _rows_table43(VARIANT_5_TASK1_TEST_ROWS)

    tree = DecisionTreeRegressor(random_state=random_state, criterion="squared_error")
    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)

    return {
        "train_rows": VARIANT_5_TASK1_TRAIN_ROWS,
        "test_rows": VARIANT_5_TASK1_TEST_ROWS,
        "model": tree,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "mse_train": _mse(y_train, y_pred_train),
        "mse_test": _mse(y_test, y_pred_test),
    }


def run_task2_decision_tree(random_state: int = 43) -> dict:
    """
    Для минимизации MAE используем criterion='absolute_error' (sklearn ≥ 1.0).
    При недоступности критерия — squared_error и отчёт MAE всё равно считается на тесте.
    """
    X_train, y_train = _rows_table45(VARIANT_5_TASK2_TRAIN_ROWS)
    X_test, y_test = _rows_table45(VARIANT_5_TASK2_TEST_ROWS)

    try:
        tree = DecisionTreeRegressor(random_state=random_state, criterion="absolute_error")
    except TypeError:
        tree = DecisionTreeRegressor(random_state=random_state, criterion="squared_error")

    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)

    return {
        "train_rows": VARIANT_5_TASK2_TRAIN_ROWS,
        "test_rows": VARIANT_5_TASK2_TEST_ROWS,
        "model": tree,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "tree_criterion": getattr(tree, "criterion", "unknown"),
    }


def f_variant5_forward_backward(x: float, y: float) -> dict:
    """
    Прямой проход по узлам графа и обратное распространение по тем же узлам.

    Узлы:
      n1 = x**y, n2 = exp(n1), n3 = x/y, n4 = cos(n3),
      n5 = x+y, n6 = sin(n5), n7 = 1/n6,
      f = n2 - n4 + n7.

    Для аналитических ∂n1/∂x, ∂n1/∂y требуется x > 0 (∂(x^y)/∂y = x^y ln x).
    """
    if x <= 0:
        raise ValueError("Для узла x^y и производной по y нужно x > 0 (используется ln(x)).")
    if y == 0:
        raise ValueError("y не может быть нулём (узел x/y).")
    n5 = x + y
    sn = np.sin(n5)
    if abs(sn) < 1e-15:
        raise ValueError("sin(x+y) не может быть нулём (узел 1/sin(x+y)).")

    n1 = x**y
    n2 = np.exp(n1)
    n3 = x / y
    n4 = np.cos(n3)
    n6 = sn
    n7 = 1.0 / n6
    fval = n2 - n4 + n7

    # Обратный проход (accumulate градиенты по промежуточным узлам)
    gn2 = 1.0
    gn4 = -1.0
    gn7 = 1.0

    gn6 = gn7 * (-1.0 / (n6 * n6))
    gn5 = gn6 * np.cos(n5)

    gn3 = gn4 * (-np.sin(n3))

    gn1 = gn2 * n2

    dn1_dx = y * x ** (y - 1.0)
    dn1_dy = n1 * np.log(x)

    gx = gn1 * dn1_dx + gn3 * (1.0 / y) + gn5 * 1.0
    gy = gn1 * dn1_dy + gn3 * (-x / (y * y)) + gn5 * 1.0

    return {
        "x": x,
        "y": y,
        "nodes": {
            "n1_x_pow_y": float(n1),
            "n2_exp": float(n2),
            "n3_x_div_y": float(n3),
            "n4_cos": float(n4),
            "n5_x_plus_y": float(n5),
            "n6_sin": float(n6),
            "n7_inv_sin": float(n7),
        },
        "f": float(fval),
        "df_dx": float(gx),
        "df_dy": float(gy),
    }


def _numeric_grad(x: float, y: float, eps: float = 1e-6) -> tuple[float, float]:
    fxp = f_variant5_forward_backward(x + eps, y)["f"]
    fxm = f_variant5_forward_backward(x - eps, y)["f"]
    df_dx = (fxp - fxm) / (2 * eps)

    fyp = f_variant5_forward_backward(x, y + eps)["f"]
    fym = f_variant5_forward_backward(x, y - eps)["f"]
    df_dy = (fyp - fym) / (2 * eps)
    return df_dx, df_dy


def run_task3_graph_demo(points: list[tuple[float, float]] | None = None) -> dict:
    if points is None:
        # x > 0 — для степени x^y и производной ∂/∂y = x^y ln x в вещественном анализе
        points = [(0.5, 1.0), (1.0, 0.8), (2.0, 0.5)]

    rows = []
    for x, y in points:
        ana = f_variant5_forward_backward(x, y)
        num_dx, num_dy = _numeric_grad(x, y)
        rows.append(
            {
                "point": (x, y),
                "forward": ana["nodes"],
                "f": ana["f"],
                "df_dx_analytic": ana["df_dx"],
                "df_dy_analytic": ana["df_dy"],
                "df_dx_numeric": num_dx,
                "df_dy_numeric": num_dy,
            }
        )

    return {"points": rows}


def _save_tree_figure(tree: DecisionTreeRegressor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(tree, ax=ax, filled=True, rounded=True, fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_task1_predictions_plot(task1: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-1.5, 2.5, 400).reshape(-1, 1)
    model: DecisionTreeRegressor = task1["model"]
    y_grid = model.predict(grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(task1["X_train"].ravel(), task1["y_train"], c="tab:blue", s=80, label="Обучение", zorder=3)
    ax.scatter(task1["X_test"].ravel(), task1["y_test"], c="tab:orange", s=80, marker="s", label="Тест", zorder=3)
    ax.step(grid.ravel(), y_grid, where="post", color="tab:green", lw=1.8, label="Предсказание дерева")
    ax.set_xlabel("x")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Задание 1: дерево решений (вариант 5)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_task2_bar(task2: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_t = task2["y_test"]
    y_p = task2["y_pred_test"]
    idx = np.arange(len(y_t))

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    ax.bar(idx - w / 2, y_t, width=w, label="Факт Y")
    ax.bar(idx + w / 2, y_p, width=w, label="Предсказание")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"стр.{r}" for r in task2["test_rows"]])
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Задание 2: тест MAE={task2['mae_test']:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_practice04(
    out_dir: Path | str | None = None,
    save_png: bool = True,
) -> dict:
    """
    Полный прогон практики №4. Возвращает словарь метрик и объектов для отчёта DOCX.

    По умолчанию PNG — в outputs/practice04/ (относительно текущей рабочей директории).

    Консольный вывод не выполняется; для вывода в терминале см. print_results().
    """
    out = Path(out_dir) if out_dir is not None else Path("outputs") / "practice04"
    out.mkdir(parents=True, exist_ok=True)

    task1 = run_task1_decision_tree()
    task2 = run_task2_decision_tree()
    task3 = run_task3_graph_demo()

    png_task1_tree = out / "practice04_task1_tree.png"
    png_task1_pred = out / "practice04_task1_predictions.png"
    png_task2_tree = out / "practice04_task2_tree.png"
    png_task2_bar = out / "practice04_task2_test_bar.png"

    if save_png:
        _save_tree_figure(task1["model"], png_task1_tree, "Дерево (задание 1)")
        _save_task1_predictions_plot(task1, png_task1_pred)
        _save_tree_figure(task2["model"], png_task2_tree, "Дерево (задание 2)")
        _save_task2_bar(task2, png_task2_bar)

    return {
        "out_dir": str(out.resolve()),
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "plots": {
            "task1_tree": str(png_task1_tree) if save_png else None,
            "task1_predictions": str(png_task1_pred) if save_png else None,
            "task2_tree": str(png_task2_tree) if save_png else None,
            "task2_bar": str(png_task2_bar) if save_png else None,
        },
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    """Подробный вывод в консоль (при запуске модуля как скрипта)."""
    bar = "=" * 72

    if verbose:
        print(bar)
        print("Практическая работа №4, вариант 5")
        print(bar)

    task1 = data["task1"]
    print(f"\n{bar}\n[1/3] Задание 1: дерево решений по табл. 4.3 (обучение: строки 7–10), метрика MSE\n{bar}")
    print(f"Строки обучения (1-based): {task1['train_rows']}")
    print(f"Строки теста: {task1['test_rows']}")
    print(f"MSE на обучении: {task1['mse_train']:.6g}")
    print(f"MSE на логическом тесте: {task1['mse_test']:.6g}")

    task2 = data["task2"]
    print(f"\n{bar}\n[2/3] Задание 2: дерево по табл. 4.5 (обучение: строки 2,3,5,8,9,10), метрика MAE\n{bar}")
    print(f"Строки обучения: {task2['train_rows']}")
    print(f"Строки теста: {task2['test_rows']}")
    print(f"Критерий дерева (sklearn): {task2['tree_criterion']}")
    print(f"MAE на обучении: {task2['mae_train']:.6g}")
    print(f"MAE на тесте: {task2['mae_test']:.6g}")

    task3 = data["task3"]
    print(
        f"\n{bar}\n[3/3] Задание 3: граф f(x,y)=exp(x^y)-cos(x/y)+1/sin(x+y), прямой и обратный проход\n{bar}"
    )
    for row in task3["points"]:
        x, y = row["point"]
        print(f"\nТочка ({x}, {y}):")
        print(f"  f = {row['f']:.6g}")
        for name, val in row["forward"].items():
            print(f"  {name} = {val:.6g}")
        print(f"  ∂f/∂x (аналит.) = {row['df_dx_analytic']:.6g}, (числ.) = {row['df_dx_numeric']:.6g}")
        print(f"  ∂f/∂y (аналит.) = {row['df_dy_analytic']:.6g}, (числ.) = {row['df_dy_numeric']:.6g}")

    print("\n[файлы]")
    print(f"      {data['out_dir']}")
    for key, p in data["plots"].items():
        if p:
            print(f"      • {key}: {p}")


def main() -> None:
    data = run_practice04()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
