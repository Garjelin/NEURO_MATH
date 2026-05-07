"""
Практическая работа №1, вариант 5.

Задание А: минимум f(x) = x^3 - 3 sin(x), интервал начальной точки [0; 1], x0 = 0.
Задание Б: минимум f(x1,x2) = 2x1^2 + 2x2^2 + 2x1x2 - 14x1 - 12x2 + 29, x0 = (-1, 4).
Методы: градиентный спуск, покоординатный спуск, адаптивный градиентный спуск (Adagrad).

Дополнительное задание: подбор w1, w2, h для персептрона по таблице (вес, рост) -> пол.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# Вектор неизвестных везде как ndarray float64 — так удобно и для 1D, и для 2D.
Vector = np.ndarray


@dataclass
class OptResult:
    """Итог одного метода: точка, значение f, число шагов и история f для графика."""

    method: str
    point: Vector
    value: float
    iterations: int
    history: list[float]


def f_a(x: Vector) -> float:
    """Целевая функция задания А (одна переменная)."""
    x0 = float(x[0])
    return x0**3 - 3.0 * math.sin(x0)


def grad_a(x: Vector) -> Vector:
    """Производная f_a: нужна для всех градиентных методов."""
    x0 = float(x[0])
    return np.array([3.0 * x0**2 - 3.0 * math.cos(x0)], dtype=float)


def f_b(x: Vector) -> float:
    """Квадратичная целевая функция задания Б (выпуклая → один глобальный минимум)."""
    x1, x2 = float(x[0]), float(x[1])
    return 2 * x1**2 + 2 * x2**2 + 2 * x1 * x2 - 14 * x1 - 12 * x2 + 29


def grad_b(x: Vector) -> Vector:
    """Градиент f_b: частные производные по x1 и x2."""
    x1, x2 = float(x[0]), float(x[1])
    return np.array([4 * x1 + 2 * x2 - 14, 2 * x1 + 4 * x2 - 12], dtype=float)


def gradient_descent(
    f: Callable[[Vector], float],
    grad: Callable[[Vector], Vector],
    x0: Vector,
    lr: float = 0.05,
    tol: float = 1e-7,
    max_iter: int = 10_000,
) -> OptResult:
    # Классический спуск против антиградиента. Backtracking: если шаг не уменьшает f,
    # уменьшаем длину шага — так проще подобрать lr без ручного подбора под каждую задачу.
    x = x0.astype(float).copy()
    history = [f(x)]

    for _ in range(1, max_iter + 1):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        step = lr
        current = f(x)
        while step > 1e-12:
            xn = x - step * g
            fn = f(xn)
            if fn < current:
                x = xn
                history.append(fn)
                break
            step *= 0.5
        else:
            break
    return OptResult("Градиентный спуск", x, f(x), len(history) - 1, history)


def coordinate_descent(
    f: Callable[[Vector], float],
    grad: Callable[[Vector], Vector],
    x0: Vector,
    lr: float = 0.1,
    tol: float = 1e-7,
    max_iter: int = 10_000,
) -> OptResult:
    # За один «внешний» проход цикла обновляем координаты по очереди (Gauss–Seidel-подобно).
    x = x0.astype(float).copy()
    history = [f(x)]
    n = len(x)

    for k in range(1, max_iter + 1):
        x_prev = x.copy()
        for j in range(n):
            gj = grad(x)[j]
            step = lr
            current = f(x)
            while step > 1e-12:
                xn = x.copy()
                xn[j] = xn[j] - step * gj
                fn = f(xn)
                if fn < current:
                    x = xn
                    history.append(fn)
                    break
                step *= 0.5
        if np.linalg.norm(x - x_prev) < tol:
            break
        if k >= max_iter:
            break
    return OptResult("Покоординатный спуск", x, f(x), len(history) - 1, history)


def adaptive_gradient_descent(
    f: Callable[[Vector], float],
    grad: Callable[[Vector], Vector],
    x0: Vector,
    lr: float = 0.8,
    tol: float = 1e-7,
    max_iter: int = 20_000,
    eps: float = 1e-8,
) -> OptResult:
    # Adagrad: накапливаем квадраты компонент градиента и делим шаг по каждой оси.
    # Меньше ручной настройки lr по сравнению с обычным градиентным спуском.
    x = x0.astype(float).copy()
    g2 = np.zeros_like(x)
    history = [f(x)]

    for _ in range(1, max_iter + 1):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        g2 += g * g
        adj_lr = lr / (np.sqrt(g2) + eps)
        x = x - adj_lr * g
        history.append(f(x))
    return OptResult("Адаптивный градиентный спуск", x, f(x), len(history) - 1, history)


def run_additional_task(
    samples: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    lr: float = 0.2,
    max_epochs: int = 300,
) -> tuple[Vector, float, list[int]]:
    """
    Персептрон: y in {0,1}. Обучение правилом Розенблатта на нормированных признаках.

    samples: матрица Nx2 [вес_кг, рост_см]. По умолчанию A1–A3 из методички.
    labels: вектор длины N, 1 — муж., 0 — жен.
    """
    if samples is None:
        samples = np.array(
            [
                [84.0, 180.0],
                [62.0, 172.0],
                [57.0, 165.0],
            ],
            dtype=float,
        )
    if labels is None:
        labels = np.array([1, 0, 0], dtype=int)

    x = np.asarray(samples, dtype=float)
    y = np.asarray(labels, dtype=int)

    # Нормировка: вес и рост в разных единицах и масштабах; без неё шаг обучения
    # «тянет» вес в сторону признака с большей дисперсией.
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    xn = (x - mu) / sigma

    w = np.zeros(2, dtype=float)
    h = 0.0

    # Правило Розенблатта: при ошибке классификации сдвигаем веса в сторону входа.
    for _ in range(max_epochs):
        errors = 0
        for xi, target in zip(xn, y):
            score = float(np.dot(w, xi) + h)
            pred = 1 if score >= 0 else 0
            delta = int(target) - pred
            if delta != 0:
                w += lr * delta * xi
                h += lr * delta
                errors += 1
        if errors == 0:
            break

    preds = [1 if float(np.dot(w, xi) + h) >= 0 else 0 for xi in xn]
    return w, h, preds


def save_loss_plot(path: Path, title: str, results: list[OptResult]) -> None:
    # Один рисунок — три кривые f по номеру шага; удобно для отчёта и проверки сходимости.
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for r in results:
        plt.plot(r.history, label=r.method)
    plt.title(title)
    plt.xlabel("Итерация")
    plt.ylabel("Значение функции")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_practice01(
    out_dir: Path | None = None,
    save_png: bool = True,
) -> dict:
    """
    Выполняет все части практики 1. Возвращает словарь с результатами.

    Используется и при запуске этого модуля (консоль + графики), и отдельным скриптом
    сборки отчёта — без вывода в консоль и без создания DOCX (только данные и PNG).
    """
    if out_dir is None:
        out_dir = Path("outputs") / "practice01"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Начальные точки из варианта 5: А — внутри [0;1] взято x0=0; Б — (-1, 4).
    x0_a = np.array([0.0])
    x0_b = np.array([-1.0, 4.0])

    # Параметры lr подобраны так, чтобы методы стабильно сходились на обеих функциях.
    task_a_results = [
        gradient_descent(f_a, grad_a, x0_a, lr=0.05),
        coordinate_descent(f_a, grad_a, x0_a, lr=0.1),
        adaptive_gradient_descent(f_a, grad_a, x0_a, lr=0.8),
    ]
    task_b_results = [
        gradient_descent(f_b, grad_b, x0_b, lr=0.1),
        coordinate_descent(f_b, grad_b, x0_b, lr=0.2),
        adaptive_gradient_descent(f_b, grad_b, x0_b, lr=0.8),
    ]

    add_w, add_h, add_preds = run_additional_task()

    if save_png:
        save_loss_plot(
            out_dir / "task_A_convergence.png",
            "Сходимость методов (задание А)",
            task_a_results,
        )
        save_loss_plot(
            out_dir / "task_B_convergence.png",
            "Сходимость методов (задание Б)",
            task_b_results,
        )

    return {
        "out_dir": out_dir.resolve(),
        "task_a": task_a_results,
        "task_b": task_b_results,
        "additional": {"w": add_w, "h": add_h, "preds": add_preds},
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    """Печать в консоль: при verbose=True — пошагово видно ход работы программы."""
    sep = "=" * 72

    if verbose:
        print(sep)
        print("Практическая работа №1, вариант 5 — результаты численного эксперимента")
        print(sep)
        print("\n[1/3] Задание А: f(x) = x³ − 3·sin(x)")
        print(f"      Старт: x₀ = 0 (допустимый интервал по заданию: [0; 1])")
        print(f"      Сравнение трёх методов минимизации (см. также график в PNG).\n")

    print("Задание А — итоги по методам:")
    for r in data["task_a"]:
        gn = np.linalg.norm(grad_a(r.point))
        print(f"  • {r.method}")
        print(f"      x*     = {float(r.point[0]):.10f}")
        print(f"      f(x*)  = {r.value:.10f}")
        print(f"      ‖f'‖ в конце ≈ {gn:.2e}  |  шагов в истории: {r.iterations}")

    if verbose:
        print(f"\n[2/3] Задание Б: квадратичная f(x₁, x₂) из варианта 5")
        print(f"      Старт: (x₁⁽⁰⁾, x₂⁽⁰⁾) = (-1, 4)")
        print(
            "      Эталон (аналитика, ∇f=0): x* ≈ (8/3, 5/3), f_min = 1/3 ≈ 0.333333...\n"
        )

    print("Задание Б — итоги по методам:")
    for r in data["task_b"]:
        gn = np.linalg.norm(grad_b(r.point))
        print(f"  • {r.method}")
        print(f"      (x₁*, x₂*) = ({float(r.point[0]):.10f}, {float(r.point[1]):.10f})")
        print(f"      f(x*)       = {r.value:.10f}")
        print(f"      ‖∇f‖ в конце ≈ {gn:.2e}  |  шагов в истории: {r.iterations}")

    add = data["additional"]
    if verbose:
        print(f"\n[3/3] Дополнительное задание: персептрон (вес, рост) → пол")
        print("      Веса ниже — в пространстве нормированных признаков (z-score по столбцам).")

    print("\nДополнительное задание:")
    print(f"  • w₁ = {float(add['w'][0]):.10f},  w₂ = {float(add['w'][1]):.10f},  h = {float(add['h']):.10f}")
    print(f"  • Предсказания на обучающей выборке (0/1): {add['preds']}")


def main() -> None:
    data = run_practice01()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
