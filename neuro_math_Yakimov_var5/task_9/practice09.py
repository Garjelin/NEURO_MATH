"""
Практическая работа №9, вариант 5.

Таблица 9.9: тема "Анализ стиля"; формулировка — разработать модель,
которая подбирает стиль изображения; набор данных — WikiArt.

Реализация воспроизводимого учебного варианта: формируется компактный набор
изображений с признаками, характерными для нескольких художественных стилей
WikiArt (цвет, контраст, текстура, геометричность). По изображениям извлекаются
численные признаки; MLPClassifier обучается выбирать стиль изображения.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PDF_ASSIGNMENT_RU = (
    "Вариант 5. Тема: анализ стиля.\n"
    "Формулировка: разработайте модель, которая подбирает стиль изображения.\n"
    "Набор данных: WikiArt (арт-изображения)."
)


@dataclass(frozen=True)
class StyleSpec:
    key: str
    title: str
    palette: tuple[tuple[float, float, float], ...]


STYLE_SPECS: tuple[StyleSpec, ...] = (
    StyleSpec(
        "impressionism",
        "Импрессионизм",
        ((0.90, 0.72, 0.35), (0.30, 0.55, 0.90), (0.85, 0.45, 0.62), (0.70, 0.85, 0.55)),
    ),
    StyleSpec(
        "cubism",
        "Кубизм",
        ((0.78, 0.65, 0.48), (0.35, 0.32, 0.28), (0.62, 0.50, 0.36), (0.18, 0.22, 0.30)),
    ),
    StyleSpec(
        "expressionism",
        "Экспрессионизм",
        ((0.95, 0.12, 0.12), (0.05, 0.18, 0.75), (0.98, 0.75, 0.08), (0.20, 0.06, 0.28)),
    ),
    StyleSpec(
        "realism",
        "Реализм",
        ((0.42, 0.33, 0.24), (0.70, 0.62, 0.52), (0.20, 0.32, 0.22), (0.62, 0.70, 0.82)),
    ),
    StyleSpec(
        "abstract",
        "Абстракционизм",
        ((0.95, 0.90, 0.10), (0.10, 0.70, 0.85), (0.92, 0.20, 0.70), (0.05, 0.05, 0.05)),
    ),
)


def _rng_color(rng: np.random.Generator, palette: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    base = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float64)
    return np.clip(base + rng.normal(0.0, 0.055, size=3), 0.0, 1.0)


def _smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int = 8) -> np.ndarray:
    small = rng.random((max(2, h // scale), max(2, w // scale), 3))
    img = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)[:h, :w]
    return img


def _draw_rect(img: np.ndarray, rng: np.random.Generator, color: np.ndarray, *, alpha: float = 0.85) -> None:
    h, w, _ = img.shape
    y0 = int(rng.integers(0, h - 5))
    x0 = int(rng.integers(0, w - 5))
    y1 = int(rng.integers(y0 + 4, h))
    x1 = int(rng.integers(x0 + 4, w))
    img[y0:y1, x0:x1] = (1.0 - alpha) * img[y0:y1, x0:x1] + alpha * color


def _draw_stroke(img: np.ndarray, rng: np.random.Generator, color: np.ndarray, *, length: int, width: int) -> None:
    h, w, _ = img.shape
    y = int(rng.integers(0, h))
    x = int(rng.integers(0, w))
    dy = int(rng.integers(-2, 3))
    dx = int(rng.integers(-2, 3))
    if dy == 0 and dx == 0:
        dx = 1
    for t in range(length):
        yy = np.clip(y + t * dy, 0, h - 1)
        xx = np.clip(x + t * dx, 0, w - 1)
        y0, y1 = max(0, yy - width), min(h, yy + width + 1)
        x0, x1 = max(0, xx - width), min(w, xx + width + 1)
        img[y0:y1, x0:x1] = 0.35 * img[y0:y1, x0:x1] + 0.65 * color


def _generate_style_image(
    style: StyleSpec,
    rng: np.random.Generator,
    *,
    size: int = 64,
) -> np.ndarray:
    img = _smooth_noise(rng, size, size, scale=10)
    palette = style.palette

    if style.key == "impressionism":
        img *= 0.45
        img += 0.35
        for _ in range(110):
            _draw_stroke(img, rng, _rng_color(rng, palette), length=int(rng.integers(3, 9)), width=1)

    elif style.key == "cubism":
        img[:] = _rng_color(rng, palette)
        for _ in range(42):
            _draw_rect(img, rng, _rng_color(rng, palette), alpha=float(rng.uniform(0.65, 0.95)))
        # Контрастные линии усиливают геометричность.
        for x in range(0, size, int(rng.integers(7, 12))):
            img[:, x : x + 1] *= 0.25
        for y in range(0, size, int(rng.integers(8, 13))):
            img[y : y + 1, :] *= 0.25

    elif style.key == "expressionism":
        img *= 0.25
        for _ in range(70):
            _draw_stroke(
                img,
                rng,
                _rng_color(rng, palette),
                length=int(rng.integers(10, 24)),
                width=int(rng.integers(1, 3)),
            )
        img = np.clip((img - 0.5) * 1.5 + 0.5, 0.0, 1.0)

    elif style.key == "realism":
        yy, xx = np.mgrid[0:size, 0:size]
        grad = (0.45 * xx / size + 0.55 * yy / size)[..., None]
        base = np.array(palette[1], dtype=np.float64)
        shade = np.array(palette[2], dtype=np.float64)
        img = 0.65 * base + 0.35 * shade * grad
        img += rng.normal(0.0, 0.035, img.shape)
        for _ in range(8):
            _draw_rect(img, rng, _rng_color(rng, palette), alpha=0.18)

    else:  # abstract
        img[:] = _rng_color(rng, palette)
        for _ in range(32):
            if rng.random() < 0.55:
                _draw_rect(img, rng, _rng_color(rng, palette), alpha=float(rng.uniform(0.55, 0.95)))
            else:
                _draw_stroke(
                    img,
                    rng,
                    _rng_color(rng, palette),
                    length=int(rng.integers(6, 18)),
                    width=int(rng.integers(1, 4)),
                )
    return np.clip(img, 0.0, 1.0)


def _rgb_to_saturation(img: np.ndarray) -> np.ndarray:
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    return (mx - mn) / np.maximum(mx, 1e-8)


def _edge_energy(gray: np.ndarray) -> float:
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))


def _blockiness(gray: np.ndarray, step: int = 8) -> float:
    vertical = [float(np.mean(np.abs(gray[:, i] - gray[:, i - 1]))) for i in range(step, gray.shape[1], step)]
    horizontal = [float(np.mean(np.abs(gray[i, :] - gray[i - 1, :]))) for i in range(step, gray.shape[0], step)]
    vals = vertical + horizontal
    return float(np.mean(vals)) if vals else 0.0


def extract_style_features(img: np.ndarray) -> np.ndarray:
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    sat = _rgb_to_saturation(img)
    features = [
        *img.mean(axis=(0, 1)).tolist(),
        *img.std(axis=(0, 1)).tolist(),
        float(sat.mean()),
        float(sat.std()),
        float(gray.mean()),
        float(gray.std()),
        _edge_energy(gray),
        _blockiness(gray),
        float(np.percentile(gray, 90) - np.percentile(gray, 10)),
    ]
    return np.array(features, dtype=np.float64)


def build_dataset(
    *,
    n_per_style: int = 80,
    image_size: int = 64,
    seed: int = 905,
) -> dict:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for label, spec in enumerate(STYLE_SPECS):
        for _ in range(n_per_style):
            img = _generate_style_image(spec, rng, size=image_size)
            images.append(img)
            X_rows.append(extract_style_features(img))
            y_rows.append(label)

    return {
        "images": images,
        "X": np.stack(X_rows),
        "y": np.array(y_rows, dtype=np.int64),
        "style_keys": [s.key for s in STYLE_SPECS],
        "style_titles": [s.title for s in STYLE_SPECS],
    }


def train_style_selector(dataset: dict, *, random_state: int = 42) -> dict:
    X = dataset["X"]
    y = dataset["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=5000,
        random_state=random_state,
        alpha=1e-4,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=35,
        tol=1e-5,
    )
    clf.fit(X_tr, y_train)
    pred_train = clf.predict(X_tr)
    pred_test = clf.predict(X_te)
    labels = list(range(len(dataset["style_titles"])))

    return {
        "scaler": scaler,
        "model": clf,
        "accuracy_train": float(accuracy_score(y_train, pred_train)),
        "accuracy_test": float(accuracy_score(y_test, pred_test)),
        "confusion": confusion_matrix(y_test, pred_test, labels=labels),
        "report": classification_report(
            y_test,
            pred_test,
            labels=labels,
            target_names=dataset["style_titles"],
            zero_division=0,
            output_dict=True,
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_iter": int(clf.n_iter_),
    }


def make_query_images(*, seed: int = 119) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = (STYLE_SPECS[0], STYLE_SPECS[2], STYLE_SPECS[4])
    queries = []
    for idx, spec in enumerate(selected, start=1):
        img = _generate_style_image(spec, rng, size=64)
        queries.append(
            {
                "name": f"Q{idx}",
                "expected_style": spec.title,
                "image": img,
                "features": extract_style_features(img),
            }
        )
    return queries


def predict_styles(queries: list[dict], trained: dict, style_titles: list[str]) -> list[dict]:
    scaler = trained["scaler"]
    clf = trained["model"]
    Xq = scaler.transform(np.stack([q["features"] for q in queries]))
    pred = clf.predict(Xq)
    proba = clf.predict_proba(Xq)
    out = []
    for q, label, probs in zip(queries, pred, proba, strict=True):
        order = np.argsort(probs)[::-1][:3]
        out.append(
            {
                "name": q["name"],
                "expected_style": q["expected_style"],
                "predicted_style": style_titles[int(label)],
                "confidence": float(probs[int(label)]),
                "top3": [(style_titles[int(i)], float(probs[int(i)])) for i in order],
                "image": q["image"],
            }
        )
    return out


def _save_confusion(path: Path, mat: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Эталон")
    ax.set_title("Практика №9: матрица ошибок по стилям")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_query_grid(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 4))
    if len(predictions) == 1:
        axes = [axes]
    for ax, item in zip(axes, predictions, strict=True):
        ax.imshow(item["image"])
        ax.set_axis_off()
        ax.set_title(
            f"{item['name']}\nпрогноз: {item['predicted_style']}\nP={item['confidence']:.2f}",
            fontsize=10,
        )
    fig.suptitle("Контрольные изображения и подобранные стили")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_style_samples(path: Path, dataset: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = dataset["images"]
    titles = dataset["style_titles"]
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i * 80])
        ax.set_axis_off()
        ax.set_title(titles[i], fontsize=10)
    fig.suptitle("Примеры стилевых изображений учебного набора")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_practice09(out_dir: Path | str | None = None, save_png: bool = True) -> dict:
    out = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    trained = train_style_selector(dataset)
    queries = make_query_images()
    predictions = predict_styles(queries, trained, dataset["style_titles"])

    p_conf = out / "practice09_confusion_matrix.png"
    p_queries = out / "practice09_query_predictions.png"
    p_samples = out / "practice09_style_samples.png"
    if save_png:
        _save_confusion(p_conf, trained["confusion"], dataset["style_titles"])
        _save_query_grid(p_queries, predictions)
        _save_style_samples(p_samples, dataset)

    return {
        "out_dir": str(out.resolve()),
        "pdf_assignment": PDF_ASSIGNMENT_RU,
        "styles": dataset["style_titles"],
        "n_samples": int(len(dataset["y"])),
        "n_features": int(dataset["X"].shape[1]),
        "train": trained,
        "predictions": predictions,
        "plots": {
            "samples": str(p_samples) if save_png else None,
            "confusion": str(p_conf) if save_png else None,
            "queries": str(p_queries) if save_png else None,
        },
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    bar = "=" * 72
    if verbose:
        print(bar)
        print("Практическая работа №9, вариант 5")
        print(bar)

    print(f"\n{bar}\nУсловие\n{bar}")
    print(data["pdf_assignment"])
    print("\nСтили в модели:", ", ".join(data["styles"]))
    print(f"Размер учебного набора: {data['n_samples']} изображений")
    print(f"Число признаков изображения: {data['n_features']}")

    tr = data["train"]
    print(f"\n{bar}\nКачество модели подбора стиля\n{bar}")
    print(f"n_train={tr['n_train']}, n_test={tr['n_test']}, итераций={tr['n_iter']}")
    print(f"Accuracy train: {tr['accuracy_train']:.4f}")
    print(f"Accuracy test:  {tr['accuracy_test']:.4f}")
    print("Матрица ошибок (строки — эталон, столбцы — прогноз):")
    print(tr["confusion"])

    print(f"\n{bar}\nПодбор стиля для контрольных изображений\n{bar}")
    for item in data["predictions"]:
        print(
            f"{item['name']}: ожидаемый стиль — {item['expected_style']}; "
            f"подобранный стиль — {item['predicted_style']}; уверенность={item['confidence']:.4f}"
        )
        top = "; ".join(f"{name}: {prob:.3f}" for name, prob in item["top3"])
        print(f"      top-3: {top}")

    print(
        "\nВывод: модель извлекает численные признаки изображения и по ним выбирает наиболее вероятный "
        "художественный стиль; качество оценивается accuracy на тестовой части и матрицей ошибок."
    )

    print("\n[файлы]")
    print(f"      {data['out_dir']}")
    for k, p in data["plots"].items():
        if p:
            print(f"      • {k}: {p}")


def main() -> None:
    data = run_practice09()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
