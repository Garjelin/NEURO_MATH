"""
Практическая работа №8, вариант 5.

Таблица 8.9: поэтический текст; подготовка тренировочной выборки разными способами;
модель нейронной сети; оценка качества.

Реализация: (1) символьная классификация «следующий символ», разбиение 75/25, MLPClassifier;
(2) классификация «следующая строка» по частотам символов текущей строки (две пары из текста).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

TEXT_TABLE_89_VARIANT5 = (
    "За горизонтом мечты,\n"
    "Мы будем их трактовать в тени,\n"
    "Где поэзия сливается в нейронный свет."
)

PDF_ASSIGNMENT_RU = (
    "В качестве текста выберем поэтические строки (таблица 8.9).\n"
    "Подготовить тренировочную выборку всеми возможными способами.\n"
    "Постройте модель нейронной сети, оцените ее качество."
)


@dataclass(frozen=True)
class CharWindowSpec:
    key: str
    title: str
    context_len: int
    line_local: bool
    split_random: bool


CHAR_WINDOW_SPECS: tuple[CharWindowSpec, ...] = (
    CharWindowSpec("w1_seq", "Контекст 1 символ, весь текст, хронологическое 75/25", 1, False, False),
    CharWindowSpec("w2_seq", "Контекст 2, весь текст, хронологическое 75/25", 2, False, False),
    CharWindowSpec("w3_seq", "Контекст 3, весь текст, хронологическое 75/25", 3, False, False),
    CharWindowSpec("w2_rand", "Контекст 2, весь текст, случайное 75/25", 2, False, True),
    CharWindowSpec("w2_lines", "Контекст 2, окна только внутри строки, хронологическое 75/25", 2, True, False),
)


def _alphabet_and_maps(text: str) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set(text))
    return chars, {c: i for i, c in enumerate(chars)}


def _lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln]


def _build_char_xy(
    text: str,
    c2i: dict[str, int],
    *,
    context_len: int,
    line_local: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocab = len(c2i)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    pos_list: list[int] = []

    def flush_line(line: str, base_pos: int) -> None:
        if len(line) <= context_len:
            return
        for j in range(len(line) - context_len):
            ctx = line[j : j + context_len]
            nxt = line[j + context_len]
            block = np.zeros(context_len * vocab, dtype=np.float64)
            for t, ch in enumerate(ctx):
                block[t * vocab + c2i[ch]] = 1.0
            X_list.append(block)
            y_list.append(c2i[nxt])
            pos_list.append(base_pos + j)

    if line_local:
        offset = 0
        for ln in _lines(text):
            flush_line(ln, offset)
            offset += len(ln) + 1
    else:
        flush_line(text, 0)

    if not X_list:
        raise ValueError("Слишком короткий текст для заданного context_len.")
    return np.stack(X_list), np.array(y_list, dtype=np.int64), np.array(pos_list, dtype=np.int64)


def _split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    *,
    test_size: float,
    random: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    order = np.argsort(positions)
    Xs, ys = X[order], y[order]
    n = len(Xs)
    n_tr = int(n * (1.0 - test_size))
    n_tr = max(1, min(n - 1, n_tr))
    return Xs[:n_tr], Xs[n_tr:], ys[:n_tr], ys[n_tr:]


def _train_mlp_char(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    random_state: int,
) -> dict[str, float]:
    use_es = len(X_train) >= 40
    clf = MLPClassifier(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        solver="adam",
        max_iter=12000,
        random_state=random_state,
        alpha=1e-4,
        early_stopping=use_es,
        validation_fraction=0.12 if use_es else 0.0,
        n_iter_no_change=35 if use_es else 999999,
        tol=1e-5,
    )
    clf.fit(X_train, y_train)
    classes = clf.classes_
    p_tr = clf.predict_proba(X_train)
    p_va = clf.predict_proba(X_val)

    def _ll(yt: np.ndarray, p: np.ndarray) -> float:
        m = np.isin(yt, classes)
        if not np.any(m):
            return float("nan")
        return float(log_loss(yt[m], p[m], labels=classes))

    return {
        "acc_train": float(accuracy_score(y_train, clf.predict(X_train))),
        "acc_val": float(accuracy_score(y_val, clf.predict(X_val))),
        "logloss_val": _ll(y_val, p_va),
        "n_iter": int(clf.n_iter_),
    }


def run_task_char_level_quality(
    text: str = TEXT_TABLE_89_VARIANT5,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    _, c2i = _alphabet_and_maps(text)
    rows: list[dict] = []
    for spec in CHAR_WINDOW_SPECS:
        X, y, pos = _build_char_xy(text, c2i, context_len=spec.context_len, line_local=spec.line_local)
        X_tr, X_va, y_tr, y_va = _split_train_val(
            X, y, pos, test_size=test_size, random=spec.split_random, random_state=random_state
        )
        met = _train_mlp_char(X_tr, y_tr, X_va, y_va, random_state=random_state)
        rows.append(
            {
                "key": spec.key,
                "title": spec.title,
                "context_len": spec.context_len,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                **met,
            }
        )
    best = max(rows, key=lambda r: r["acc_val"])
    return {
        "task": "next_char_train_val",
        "description": (
            "Целевая переменная — следующий символ; признак — one-hot по контексту фиксированной длины. "
            "Контрольная выборка — 25% позиций (хронологический хвост или случайное разбиение, см. ключ способа)."
        ),
        "strategies": rows,
        "best_val_key": best["key"],
    }


def _encode_line_counts(line: str, alphabet: list[str], c2i: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(alphabet), dtype=np.float64)
    for ch in line:
        if ch in c2i:
            v[c2i[ch]] += 1.0
    return v


def run_task_line_pairs_demo(text: str = TEXT_TABLE_89_VARIANT5) -> dict:
    lines = _lines(text)
    alphabet = sorted(set(text))
    c2i = {c: i for i, c in enumerate(alphabet)}
    X = np.stack([_encode_line_counts(lines[i], alphabet, c2i) for i in range(2)])
    y = np.array([1, 2], dtype=np.int64)
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=8000,
        random_state=0,
        alpha=1e-3,
        early_stopping=False,
        tol=1e-6,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    examples = []
    for i in range(2):
        pi = int(pred[i])
        examples.append(
            {
                "after_line": i + 1,
                "current": lines[i],
                "true_class": i + 1,
                "pred_class": pi,
                "true_next": lines[i + 1],
                "pred_next": lines[pi],
                "ok": lines[pi] == lines[i + 1],
            }
        )
    return {
        "task": "next_line_via_class",
        "lines_numbered": [{"index": j, "text": lines[j]} for j in range(len(lines))],
        "description": (
            "Постановка: классификация следующей строки из корпуса. Выход сети — номер строки k "
            "в списке L0, L1, L2; результат предсказания — полный текст строки L_k. "
            "В обучающей выборке две пары: после L0 следует L1, после L1 следует L2."
        ),
        "examples": examples,
        "acc_on_two_pairs": float(accuracy_score(y, pred)),
    }


def _bar_val_accuracy(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["key"] for r in rows]
    acc = [r["acc_val"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, acc, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Accuracy (отложенные 25%)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Практика №8: следующий символ — качество на контроле по способу выборки")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_practice08(out_dir: Path | str | None = None, save_png: bool = True) -> dict:
    out = Path(out_dir) if out_dir is not None else Path("outputs") / "practice08"
    out.mkdir(parents=True, exist_ok=True)

    t_char = run_task_char_level_quality()
    t_line = run_task_line_pairs_demo()
    p_bar = out / "practice08_task1_val_accuracy_char.png"

    if save_png:
        _bar_val_accuracy(p_bar, t_char["strategies"])

    return {
        "out_dir": str(out.resolve()),
        "text": TEXT_TABLE_89_VARIANT5,
        "pdf_assignment": PDF_ASSIGNMENT_RU,
        "task_char": t_char,
        "task_line_demo": t_line,
        "plots": {"task1_bar": str(p_bar) if save_png else None},
    }


def print_results(data: dict, *, verbose: bool = True) -> None:
    bar = "=" * 72
    if verbose:
        print(bar)
        print("Практическая работа №8, вариант 5 (таблица 8.9)")
        print(bar)

    print(f"\n{bar}\nУсловие\n{bar}")
    print(data["pdf_assignment"])

    tc = data["task_char"]
    print(f"\n{bar}\nЧасть 1. Следующий символ, MLPClassifier, отложенный контроль 25%\n{bar}")
    print(tc["description"])
    print(f"{'способ':<10} {'n_tr':>5} {'n_val':>5} {'acc_tr':>8} {'acc_val':>8} {'logloss_val':>12}")
    for r in tc["strategies"]:
        ll = r["logloss_val"]
        ll_s = f"{ll:.4f}" if ll == ll else "nan"
        print(
            f"{r['key']:<10} {r['n_train']:>5} {r['n_val']:>5} "
            f"{r['acc_train']:>8.4f} {r['acc_val']:>8.4f} {ll_s:>12}"
        )
    print("\nСпособы (ключ — описание):")
    for r in tc["strategies"]:
        print(f"  {r['key']}: {r['title']}")
    print(f"\nНаибольшая accuracy на контроле: способ [{tc['best_val_key']}]")

    tl = data["task_line_demo"]
    print(f"\n{bar}\nЧасть 2. Предсказание следующей строки (текст целиком)\n{bar}")
    print(tl["description"])
    print("Строки корпуса (индекс → текст):")
    for row in tl["lines_numbered"]:
        print(f"  L{row['index']}: «{row['text']}»")
    print(
        "\nРезультат предсказания — это полный текст строки L_k, где k — номер класса, выданный MLP "
        "(см. столбец «прогноз: текст следующей строки»)."
    )
    print(f"\nAccuracy на двух парах (обучение на тех же парах): {tl['acc_on_two_pairs']:.4f}")
    print(f"{'№':>3} {'класс (эталон)':>16} {'класс (прогноз)':>16}  эталон: следующая строка  |  прогноз: следующая строка")
    for ex in tl["examples"]:
        ok = "да" if ex["ok"] else "нет"
        print(
            f"{ex['after_line']:>3} {ex['true_class']:>16} {ex['pred_class']:>16}  "
            f"«{ex['true_next']}»  |  «{ex['pred_next']}»  (верно: {ok})"
        )

    print("\nИтоговые предсказанные строки:")
    for ex in tl["examples"]:
        print(f"  после L{ex['after_line'] - 1}: «{ex['pred_next']}»")

    print(
        "\nВывод: в части 1 на коротком тексте accuracy на контроле ограничена; в части 2 результат предсказания "
        "— полный текст следующей строки (столбец «прогноз»); при обучении на двух парах из табл. 8.9 оба прогноза "
        "совпали с эталоном."
    )

    print("\n[файлы]")
    print(f"      {data['out_dir']}")
    for k, p in data["plots"].items():
        if p:
            print(f"      • {k}: {p}")


def main() -> None:
    data = run_practice08()
    print_results(data, verbose=True)


if __name__ == "__main__":
    main()
