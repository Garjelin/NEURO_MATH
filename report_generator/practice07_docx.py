"""DOCX для практической работы №7 (вариант 5). Структура: задание → этапы (как в практике №1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def _p(doc: Document, text: str) -> None:
    doc.add_paragraph(text)


def _add_figure(doc: Document, image_path: Path | None, caption: str, width_inches: float = 4.2) -> None:
    if image_path is not None and image_path.is_file():
        doc.add_picture(str(image_path), width=Inches(width_inches))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True


def _add_console_placeholder(doc: Document, title: str, instruction: str) -> None:
    doc.add_heading(title, level=3)
    _p(doc, instruction)
    doc.add_paragraph()
    doc.add_paragraph()


def _matrix_paragraph_and_table(doc: Document, label: str, mat: np.ndarray, fmt: str = "{:.4g}") -> None:
    _p(doc, label)
    table = doc.add_table(rows=int(mat.shape[0]), cols=int(mat.shape[1]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            table.rows[i].cells[j].text = fmt.format(mat[i, j])


def build_practice07_docx(
    report_path: Path,
    data: dict,
    plot_task1_patterns: Path | None = None,
    plot_task2_image: Path | None = None,
    plot_task2_kernel: Path | None = None,
    plot_task2_output: Path | None = None,
    plot_task3_E: Path | None = None,
    plot_task3_scatter: Path | None = None,
    source_file_hint: str = "neuro_math_Yakimov_var5/task_7/practice07.py",
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    t1 = data["task1"]
    t2 = data["task2"]
    t3 = data["task3"]

    doc.add_heading("Практическая работа №7. Вариант 5", level=1)
    _p(
        doc,
        "Работа выполняется в три задания по полному тексту PDF «Задание к практическим работам 7»: "
        "регрессия Y по бинарным изображениям (таблица 7.7), деконволюция по таблице 7.8 и разбор "
        "triplet loss по таблице 7.9. В выданном PDF нет рисунка 7.42 из основного пособия; схема "
        "embedding в задании 3 задана явно как четыре обучаемых вектора для объектов A1, A2, B1, B2.",
    )
    _p(
        doc,
        "Изложение внутри каждого задания разбито на этапы по той же логике, что и в практической работе №1: "
        "подготовка и формализация; выбор модели или оператора; выполнение расчёта; анализ результатов "
        "и иллюстрации; краткие выводы по заданию.",
    )

    # ===================== Задание 1 =====================
    doc.add_heading("Задание 1. Таблица 7.7 — регрессия Y по изображению", level=2)

    doc.add_heading("1-й этап: подготовка данных и формализация задачи", level=3)
    _p(
        doc,
        "По условию для каждого объекта P1, P2, P3 задано бинарное изображение (матрица 0/1) и значение Y. "
        "Требуется построить модель зависимости Y от пиксельного шаблона и оценить качество прогноза.",
    )
    _p(doc, t1["source"] + " Для варианта 5 используются три образца размера 2×3.")

    doc.add_heading("2-й этап: выбор модели и представление признаков", level=3)
    _p(
        doc,
        "В качестве аппроксиматора выбран многослойный персептрон `MLPRegressor` (scikit-learn): "
        "вход — вектор из шести значений пикселей после построчного развёртывания матрицы 2×3.",
    )
    _p(
        doc,
        "Признаки стандартизуются через `StandardScaler` по обучающей подвыборке в каждом эксперименте.",
    )

    doc.add_heading("3-й этап: обучение и контрольный эксперимент", level=3)
    _p(
        doc,
        "Поскольку в таблице всего три объекта, для оценки обобщения применяется схема leave-one-out: "
        "на каждом шаге два объекта участвуют в обучении, третий — удерживается для прогноза.",
    )
    _p(
        doc,
        f"Истинные Y по таблице: {np.array2string(t1['y_table'], precision=6)}. "
        f"Прогнозы LOOCV: {np.array2string(t1['loocv_pred'], precision=6)}. "
        f"LOOCV MSE = {t1['loocv_mse']:.6g}, MAE = {t1['loocv_mae']:.6g}.",
    )
    _p(
        doc,
        f"Дополнительно: обучение на всех трёх объектах сразу даёт MSE = {t1['fit_all_mse']:.6g}, "
        f"MAE = {t1['fit_all_mae']:.6g}, R² = {t1['fit_all_r2']:.6g} (на столь малой выборке R² "
        "интерпретировать осторожно).",
    )

    doc.add_heading("4-й этап: численные данные и иллюстрация образцов", level=3)
    for nm, im, y in zip(t1["pattern_names"], t1["images_2x3"], t1["y_table"], strict=True):
        _matrix_paragraph_and_table(doc, f"{nm}, Y = {y:g}:", im, fmt="{:.0f}")
        doc.add_paragraph()
    _add_figure(doc, plot_task1_patterns, "Рисунок 1 — три образца из таблицы 7.7 (вариант 5).")

    doc.add_heading("5-й этап: выводы по заданию 1", level=3)
    _p(
        doc,
        "Построена регрессионная модель, сопоставляющая бинарное изображение 2×3 с числом Y из таблицы 7.7. "
        "Продемонстрированы прогнозы в режиме leave-one-out и подгонка на полной тройке объектов.",
    )

    # ===================== Задание 2 =====================
    doc.add_heading("Задание 2. Таблица 7.8 — деконволюция размером 6×6", level=2)

    doc.add_heading("1-й этап: постановка и условия операции", level=3)
    _p(
        doc,
        "Заданы двумерное изображение и фильтр (ядро). Шаг по обеим осям stride_x = stride_y = 1, "
        "дополнение padding = 0. Требуется выполнить операцию деконволюции (транспонированной свёртки) "
        "и получить карту размера 6×6.",
    )

    doc.add_heading("2-й этап: фиксация исходных численных данных (вариант 5)", level=3)
    _p(doc, "Матрица изображения 4×4 и матрица фильтра 3×3 согласно таблице 7.8 для варианта 5:")
    _matrix_paragraph_and_table(doc, "Изображение 4×4:", t2["image"])
    doc.add_paragraph()
    _matrix_paragraph_and_table(doc, "Фильтр 3×3:", t2["kernel"])

    doc.add_heading("3-й этап: выполнение деконволюции", level=3)
    _p(
        doc,
        "Реализована транспонированная свёртка в согласовании с оператором `ConvTranspose2d` при "
        "stride 1 и padding 0: каждый элемент входной карты добавляет в выход вклад, пропорциональную ядру.",
    )
    _matrix_paragraph_and_table(doc, "Результирующая карта 6×6:", t2["output_6x6"], fmt="{:.6g}")

    doc.add_heading("4-й этап: графическая иллюстрация", level=3)
    _add_figure(doc, plot_task2_image, "Рисунок 2 — изображение 4×4 (задание 2).")
    _add_figure(doc, plot_task2_kernel, "Рисунок 3 — фильтр 3×3 (задание 2).")
    _add_figure(doc, plot_task2_output, "Рисунок 4 — результат деконволюции 6×6 (задание 2).")

    doc.add_heading("5-й этап: выводы по заданию 2", level=3)
    _p(
        doc,
        "Получена выходная карта заданного размера 6×6; значения согласованы с дискретной формулой "
        "транспонированной свёртки при указанных stride и padding.",
    )

    # ===================== Задание 3 =====================
    doc.add_heading("Задание 3. Таблица 7.9 и суммарный triplet loss", level=2)

    doc.add_heading("1-й этап: постановка по тексту методички", level=3)
    _p(
        doc,
        "После таблицы 7.9 в задании требуется перечислить все допустимые тройки (Anchor, Positive, Negative): "
        "положительный объект того же класса, что и якорь, отрицательный — из другого класса. "
        "Для каждой тройки записывается triplet loss; сумма по всем тройкам — итоговая функция потерь для минимизации.",
    )
    _p(
        doc,
        "Класс A: объекты A1 и A2; класс B: B1 и B2. В программе используется стандартная форма с квадратами "
        "евклидовых расстояний и полем margin m: для тройки (a, p, n) с эмбеддингами e_a, e_p, e_n "
        "слагаемое L = max(0, ||e_a − e_p||² − ||e_a − e_n||² + m). Сумма L по всем допустимым тройкам "
        "минимизируется градиентным спуском по матрице эмбеддингов размера 4×d.",
    )
    _p(doc, t3["source_table"])

    doc.add_heading("2-й этап: данные таблицы 7.9 (вариант 5)", level=3)
    _p(doc, "Столбцы X и Y для четырёх объектов:")
    tbl = doc.add_table(rows=5, cols=3)
    hdr = tbl.rows[0].cells
    hdr[0].text = "Объект"
    hdr[1].text = "X"
    hdr[2].text = "Y"
    for i, name in enumerate(t3["object_names"]):
        row = tbl.rows[i + 1].cells
        row[0].text = name
        row[1].text = f"{float(t3['table_X'][i]):g}"
        row[2].text = f"{float(t3['table_Y'][i]):g}"

    doc.add_heading("3-й этап: перечень троек и значения потерь", level=3)
    _p(
        doc,
        f"Размерность эмбеддинга d = {t3['embed_dim']}, margin m = {t3['margin']}. "
        f"Число троек: {t3['triplets_count']}.",
    )
    trip_lines = ", ".join(f"({a}, {p}, {n})" for a, p, n in t3["triplets"])
    _p(doc, "Допустимые тройки (якорь, положительный, отрицательный): " + trip_lines + ".")
    _p(
        doc,
        f"Сумма triplet loss до минимизации (случайная инициализация эмбеддингов): {t3['loss_sum_initial']:.6g}. "
        f"После градиентного спуска: {t3['loss_sum_final']:.6g}.",
    )

    doc.add_heading("4-й этап: визуализация эмбеддингов", level=3)
    _add_figure(doc, plot_task3_E, "Рисунок 5 — матрица эмбеддингов 4×d после обучения (задание 3).")
    _add_figure(doc, plot_task3_scatter, "Рисунок 6 — объекты на плоскости первых двух координат (задание 3).")

    doc.add_heading("5-й этап: выводы по заданию 3", level=3)
    _p(
        doc,
        "Выполнено перечисление всех троек для двух классов по два объекта, задана суммарная triplet loss "
        "и численно показано её уменьшение при подборе эмбеддингов. Столбцы X, Y таблицы 7.9 зафиксированы "
        "в отчёте; сама постановка triplet loss опирается на разделение объектов на классы A и B.",
    )

    _add_console_placeholder(
        doc,
        "Рисунок 7 — Скрин консоли (вставить вручную)",
        "Вставьте скрин терминала после запуска рабочего скрипта с блоками [1/3]–[3/3], "
        "чтобы были видны метрики задания 1 и матрицы задания 2.",
    )

    doc.add_heading("Общий вывод по практической работе №7", level=2)
    _p(
        doc,
        "Выполнены три задания варианта 5 по данным из полного PDF: регрессия Y по трём образам из таблицы 7.7, "
        "деконволюция по таблице 7.8 с получением карты 6×6, а также формулировка и минимизация суммарного "
        "triplet loss для объектов из таблицы 7.9.",
    )

    doc.add_heading("Ссылка на исходный код", level=2)
    _p(doc, f"Рабочий скрипт: `{source_file_hint}`. Генератор отчёта: `report_generator/practice07_docx.py`.")

    doc.save(report_path)
