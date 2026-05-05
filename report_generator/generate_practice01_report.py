#!/usr/bin/env python3
"""
Генерация DOCX-отчёта для практики №1 (отдельно от рабочего решения).

Запуск из корня репозитория (чтобы работали импорты):
  python report_generator/generate_practice01_report.py

Скрипт только пересчитывает числа/графики при необходимости и пишет .docx.
Он не является частью сдаваемого решения: в отчёт по заданию идёт
assignments/practice01.py и результаты его запуска в консоли.

Папку report_generator к отчёту обычно не прикладывают.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Корень репозитория в sys.path при запуске как скрипта
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assignments.practice01 import run_practice01  # noqa: E402
from report_generator.practice01_docx import build_practice01_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №1")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "practice01",
        help="Каталог для PNG и отчёта",
    )
    parser.add_argument(
        "--docx-name",
        type=str,
        default="Отчет_практическая_1_вариант_5.docx",
        help="Имя файла отчёта внутри out-dir",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не пересоздавать PNG (только docx из текущих данных)",
    )
    args = parser.parse_args()

    data = run_practice01(out_dir=args.out_dir, save_png=not args.no_plots)
    add = data["additional"]
    plot_a = args.out_dir / "task_A_convergence.png"
    plot_b = args.out_dir / "task_B_convergence.png"

    docx_path = args.out_dir / args.docx_name
    build_practice01_docx(
        docx_path,
        data["task_a"],
        data["task_b"],
        add["w"],
        float(add["h"]),
        add["preds"],
        plot_a=plot_a if plot_a.exists() else None,
        plot_b=plot_b if plot_b.exists() else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
