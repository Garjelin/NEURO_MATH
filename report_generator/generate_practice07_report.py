#!/usr/bin/env python3
"""
Генерация DOCX для практики №7 (вариант 5).

  python report_generator/generate_practice07_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_7.practice07 import run_practice07  # noqa: E402
from report_generator.practice07_docx import build_practice07_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №7")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice07")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_7_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice07(out_dir=args.out_dir, save_png=not args.no_plots)
    od = args.out_dir
    docx_path = od / args.docx_name

    build_practice07_docx(
        docx_path,
        data,
        plot_task1_patterns=od / "practice07_task1_patterns.png"
        if (od / "practice07_task1_patterns.png").exists()
        else None,
        plot_task2_image=od / "practice07_task2_image.png" if (od / "practice07_task2_image.png").exists() else None,
        plot_task2_kernel=od / "practice07_task2_kernel.png" if (od / "practice07_task2_kernel.png").exists() else None,
        plot_task2_output=od / "practice07_task2_deconv_6x6.png" if (od / "practice07_task2_deconv_6x6.png").exists() else None,
        plot_task3_E=od / "practice07_task3_embeddings_heatmap.png"
        if (od / "practice07_task3_embeddings_heatmap.png").exists()
        else None,
        plot_task3_scatter=od / "practice07_task3_embeddings_scatter.png"
        if (od / "practice07_task3_embeddings_scatter.png").exists()
        else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
