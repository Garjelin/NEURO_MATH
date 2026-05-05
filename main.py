"""
Корневая точка входа: краткая справка по структуре репозитория.

Решения практик — в каталоге assignments/ (по одному файлу на работу).
Генерация DOCX — в report_generator/ (к отчёту обычно не прикладывают).

Подробности см. README.md.
"""

from __future__ import annotations


def main() -> None:
    print(
        "NEURO_MATH\n"
        "\n"
        "Практика №1 (вариант 5):\n"
        "  python -m assignments.practice01\n"
        "\n"
        "Сгенерировать DOCX отдельно (не часть сдаваемого кода задания):\n"
        "  python report_generator/generate_practice01_report.py\n"
        "\n"
        "Полная инструкция: README.md\n"
    )


if __name__ == "__main__":
    main()
