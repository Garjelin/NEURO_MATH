# NEURO_MATH

Репозиторий с решениями практических работ по нейроматематике. Планируется **9 практик**; каждая — отдельный модуль в каталоге `assignments/`.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `assignments/practice01.py` | Практическая работа №1 (вариант 5): минимизация, доп. задание |
| `assignments/practice02.py` … | Будущие работы (добавляются по мере выполнения) |
| `report_generator/` | Скрипты и шаблоны **только для генерации DOCX**; к отчёту по заданию обычно **не** прикладывают — в приложение идёт код из `assignments/` |
| `outputs/` | Результаты запусков (графики, при необходимости отчёты) — не коммитьте большие бинарники, если не нужно |
| `main.py` | Краткая справка в консоли |
| `requirements.txt` | Зависимости Python |

## Окружение и установка

Требуется Python 3.10+ (рекомендуется 3.11+).

Из корня репозитория:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Matplotlib и права на каталог конфигурации

Если при импорте `matplotlib` появляется предупреждение про `MPLCONFIGDIR` или `Permission denied` для `~/.config/matplotlib`, задайте каталог внутри проекта:

```bash
export MPLCONFIGDIR="$(pwd)/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

(На Windows в PowerShell: `$env:MPLCONFIGDIR = "$PWD\.mplconfig"; New-Item -ItemType Directory -Force $env:MPLCONFIGDIR`)

## Как запустить практическую работу и получить результаты

Все команды ниже выполняйте **из корня репозитория** (`NEURO_MATH/`), чтобы корректно работали пакетные импорты `assignments.*`.

### Практика №1 (`assignments/practice01.py`)

**Что делает скрипт**

- Задание **А**: минимум \(f(x) = x^3 - 3\sin x\), старт \(x_0 = 0\) (интервал из методички \([0;1]\)).
- Задание **Б**: минимум \(f(x_1,x_2) = 2x_1^2 + 2x_2^2 + 2x_1x_2 - 14x_1 - 12x_2 + 29\), старт \((-1, 4)\).
- Методы: градиентный спуск, покоординатный спуск, адаптивный градиентный спуск (Adagrad).
- **Доп. задание**: персептрон, подбор `w1`, `w2`, `h` по таблице (вес, рост) → пол.

**Запуск**

```bash
python -m assignments.practice01
```

**Что появится после запуска**

1. **Подробный вывод в консоль** — блоки `[1/4]`…`[4/4]`: задания А и Б (для каждого метода — `x*`, `f(x*)`, норма градиента в конце, число шагов), дополнительное задание, абсолютный путь к каталогу с PNG и напоминание, что DOCX создаётся **отдельным** скриптом. Этот вывод удобно снимать скриншотом для вставки в отчёт (в сгенерированном DOCX указано место «Рисунок 3»).
2. **Каталог артефактов** (по умолчанию): `outputs/practice01/`
   - `task_A_convergence.png` — сходимость по заданию А
   - `task_B_convergence.png` — сходимость по заданию Б

**Свой каталог для графиков** (из кода):

```python
from pathlib import Path
from assignments.practice01 import run_practice01, print_results

data = run_practice01(out_dir=Path("outputs/my_run"), save_png=True)
print_results(data)
```

**Персональные данные для строки A4 (доп. задание)**

В `assignments/practice01.py` используйте функцию `run_additional_task(samples=..., labels=...)`: передайте матрицу `Nx2` [вес кг, рост см] и вектор меток `0/1`.

## Как сгенерировать DOCX-отчёт

Файл `assignments/practice01.py` **не** создаёт отчёт — только считает и печатает в консоль (и при необходимости сохраняет PNG). DOCX собирается вручную отдельной командой, когда нужно.

Папка `report_generator/` предназначена **только для автоматической сборки отчёта**; её можно не включать в сдаваемый архив с кодом задания.

Содержание отчёта по практике №1 оформлено **развёрнуто** (несколько страниц): пять этапов по аналогии с методичкой, пояснения, таблицы результатов, в документ **встраиваются** графики сходимости (если PNG уже построены).

Из корня репозитория:

```bash
python report_generator/generate_practice01_report.py
```

По умолчанию:

- пересчитывается практика №1 и строятся PNG в `outputs/practice01/`;
- создаётся файл `outputs/practice01/Отчет_практическая_1_вариант_5.docx`.

Опции:

```bash
python report_generator/generate_practice01_report.py --out-dir outputs/practice01 --docx-name отчет.docx
```

Если графики уже есть и нужен только DOCX без перерисовки:

```bash
python report_generator/generate_practice01_report.py --no-plots
```

(При `--no-plots` в документ попадут пути к файлам графиков, если PNG уже лежат в `out-dir`.)

## Добавление практик 2–9

1. Создайте файл `assignments/practiceNN.py` с функцией `run_practiceNN()` и блоком `if __name__ == "__main__":`.
2. Запуск: `python -m assignments.practiceNN`.
3. При необходимости добавьте `report_generator/generate_practiceNN_report.py` по аналогии с практикой 1.

## Корневая справка

```bash
python main.py
```
