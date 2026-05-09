# NEURO_MATH

Репозиторий с решениями практических работ по нейроматематике (вариант 5). Код работ размещается в **`neuro_math_Yakimov_var5/task_N/`** (`practiceNN.py`). Планируется **9 практик**.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `neuro_math_Yakimov_var5/task_1/practice01.py` | Практика №1: минимизация, доп. персептрон |
| `neuro_math_Yakimov_var5/task_2/practice02.py` | Практика №2: НС по табл. 2.7/2.8, прогноз «акций» (зад. 2, табл. 2.9) |
| `neuro_math_Yakimov_var5/task_4/practice04.py` | Практика №4: деревья решений (табл. 4.3, 4.5), граф и backprop (табл. 4.7) |
| `report_generator/` | Только генерация DOCX (**не** входит в сдаваемый код задания) |
| `outputs/` | PNG и при желании DOCX после запусков |
| `requirements.txt` | Зависимости (дубликат списка из `neuro_math_Yakimov_var5/requirements.txt`) |
| `main.py` | Краткая справка по командам |

## Установка

Из корня `NEURO_MATH/`:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Matplotlib: каталог конфигурации

При предупреждении про `MPLCONFIGDIR`:

```bash
export MPLCONFIGDIR="$(pwd)/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

Команды ниже выполняйте **из корня репозитория**.

---

## Практика №1 — `neuro_math_Yakimov_var5/task_1/practice01.py`

- Задания А/Б минимизации (градиентный, покоординатный, адаптивный спуск), доп. задание по персептрону.
- **Запуск:** `python -m neuro_math_Yakimov_var5.task_1.practice01`
- **Вывод:** консоль `[1/4]`…`[4/4]`, PNG в `outputs/practice01/`.
- **DOCX (отдельно):** `python report_generator/generate_practice01_report.py`

В отчёте место для скрина консоли — «Рисунок 3» в сгенерированном DOCX.

---

## Практика №2 — `neuro_math_Yakimov_var5/task_2/practice02.py`

**Задание 1** (табл. 2.7, вариант 5 по табл. 2.8 — строки **7, 8, 9, 10**):

- Обучение `MLPRegressor` на точках `(0,1)`, `(1,0)`, `(2,3)`, `(2,-2)`.
- Тест — остальные строки табл. 2.7.
- График: `practice02_task1_fit.png`.

**Задание 2** (табл. 2.9, вариант 5 — **прогноз цен акций**):

- Учебный **синтетический** ряд (цена + объём + признаки доходности/волы/уровня цены); можно заменить реальными котировками (CSV / yfinance).
- Временное разбиение train/test, метрики MSE, MAE, R².
- График: `practice02_task2_forecast.png`.

**Запуск:**

```bash
python -m neuro_math_Yakimov_var5.task_2.practice02
```

**Вывод в консоль:** блоки `[1/3]`…`[3/3]` — выборки, метрики, абсолютные пути к PNG.

Рабочий скрипт **не** создаёт DOCX.

**DOCX (отдельно, по желанию):**

```bash
python report_generator/generate_practice02_report.py
```

Файл по умолчанию: `outputs/practice02/Отчет_практическая_2_вариант_5.docx`. В документе указано, куда вставить **Рисунок 3** — скрин консоли после запуска практики №2.

Опции:

```bash
python report_generator/generate_practice02_report.py --out-dir outputs/practice02 --docx-name отчет.docx
python report_generator/generate_practice02_report.py --no-plots
```

---

## Практика №4 — `neuro_math_Yakimov_var5/task_4/practice04.py`

**Задание 1** (табл. 4.3 / 4.4, вариант 5 — строки **7–10**): `DecisionTreeRegressor`, на логическом тесте **MSE**.

**Задание 2** (табл. 4.5 / 4.6, вариант 5 — строки **2, 3, 5, 8, 9, 10**): дерево по признакам `(x, z)`, на тесте **MAE** (при поддержке sklearn — критерий `absolute_error`).

**Задание 3** (табл. 4.7, вариант 5): \(f(x,y)=e^{xy}-\cos(x/y)+1/\sin(x+y)\) — узлы графа, прямой проход и ручной обратный проход; сверка градиента с конечными разностями.

**Запуск:**

```bash
python -m neuro_math_Yakimov_var5.task_4.practice04
```

PNG по умолчанию: `outputs/practice04/` (относительно текущей директории).

**DOCX:**

```bash
python report_generator/generate_practice04_report.py
```

Файл по умолчанию: `outputs/practice04/Отчет_практическая_4_вариант_5.docx`.

---

## Программный вызов из своего кода

```python
from pathlib import Path
from neuro_math_Yakimov_var5.task_2.practice02 import run_practice02, print_results

data = run_practice02(out_dir=Path("outputs/practice02"), save_png=True)
print_results(data, verbose=True)
```

---

## Корневая справка

```bash
python main.py
```
