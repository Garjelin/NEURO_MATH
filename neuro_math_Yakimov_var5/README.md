# Нейроматематика
## Выполнил: Сергей Якимов
## Группа: НТм(до)з-25-1

Репозиторий с решениями практических работ по нейроматематике. Каждая работа размещена в отдельном модуле в каталогах `task_1/`, `task_2/` и т.д.

## Окружение и установка

Требуется Python 3.10+ (рекомендуется 3.11+).

Из корня репозитория:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Практическая работа №1 (`task_1/practice01.py`)

**Запуск**

```bash
python -m task_1.practice01
```

### Практическая работа №2 (`task_2/practice02.py`)

**Запуск**

```bash
python -m task_2.practice02
```

### Практическая работа №3 (`task_3/practice03.py`)

Тема по варианту 5: **прогноз стоимости акций**.  
Выполняется нормализация данных двумя способами: **Z-score** и **Min-Max** с последующим сравнением метрик.

**Запуск**

```bash
python -m task_3.practice03
```

**Отчёт (отдельно)**

```bash
python ../report_generator/generate_practice03_report.py --out-dir outputs/practice03
```

### Практическая работа №4 (`task_4/practice04.py`)

**Задание 1** — дерево решений по табл. 4.3 (вариант 5: строки **7–10**), на тесте метрика **MSE**.  
**Задание 2** — дерево по двум бинарным признакам (табл. 4.5), обучение по строкам **2, 3, 5, 8, 9, 10**, на тесте **MAE**.  
**Задание 3** — граф для \(f(x,y)=e^{x^{y}}-\cos(x/y)+1/\sin(x+y)\), прямой и обратный проход.

**Запуск**

```bash
python -m task_4.practice04
```

**Отчёт (отдельно)** — из корня репозитория `NEURO_MATH`:

```bash
python report_generator/generate_practice04_report.py --out-dir outputs/practice04
```

### Практическая работа №5 (`task_5/practice05.py`)

**Задание 1** — таблица **5.16**: классификация «ремонт двигателя» (да/нет) по температуре и вибрации.  
**Задание 2** — тема варианта 5 из таблицы **5.21**: учебная **метеоклассификация** (осадки \(t \to t+1\)) на синтетическом ряду.

**Запуск**

```bash
python -m task_5.practice05
```

**Отчёт** — из корня репозитория `NEURO_MATH`:

```bash
python report_generator/generate_practice05_report.py --out-dir outputs/practice05
```

### Практическая работа №6 (`task_6/practice06.py`)

Та же постановка, что **практика 2, задание 2** (прогноз цены); сравнение **dropout** при **p = 0.1, 0.2 и 0.5**. Обучение на **NumPy** (`MLPRegressor` в sklearn не содержит dropout).

**Запуск**

```bash
python -m task_6.practice06
```

**Отчёт** — из корня репозитория `NEURO_MATH`:

```bash
python report_generator/generate_practice06_report.py --out-dir outputs/practice06
```

### Практическая работа №7 (`task_7/practice07.py`)

Табл. **7.7** (вар. 5, регрессия Y по 2×3), **7.8** (деконволюция 6×6), **7.9** (triplet loss; рис. 7.42 в PDF отсутствует).

**Запуск**

```bash
python -m task_7.practice07
```

**Отчёт** — из корня `NEURO_MATH`:

```bash
python report_generator/generate_practice07_report.py --out-dir outputs/practice07
```

### Практическая работа №8 (`task_8/practice08.py`)

Табл. **8.9** (текст вар. 5): символьная модель с отложенным контролем; постановка «следующая строка» на двух парах.

**Запуск**

```bash
python -m task_8.practice08
```

**Отчёт** — из корня `NEURO_MATH`:

```bash
python report_generator/generate_practice08_report.py --out-dir outputs/practice08
```

### Практическая работа №9 (`task_9/practice09.py`)

Табл. **9.9** (вариант 5): **анализ стиля**; модель подбирает стиль изображения. Реализация: WikiArt-like учебный набор, признаки изображения, **MLPClassifier**, accuracy и матрица ошибок.

**Запуск**

```bash
python -m task_9.practice09
```

PNG при обычном запуске сохраняются в `task_9/outputs/`.

**Отчёт** — из корня `NEURO_MATH`:

```bash
python report_generator/generate_practice09_report.py --out-dir outputs/practice09
```
