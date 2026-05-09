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
