# Приклади (examples)

Ця папка містить невеликі, штучно згенеровані дані лише для демонстрації формату.

## 1) Матриця (1 CSV = 1 тест)
- `leary_interpersonal.csv` — рядок = респондент, колонка = пункт/шкала, значення = бал.

Приклад запуску:
```bash
python -m psyadapt examples/leary_interpersonal.csv --primary cronbach --out report.xlsx
```

## 2) Псевдо-ретест / експерти (2+ CSV)
- `leary_t1.csv`, `leary_t2.csv`, `leary_t3.csv` — однакова структура колонок; різні "оцінювачі"/часи.

Приклад:
```bash
python -m psyadapt examples/leary_t1.csv examples/leary_t2.csv --primary cohen --out report.xlsx
python -m psyadapt examples/leary_t1.csv examples/leary_t2.csv examples/leary_t3.csv --primary fleiss --out report.xlsx
python -m psyadapt examples/leary_t1.csv examples/leary_t2.csv examples/leary_t3.csv --primary krippendorff --out report.xlsx
```

## 3) Вертикальний Excel (кожен лист = респондент)
- `vertical_workbook_example.xlsx` — на кожному аркуші: колонка A = питання, колонка B = бал. Дані починаються з рядка 10.

Перетворення у матрицю:
```python
from psyadapt.io.excel import workbook_vertical_to_matrix

df, questions = workbook_vertical_to_matrix(
    "examples/vertical_workbook_example.xlsx",
    start_row=10,
    question_col="A",
    score_col="B",
)
print(df.shape)
```
