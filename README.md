# PsyAdapt — v1.0

**PsyAdapt** — це open-source Python-бібліотека для оцінки **надійності психодіагностичних методик** на основі даних у форматах CSV та Excel.

Проєкт орієнтований на практичних психологів, дослідників та студентів, які працюють з психометричними даними та потребують коректного, прозорого і відтворюваного аналізу.

---

## Основні можливості

- Оцінка внутрішньої узгодженості тестів (Cronbach’s alpha)
- Аналіз міжекспертної узгодженості:
  - Cohen’s kappa
  - Fleiss’ kappa
  - Krippendorff’s alpha
  - Intraclass Correlation Coefficient (ICC)
- Підтримка номінальних, порядкових та інтервальних шкал
- Генерація структурованих JSON-звітів
- Генерація зрозумілих Excel-звітів для психологів
- Опціональна **AI-інтерпретація** результатів (простою мовою) — *вмикається тільки вручну*
- Робота з CSV та Excel-файлами без потреби у SPSS або R

---

## Реалізовані методи

| Метод | Призначення |
|------|-------------|
| Cronbach’s α | Внутрішня узгодженість одного тесту |
| Cohen’s κ | Два експерти або ретест |
| Fleiss’ κ | Три і більше експертів |
| Krippendorff’s α | Універсальний метод з підтримкою пропусків |
| ICC | Кількісні міжекспертні оцінки |

---

## Формати даних

### CSV (матричний формат)

- один рядок — один респондент
- одна колонка — один пункт тесту
- значення — числові бали

### Excel (вертикальний формат)

- кожен аркуш — окремий респондент
- колонка A — питання
- колонка B — бал
- підтримується автоматичне перетворення у матричний формат

---

## Excel-звіти

PsyAdapt автоматично створює Excel-файли, придатні для безпосереднього використання психологами.

У звітах присутні:
- загальний показник надійності
- таблиця діагностики пунктів (CITC, alpha if deleted)
- симулятор очищення тесту:
  - після реверсу пунктів
  - після видалення пунктів
  - після реверсу та видалення
- текстові пояснення змін
- умовне форматування для швидкої інтерпретації результатів

---

## Використання через CLI

### Один CSV — Cronbach Alpha

```bash
python -m psyadapt data/test.csv --pretty
```

### Декілька CSV — міжекспертна узгодженість

```bash
python -m psyadapt rater1.csv rater2.csv rater3.csv   --primary krippendorff   --out report.json
```

### AI-інтерпретація через CLI (опціонально)

AI **не вмикається за замовчуванням**. Щоб додати коротке пояснення результатів простими словами:

```bash
# варіант 1: ключ через змінну середовища
export PSYADAPT_OPENAI_API_KEY="sk-..."
python -m psyadapt data/leary/vertical_workbook_MATRIX.csv --ai --ai-model gpt-4.1-mini --pretty

# варіант 2: ключ прямо параметром (для швидкого тесту)
python -m psyadapt data/leary/vertical_workbook_MATRIX.csv --ai --ai-key "sk-..." --ai-model gpt-4.1-mini --pretty
```

---

## Використання через Python API

```python
from psyadapt import compare_reliability_csv

report = compare_reliability_csv(
    ["rater1.csv", "rater2.csv"],
    primary="cohen"
)

print(report["summary"])
```

### AI-інтерпретація через Python API (опціонально)

```python
from psyadapt import compare_reliability_csv, compare_reliability_to_excel

# 1) JSON report + AI
report = compare_reliability_csv(
    ["rater1.csv", "rater2.csv"],
    primary="cohen",
    ai=True,
    ai_model="gpt-4.1-mini",
    # ai_api_key можна не передавати, якщо задано OPENAI_API_KEY або PSYADAPT_OPENAI_API_KEY
)

# 2) Excel report + AI
compare_reliability_to_excel(
    ["rater1.csv", "rater2.csv"],
    out_path="report.xlsx",
    primary="cohen",
    ai=True,
    ai_model="gpt-4.1-mini",
)
```

Якщо AI успішно згенерував текст, у JSON зʼявиться ключ `report["ai"]`,
а в Excel додасться аркуш **"AI interpretation"**.

---

## Встановлення

Рекомендований спосіб (для стабільної роботи CLI та тестів):

```bash
pip install -e .
```

---

## Налаштування AI (ключ, приватність, типові помилки)

### Де взяти ключ
AI використовує **OpenAI API**. Потрібен API key з OpenAI Platform.

### Як задати ключ
Підтримуються змінні середовища:
- `PSYADAPT_OPENAI_API_KEY` (пріоритет)
- `OPENAI_API_KEY`

Приклади:

**Windows PowerShell**
```powershell
$env:PSYADAPT_OPENAI_API_KEY = "sk-..."
python -m psyadapt data/test.csv --ai --ai-model gpt-4.1-mini
```

**Windows CMD**
```bat
set PSYADAPT_OPENAI_API_KEY=sk-...
python -m psyadapt data\test.csv --ai --ai-model gpt-4.1-mini
```

**Git Bash / Linux / macOS**
```bash
export PSYADAPT_OPENAI_API_KEY="sk-..."
python -m psyadapt data/test.csv --ai --ai-model gpt-4.1-mini
```

### Приватність
PsyAdapt **не відправляє сирі дані респондентів**. Для AI передаються лише агреговані метрики
(підсумки надійності + коротка діагностика пунктів), щоб згенерувати текстову інтерпретацію.

### Типові симптоми та рішення
- **`AI: missing` / у JSON немає `report["ai"]`** → найчастіше не задано ключ або AI не увімкнено (`--ai` / `ai=True`).
- **HTTP 401** → ключ неправильний/недійсний.
- **HTTP 429 (quota/billing)** → вичерпаний ліміт API або не налаштований білінг в OpenAI Platform.
- **HTTP 404 (model not found)** → модель недоступна твоєму акаунту; спробуй іншу (`--ai-model ...`) і перевір доступні моделі через `/v1/models`.

> Порада: перед дебагом AI перевір доступні моделі:
> `curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

---

## Залежності

- numpy
- pandas
- scipy
- openpyxl

Усі залежності описані у `pyproject.toml`.

---

## Статус релізу

Версія **1.0** є стабільною та містить повний базовий функціонал для:
- психометричного аналізу,
- практичного використання,
- навчальних і наукових цілей.

Подальший розвиток передбачає розширення діагностики, покращення UX-звітів та додаткові методи аналізу.

---

## Ліцензія

Проєкт поширюється під ліцензією **MIT**.  
Дозволене вільне використання, модифікація та розповсюдження з посиланням на автора.

---

## Автори

- Yaroslav Savytskyi
- Polina Golovach
