# Python PDE Solvers

Набор учебных и исследовательских солверов для уравнений в частных производных.
Проект организован как библиотека: вычислительная логика лежит в `src/`, запуск идет через
единый CLI-контракт, проверки качества зафиксированы в тестах и CI.

## Что здесь решается

- Cauchy-задача для уравнения второго порядка.
- Линейный и нелинейный перенос (включая анализ численной вязкости).
- Явные и неявные схемы для уравнения теплопроводности.
- Sigma-семейство схем и схема с пересчетом.
- Векторная система переноса.
- Связанная параболическая задача (implicit + Fourier + нелинейная диффузия).

Если кратко: это полигон для сравнения численных схем и проверки их поведения на
одинаковом инженерном каркасе.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Как запускать

Все entrypoints поддерживают одинаковые флаги:

- `--no-plot` — headless запуск (удобно для CI и удаленных машин).
- `--save-dir <path>` — сохранить построенные фигуры в директорию.

Примеры:

```bash
run-cauchy --no-plot
run-implicit-heat --no-plot --save-dir out/implicit
run-vector-transport --save-dir out/vector
```

Также можно запускать через `scripts/` (они проксируют в тот же CLI):

```bash
python scripts/cauchy.py --no-plot
python scripts/coupled_parabolic.py --save-dir out/coupled
```

## Набор команд

Основные команды, объявленные в `pyproject.toml`:

- `run-cauchy`
- `run-advection-diffusion`
- `run-linear-advection`
- `run-numerical-viscosity`
- `run-lax-wendroff`
- `run-vector-transport`
- `run-implicit-heat`
- `run-burgers-diffusion`
- `run-sigma-heat`
- `run-recalculation-heat`
- `run-wave-task12`
- `run-wave-task13`
- `run-coupled-parabolic`

## Проверки качества

Локально:

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -q
```

В CI:

- обязательный job: линт + тесты;
- отдельный perf-signal job (не блокирует merge), который публикует артефакт
  с замером TDMA против dense baseline.

## Структура репозитория

```text
python-tasks/
├── src/                # solvers, схемы, конфиги, визуализация
├── scripts/            # thin wrappers для локального запуска
├── tests/              # regression + invariant + convergence checks
├── .github/workflows/  # CI пайплайн
├── pyproject.toml      # package metadata + project scripts
└── requirements.txt
```

## Технические принципы проекта

- **Единый runtime-контракт**: каждый solver запускается одинаково (`show_plots`, `save_dir`).
- **Fail-fast конфиги**: базовые параметры валидируются на уровне `config.py`.
- **Повторное использование кода**: общие утилиты вынесены в `solver_utils`.
- **Тесты на поведение схем**: не только smoke, но и инварианты/сходимость.

## Ограничения и roadmap

- Это в первую очередь учебно-исследовательский код, не production-сервис.
- Для тяжелых профилей стоит добавить отдельный benchmark-suite с хранением истории.
- Для больших задач можно вынести единый формат отчета по экспериментам (metrics + plots).
