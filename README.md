# Online Shoppers Purchasing Intention

[![CI](https://github.com/vivalerievich/online-shoppers-intention/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vivalerievich/online-shoppers-intention/actions/workflows/ci.yml)

**Задача.** Оценить вероятность покупки к концу визита в интернет-магазин (`Revenue`). В наборе UCI 12 330 сессий, из которых 1 908 завершились покупкой (15,5%).

**Результат.** Дерево решений получило PR-AUC **0,3892** на отложенной выборке. Площадь рассчитана методом трапеций.

**Главное ограничение.** `PageValues` исключён из модели из-за риска утечки. Качество измерено на агрегатах завершённых визитов, поэтому для скоринга во время визита требуется отдельная оценка на данных, доступных в момент прогноза.

## Готовая модель

Скачайте `online-shoppers-intention.joblib`, `sample-input.csv` и `SHA256SUMS.txt` из [релиза](https://github.com/vivalerievich/online-shoppers-intention/releases/latest) и распакуйте архив исходного кода того же релиза. Поместите три файла в корень распакованного проекта.

```bash
shasum -a 256 -c SHA256SUMS.txt
uv sync --locked --no-dev --python 3.14
uv run --locked --no-dev predict-shoppers --model online-shoppers-intention.joblib --data sample-input.csv
```

Релиз содержит дерево решений вместе с обученной предобработкой, подогнанной на 9 864 сессиях обучающей части. Для предсказания нужен CSV с 16 признаками из заголовка синтетического `sample-input.csv`. Вывод — CSV со столбцом `purchase_probability` в порядке входных строк. Он содержит оценки вероятности покупки. Рабочий порог зависит от стоимости ошибок. Сохранённый `joblib` следует загружать из доверенного релиза.

## Запуск

Нужны Python 3.11–3.14 и `uv`. После клонирования запустите команды из корня репозитория.

```bash
uv sync
mkdir -p data
curl --fail --location \
  'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv' \
  --output data/online_shoppers_intention.csv
uv run train-shoppers
uv run train-shoppers --diagnose-page-values
uv run train-shoppers --model-out artifacts/online-shoppers-intention.joblib
```

CSV можно также скачать со страницы набора, распаковать в `data/` или передать свой путь через `--data`. CLI печатает JSON с результатами. Исходные данные исключены из Git. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset)

Источник данных — C. Sakar и Y. Kastro (2018). [DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Тесты используют сгенерированные записи и временные файлы, поэтому после установки зависимостей работают без сети и без CSV UCI.

[Проверка модели из релиза](https://github.com/vivalerievich/online-shoppers-intention/actions/workflows/model-smoke.yml) запускается при публикации релиза и вручную по тегу. Она использует код тега и сравнивает прогноз на синтетическом примере с контрактом в `tests/release/<tag>/`. Перед публикацией нового релиза добавьте туда `schema.json` и `expected-output.json` для его тега.

## Результаты

Стратифицированное разбиение 80/20 с `random_state=42` дало 9 864 обучающие и 2 466 отложенных сессий. Подбор параметров и обучение преобразований выполняются внутри пяти стратифицированных фолдов train. Критерий выбора — средняя площадь под precision–recall кривой (PR-AUC, метод трапеций). У обеих моделей 16 входных признаков.

| Модель | CV PR-AUC на train | PR-AUC на holdout | ROC AUC на holdout |
| --- | ---: | ---: | ---: |
| Логистическая регрессия (`C=0.02`) | 0,3225 | 0,3251 | 0,7342 |
| Дерево решений (`max_depth=2`, `min_samples_leaf=1`) | **0,4210** | 0,3892 | 0,6864 |

Дерево выбрано по CV PR-AUC. При пороге `0.5` дерево обнаружило 0 из 382 покупок на holdout, а логистическая регрессия обнаружила 4 покупки при 6 ложных срабатываниях. `--threshold` позволяет проверить другой заранее заданный порог, а рабочий порог зависит от стоимости ошибок.

## Ограничение `PageValues`

`PageValues` отражает ценность страниц, связанную с транзакциями. Если при расчёте использованы покупки или целевые действия после момента прогноза, признак передаёт модели информацию о будущем. Его доступность до покупки по данным UCI подтвердить нельзя, поэтому он исключён из обучения основных моделей и оценки на holdout. [Описание признака в UCI](https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset)

Опция `--diagnose-page-values` сравнивает варианты с этим признаком и без него на одинаковых фолдах train при фиксированных параметрах моделей. С добавлением `PageValues` средняя CV PR-AUC выросла с 0,3225 до 0,6517 для логистической регрессии и с 0,4210 до 0,6949 для дерева. Эти метрики характеризуют чувствительность к признаку.

CSV содержит агрегаты завершённых сессий. Для оценки прогноза во время визита нужны признаки, зафиксированные в момент скоринга. Счётчики просмотров и длительности следует ограничить этим моментом, а `BounceRates` и `ExitRates` — рассчитывать по доступной на тот момент истории.

## Ноутбук

[Ноутбук](notebooks/online-shoppers-intention.ipynb) содержит EDA, кривые качества и анализ порогов. [Подробные заметки](notebooks/analysis.md) сопровождают разбор.

```bash
uv sync --group notebook
uv run --group notebook jupyter lab
```
