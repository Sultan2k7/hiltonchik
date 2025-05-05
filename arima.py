import pandas as pd
import numpy as np
from tqdm import tqdm
from pmdarima import auto_arima
from datetime import timedelta

# Загружаем данные
train = pd.read_csv('/content/venv/train.csv', parse_dates=['submitted_date'])
train['submitted_date'] = pd.to_datetime(train['submitted_date'])
split_date = train.submitted_date.max()

# Убедимся, что сортировка есть (важно для временных рядов)
train = train.sort_values(['category', 'submitted_date'])

# Получаем список последних недельных значений
last_week_df = train[(split_date - train.submitted_date ).dt.days < 7]
last_week_papers = last_week_df.groupby('category').num_papers.sum()
# Для записи результатов
records = []

# Группировка по категориям
for cat_name in tqdm(last_week_papers.index):
    # Фильтрация по категории, вся история
    cat_df = train[train['category'] == cat_name]
    ts = cat_df.set_index('submitted_date')['num_papers'].asfreq('D').fillna(0)

    try:
        # Автоматический подбор SARIMA модели
        model = auto_arima(ts,
                           seasonal=True,
                           m=7,  # недельная сезонность
                           stepwise=True,
                           suppress_warnings=True,
                           error_action='ignore',
                           max_p=3, max_q=3,
                           max_P=2, max_Q=2,
                           max_order=None,
                           d=None, D=None,
                           trace=False)

        # Прогноз на 56 дней (8 недель)
        forecast = model.predict(n_periods=56)

    except Exception as e:
        # fallback: повтор последней недели
        forecast = np.full(56, last_week_papers[cat_name])

    # Агрегация по неделям
    for week_id in range(1, 9):
        week_sum = forecast[(week_id - 1) * 7: week_id * 7].sum()
        records.append({
            'id': f"{cat_name}__{week_id}",
            'num_papers': week_sum
        })

# Финальный DataFrame
submission = pd.DataFrame(records)
#submission['num_papers'] = submission['num_papers'].round().astype(int)

# Сохраняем
submission.to_csv("submission_auto_arima.csv", index=False)
