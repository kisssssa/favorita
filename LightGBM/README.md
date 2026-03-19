Признаки общие для всех моделей: 
- store_nbr
- item_nbr
- family
- class
- perishable
- city
- state
- type
- onpromotion
- transactions
- dcoilwtico
- is_holiday_event
- dayofweek
- month
- day
- is_weekend
- лаги: lag_1, lag_7, lag_14, lag_28
- скользящие средние: rolling_mean_7, rolling_mean_14, rolling_mean_28
- кользящее стандартное отклонение: rolling_std_7

Модели:
- baseline.py — статистические baseline-модели (Naive, SeasonalNaive, AutoTheta, AutoETS)
