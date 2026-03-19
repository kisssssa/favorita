Что показал EDA:
- Самые сильные признаки — это история самого ряда: rolling_mean_7/14/28, lag_1/7/14/28, rolling_std_7. У них корреляции с log_target заметно выше, чем у внешних факторов. Например, rolling_mean_28, rolling_mean_14, rolling_mean_7 и лаги дают самые большие Spearman-значения после самого таргета.
- transactions полезен, но заметно слабее лаговых и rolling-признаков.
- onpromotion даёт сигнал, но умеренный.
- dcoilwtico и is_holiday_event по одиночной корреляции выглядят очень слабыми.
missing.py - поиск пропусков
missing fill - заполнение пропусков
