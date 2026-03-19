import pandas as pd
import numpy as np

# =========================
# Пути
# =========================
TRAIN_PATH = "train.csv"

OUT_PARQUET = "train_filled_active_period.parquet"
OUT_CSV = "train_filled_active_period_sample.csv"   # небольшой sample для просмотра

# =========================
# Загрузка train
# =========================
print("Чтение train.csv ...")
train = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["date"],
    low_memory=False,
)

print("Исходный shape:", train.shape)

# =========================
# Базовая подготовка
# =========================
train["unique_id"] = train["store_nbr"].astype(str) + "_" + train["item_nbr"].astype(str)

# На всякий случай уберём дубли store/item/date, если вдруг есть
# Для Favorita обычно их быть не должно, но лучше не надеяться на милость датасета
dup_count = train.duplicated(subset=["unique_id", "date"]).sum()
print("Дубликатов по unique_id+date:", dup_count)

if dup_count > 0:
    train = (
        train.groupby(
            ["unique_id", "store_nbr", "item_nbr", "date"],
            as_index=False
        )
        .agg({
            "unit_sales": "sum",
            "onpromotion": "max",
            "id": "first" if "id" in train.columns else "size"
        })
    )
    if "id" not in train.columns:
        train = train.drop(columns=["id"], errors="ignore")

# =========================
# first_date / last_date по каждому ряду
# =========================
print("Расчёт first_date / last_date ...")
span = (
    train.groupby("unique_id", as_index=False)
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        store_nbr=("store_nbr", "first"),
        item_nbr=("item_nbr", "first"),
    )
)

print("Число рядов:", len(span))

# =========================
# Строим полный календарь внутри активного периода
# =========================
print("Построение полного календаря внутри активного периода ...")

parts = []
for row in span.itertuples(index=False):
    dates = pd.date_range(row.first_date, row.last_date, freq="D")
    part = pd.DataFrame({
        "unique_id": row.unique_id,
        "store_nbr": row.store_nbr,
        "item_nbr": row.item_nbr,
        "date": dates,
    })
    parts.append(part)

full_calendar = pd.concat(parts, ignore_index=True)
print("Календарь shape:", full_calendar.shape)

# =========================
# Merge с исходным train
# =========================
print("Merge с исходными продажами ...")
filled = full_calendar.merge(
    train[["unique_id", "date", "unit_sales", "onpromotion"]],
    on=["unique_id", "date"],
    how="left",
)

# =========================
# Заполнение пропусков внутри активного периода
# =========================
missing_unit_sales_before = filled["unit_sales"].isna().sum()
missing_onpromotion_before = filled["onpromotion"].isna().sum()

print("Пропусков unit_sales до заполнения:", missing_unit_sales_before)
print("Пропусков onpromotion до заполнения:", missing_onpromotion_before)

filled["unit_sales_was_missing"] = filled["unit_sales"].isna().astype("int8")
filled["onpromotion_was_missing"] = filled["onpromotion"].isna().astype("int8")

filled["unit_sales"] = filled["unit_sales"].fillna(0.0)
filled["onpromotion"] = filled["onpromotion"].fillna(False)

# Приведём типы
filled["unit_sales"] = filled["unit_sales"].astype("float32")
filled["onpromotion"] = filled["onpromotion"].astype("bool")

# Доп. полезные признаки
filled["unit_sales_nonneg"] = filled["unit_sales"].clip(lower=0)
filled["log_target"] = np.log1p(filled["unit_sales_nonneg"]).astype("float32")

# =========================
# Проверки
# =========================
print("\n==================== CHECKS ====================")
print("Итоговый shape:", filled.shape)
print("Пропусков unit_sales после заполнения:", filled["unit_sales"].isna().sum())
print("Пропусков onpromotion после заполнения:", filled["onpromotion"].isna().sum())
print("Число добавленных нулевых дней:", int(filled["unit_sales_was_missing"].sum()))

series_check = (
    filled.groupby("unique_id", as_index=False)
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        observed_days=("date", "nunique"),
    )
)
series_check["calendar_days"] = (series_check["last_date"] - series_check["first_date"]).dt.days + 1
series_check["gap_after_fill"] = series_check["calendar_days"] - series_check["observed_days"]

print("Рядов с дырками после заполнения:", int((series_check["gap_after_fill"] > 0).sum()))

print("\nСводка по заполненным дням на ряд:")
filled_days_per_series = (
    filled.groupby("unique_id")["unit_sales_was_missing"]
    .sum()
)
print(filled_days_per_series.describe())

# =========================
# Сохранение
# =========================
print("\nСохранение parquet ...")
filled.to_parquet(OUT_PARQUET, index=False)

# Для быстрого просмотра сохраним sample
sample = (
    filled.sort_values(["unique_id", "date"])
    .groupby("unique_id", group_keys=False)
    .head(5)
    .head(1000)
)
sample.to_csv(OUT_CSV, index=False)

print("Сохранено:")
print(" -", OUT_PARQUET)
print(" -", OUT_CSV)
