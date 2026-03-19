import pandas as pd
import numpy as np

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
ITEMS_PATH = "items.csv"
STORES_PATH = "stores.csv"
TRANSACTIONS_PATH = "transactions.csv"
OIL_PATH = "oil.csv"
HOLIDAYS_PATH = "holidays_events.csv"

print("Чтение файлов...")
train = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["date"],
    low_memory=False
)

test = pd.read_csv(
    TEST_PATH,
    parse_dates=["date"],
    low_memory=False
)

items = pd.read_csv(ITEMS_PATH, low_memory=False)
stores = pd.read_csv(STORES_PATH, low_memory=False)
transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"], low_memory=False)
oil = pd.read_csv(OIL_PATH, parse_dates=["date"], low_memory=False)
holidays = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"], low_memory=False)

def missing_report(df: pd.DataFrame, name: str):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)

    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print("shape:", df.shape)

    if len(miss) == 0:
        print("Пропусков нет.")
        return

    report = pd.DataFrame({
        "missing_count": miss,
        "missing_pct": (miss / len(df) * 100).round(4)
    })
    print(report)


missing_report(train, "train")
missing_report(test, "test")
missing_report(items, "items")
missing_report(stores, "stores")
missing_report(transactions, "transactions")
missing_report(oil, "oil")
missing_report(holidays, "holidays_events")

print("\n==================== TARGET CHECK ====================")
print("train['unit_sales'] NaN count:", train["unit_sales"].isna().sum())
print("train['unit_sales'] < 0 count:", (train["unit_sales"] < 0).sum())

print("\n==================== ONPROMOTION CHECK ====================")
if "onpromotion" in train.columns:
    print("train onpromotion NaN:", train["onpromotion"].isna().sum())
    print(train["onpromotion"].value_counts(dropna=False))

if "onpromotion" in test.columns:
    print("\ntest onpromotion NaN:", test["onpromotion"].isna().sum())
    print(test["onpromotion"].value_counts(dropna=False))

print("\n==================== TRANSACTIONS CHECK ====================")
print("transactions NaN by column:")
print(transactions.isna().sum())

if "transactions" in transactions.columns:
    print("transactions['transactions'] NaN count:", transactions["transactions"].isna().sum())

train_store_date = train[["store_nbr", "date"]].drop_duplicates()
transactions_store_date = transactions[["store_nbr", "date"]].drop_duplicates()

merged_tx = train_store_date.merge(
    transactions_store_date.assign(has_tx=1),
    on=["store_nbr", "date"],
    how="left"
)

missing_tx_pairs = merged_tx["has_tx"].isna().sum()
print("Train store/date pairs without transactions match:", missing_tx_pairs)

print("\n==================== OIL CHECK ====================")
print("oil NaN by column:")
print(oil.isna().sum())

oil_dates = pd.DataFrame({"date": pd.date_range(train["date"].min(), train["date"].max(), freq="D")})
oil_cov = oil_dates.merge(oil.assign(has_oil=1), on="date", how="left")
print("Dates in train range without oil row:", oil_cov["has_oil"].isna().sum())

print("\n==================== HOLIDAYS CHECK ====================")
print("holidays NaN by column:")
print(holidays.isna().sum())

print("\n==================== SERIES GAP CHECK ====================")

train = train.sort_values(["store_nbr", "item_nbr", "date"]).copy()
train["unique_id"] = train["store_nbr"].astype(str) + "_" + train["item_nbr"].astype(str)

series_span = (
    train.groupby("unique_id")
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        observed_days=("date", "nunique"),
    )
    .reset_index()
)

series_span["calendar_days"] = (
    series_span["last_date"] - series_span["first_date"]
).dt.days + 1

series_span["missing_inside_span"] = series_span["calendar_days"] - series_span["observed_days"]

print("Всего рядов:", len(series_span))
print("Рядов с пропущенными датами внутри наблюдаемого диапазона:",
      (series_span["missing_inside_span"] > 0).sum())
print("Доля таких рядов, %:",
      round((series_span["missing_inside_span"] > 0).mean() * 100, 4))

print("\nТоп-20 рядов с наибольшим числом пропусков внутри диапазона:")
print(
    series_span.sort_values("missing_inside_span", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\n==================== GAP SUMMARY ====================")
print(series_span["missing_inside_span"].describe())

print("\n==================== LATE START CHECK ====================")
global_min_date = train["date"].min()
series_span["days_after_global_start"] = (series_span["first_date"] - global_min_date).dt.days

print(series_span["days_after_global_start"].describe())

print("\nТоп-20 рядов, которые стартовали позже всего:")
print(
    series_span.sort_values("days_after_global_start", ascending=False)
    .head(20)[["unique_id", "first_date", "last_date", "observed_days", "calendar_days", "missing_inside_span", "days_after_global_start"]]
    .to_string(index=False)
)

print("\n==================== TEST VS TRAIN SERIES ====================")
test["unique_id"] = test["store_nbr"].astype(str) + "_" + test["item_nbr"].astype(str)

train_uids = set(train["unique_id"].unique())
test_uids = set(test["unique_id"].unique())

print("Train unique_id:", len(train_uids))
print("Test unique_id:", len(test_uids))
print("Test unique_id absent in train:", len(test_uids - train_uids))
print("Train unique_id absent in test:", len(train_uids - test_uids))
