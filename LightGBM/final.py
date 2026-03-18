import os
import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import lightgbm as lgb

# =========================
# Ограничение числа ядер
# =========================
N_JOBS = min(32, os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)

# =========================
# Конфиг
# =========================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
ITEMS_PATH = "items.csv"
STORES_PATH = "stores.csv"
TRANSACTIONS_PATH = "transactions.csv"
OIL_PATH = "oil.csv"
HOLIDAYS_PATH = "holidays_events.csv"

# ВАЖНО: здесь должен быть JSON лучшего RAW-LightGBM прогона
BEST_PARAMS_PATH = "outputs/lightgbm_best_params.json"

OUTPUT_DIR = Path("outputs_lightgbm_final_submission_fulltrain")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = OUTPUT_DIR / "lightgbm_model.txt"
TEST_PRED_PATH = OUTPUT_DIR / "lightgbm_test_predictions.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

TRAIN_START_DATE = None

LAG_COLS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]
RANDOM_STATE = 42

# =========================
# Признаки
# =========================
STATIC_CAT_COLS = [
    "store_nbr",
    "item_nbr",
    "family",
    "class",
    "perishable",
    "city",
    "state",
    "type",
    "cluster",
]

KNOWN_NUM_COLS = [
    "onpromotion",
    "transactions",
    "dcoilwtico",
    "is_holiday_event",
    "dayofweek",
    "month",
    "day",
    "is_weekend",
]

DYNAMIC_NUM_COLS = (
    [f"lag_{l}" for l in LAG_COLS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + ["rolling_std_7"]
)

FEATURE_COLS = STATIC_CAT_COLS + KNOWN_NUM_COLS + DYNAMIC_NUM_COLS


# =========================
# Утилиты
# =========================
def make_unique_id(df: pd.DataFrame) -> pd.Series:
    return df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)


def safe_clip_forecast(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    pred = np.clip(pred, 0.0, None)
    return pred.astype(np.float32)


def inverse_log_target(pred_log: np.ndarray) -> np.ndarray:
    return safe_clip_forecast(np.expm1(pred_log))


def load_json(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            cmin = df[col].min()
            cmax = df[col].max()
            if cmin >= np.iinfo(np.int8).min and cmax <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif cmin >= np.iinfo(np.int16).min and cmax <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif cmin >= np.iinfo(np.int32).min and cmax <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
    return df


# =========================
# Загрузка исходных таблиц
# =========================
def load_raw_tables():
    print("Чтение train.csv ...")
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        parse_dates=["date"],
        low_memory=False,
    )

    print("Чтение test.csv ...")
    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    if TRAIN_START_DATE is not None:
        train = train[train["date"] >= pd.Timestamp(TRAIN_START_DATE)].copy()

    train["store_nbr"] = train["store_nbr"].astype(np.int16)
    train["item_nbr"] = train["item_nbr"].astype(np.int32)
    train["unit_sales"] = train["unit_sales"].astype(np.float32)

    test["store_nbr"] = test["store_nbr"].astype(np.int16)
    test["item_nbr"] = test["item_nbr"].astype(np.int32)

    print("Чтение справочников ...")
    items = pd.read_csv(ITEMS_PATH, low_memory=False)
    stores = pd.read_csv(STORES_PATH, low_memory=False)
    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=["date", "store_nbr", "transactions"],
        parse_dates=["date"],
        low_memory=False,
    )
    oil = pd.read_csv(
        OIL_PATH,
        usecols=["date", "dcoilwtico"],
        parse_dates=["date"],
        low_memory=False,
    )
    holidays = pd.read_csv(
        HOLIDAYS_PATH,
        usecols=["date"],
        parse_dates=["date"],
        low_memory=False,
    )

    items["item_nbr"] = pd.to_numeric(items["item_nbr"], errors="coerce").fillna(-1).astype(np.int32)
    items["class"] = pd.to_numeric(items["class"], errors="coerce").fillna(-1).astype(np.int32)
    items["perishable"] = pd.to_numeric(items["perishable"], errors="coerce").fillna(0).astype(np.int8)

    stores["store_nbr"] = pd.to_numeric(stores["store_nbr"], errors="coerce").fillna(-1).astype(np.int16)
    stores["cluster"] = pd.to_numeric(stores["cluster"], errors="coerce").fillna(-1).astype(np.int16)

    transactions["store_nbr"] = pd.to_numeric(transactions["store_nbr"], errors="coerce").fillna(-1).astype(np.int16)
    transactions["transactions"] = pd.to_numeric(transactions["transactions"], errors="coerce").fillna(0).astype(np.float32)

    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce").astype(np.float32)

    return train, test, items, stores, transactions, oil, holidays


# =========================
# Единое кодирование категорий
# =========================
def build_category_maps(items: pd.DataFrame, stores: pd.DataFrame):
    family_vals = sorted(items["family"].astype(str).fillna("__NA__").unique().tolist())
    city_vals = sorted(stores["city"].astype(str).fillna("__NA__").unique().tolist())
    state_vals = sorted(stores["state"].astype(str).fillna("__NA__").unique().tolist())
    type_vals = sorted(stores["type"].astype(str).fillna("__NA__").unique().tolist())

    maps = {
        "family": {v: i for i, v in enumerate(family_vals)},
        "city": {v: i for i, v in enumerate(city_vals)},
        "state": {v: i for i, v in enumerate(state_vals)},
        "type": {v: i for i, v in enumerate(type_vals)},
    }
    return maps


# =========================
# Подготовка train/test
# =========================
def enrich_df(df: pd.DataFrame,
              items: pd.DataFrame,
              stores: pd.DataFrame,
              transactions: pd.DataFrame,
              oil: pd.DataFrame,
              holidays: pd.DataFrame,
              cat_maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    holidays_flag = holidays[["date"]].drop_duplicates().copy()
    holidays_flag["is_holiday_event"] = np.int8(1)

    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(items, on="item_nbr", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(holidays_flag, on="date", how="left")

    df["unique_id"] = make_unique_id(df)

    if "unit_sales" in df.columns:
        df["unit_sales_nonneg"] = df["unit_sales"].clip(lower=0).astype(np.float32)
        df["log_target"] = np.log1p(df["unit_sales_nonneg"]).astype(np.float32)

    df["onpromotion"] = df["onpromotion"].astype("boolean").fillna(False).astype(np.int8)
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype(np.float32)
    df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill().astype(np.float32)
    df["is_holiday_event"] = df["is_holiday_event"].fillna(0).astype(np.int8)

    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    # кодируем строковые категории в int32
    df["family"] = df["family"].astype(str).fillna("__NA__").map(cat_maps["family"]).fillna(-1).astype(np.int32)
    df["city"] = df["city"].astype(str).fillna("__NA__").map(cat_maps["city"]).fillna(-1).astype(np.int32)
    df["state"] = df["state"].astype(str).fillna("__NA__").map(cat_maps["state"]).fillna(-1).astype(np.int32)
    df["type"] = df["type"].astype(str).fillna("__NA__").map(cat_maps["type"]).fillna(-1).astype(np.int32)

    # добиваем числовые категориальные
    df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(-1).astype(np.int32)
    df["perishable"] = pd.to_numeric(df["perishable"], errors="coerce").fillna(0).astype(np.int8)
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(np.int16)

    df = optimize_memory(df)
    return df


# =========================
# История и динамические признаки
# =========================
def build_history_dict(history_df: pd.DataFrame) -> Dict[str, List[float]]:
    hist_df = history_df.sort_values(["unique_id", "date"])
    history = (
        hist_df.groupby("unique_id")["unit_sales_nonneg"]
        .apply(lambda s: list(map(float, s.tolist())))
        .to_dict()
    )
    return history


def compute_dynamic_features_for_day(day_df: pd.DataFrame, history: Dict[str, List[float]]) -> pd.DataFrame:
    lag_1 = np.empty(len(day_df), dtype=np.float32)
    lag_7 = np.empty(len(day_df), dtype=np.float32)
    lag_14 = np.empty(len(day_df), dtype=np.float32)
    lag_28 = np.empty(len(day_df), dtype=np.float32)
    rm_7 = np.empty(len(day_df), dtype=np.float32)
    rm_14 = np.empty(len(day_df), dtype=np.float32)
    rm_28 = np.empty(len(day_df), dtype=np.float32)
    rs_7 = np.empty(len(day_df), dtype=np.float32)

    for i, uid in enumerate(day_df["unique_id"].values):
        hist = history.get(uid, [])

        if len(hist) >= 1:
            lag_1[i] = hist[-1]
        else:
            lag_1[i] = np.nan

        if len(hist) >= 7:
            lag_7[i] = hist[-7]
            rm_7[i] = float(np.mean(hist[-7:]))
            rs_7[i] = float(np.std(hist[-7:], ddof=0))
        elif len(hist) > 0:
            lag_7[i] = np.nan
            rm_7[i] = float(np.mean(hist))
            rs_7[i] = float(np.std(hist, ddof=0))
        else:
            lag_7[i] = np.nan
            rm_7[i] = np.nan
            rs_7[i] = 0.0

        if len(hist) >= 14:
            lag_14[i] = hist[-14]
            rm_14[i] = float(np.mean(hist[-14:]))
        elif len(hist) > 0:
            lag_14[i] = np.nan
            rm_14[i] = float(np.mean(hist))
        else:
            lag_14[i] = np.nan
            rm_14[i] = np.nan

        if len(hist) >= 28:
            lag_28[i] = hist[-28]
            rm_28[i] = float(np.mean(hist[-28:]))
        elif len(hist) > 0:
            lag_28[i] = np.nan
            rm_28[i] = float(np.mean(hist))
        else:
            lag_28[i] = np.nan
            rm_28[i] = np.nan

    out = day_df.copy()
    out["lag_1"] = lag_1
    out["lag_7"] = lag_7
    out["lag_14"] = lag_14
    out["lag_28"] = lag_28
    out["rolling_mean_7"] = rm_7
    out["rolling_mean_14"] = rm_14
    out["rolling_mean_28"] = rm_28
    out["rolling_std_7"] = rs_7
    return out


# =========================
# Финальные train features
# =========================
def prepare_full_train_features(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.sort_values(["unique_id", "date"]).reset_index(drop=True)

    grp = train_df.groupby("unique_id")["unit_sales_nonneg"]

    for lag in LAG_COLS:
        train_df[f"lag_{lag}"] = grp.shift(lag).astype(np.float32)

    for window in ROLLING_WINDOWS:
        train_df[f"rolling_mean_{window}"] = (
            grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .astype(np.float32)
        )

    train_df["rolling_std_7"] = (
        grp.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
        .astype(np.float32)
    )
    train_df["rolling_std_7"] = train_df["rolling_std_7"].fillna(0).astype(np.float32)

    essential_lags = [f"lag_{l}" for l in LAG_COLS]
    train_df = train_df.dropna(subset=essential_lags).reset_index(drop=True)

    # LightGBM умеет в int-коды как обычные признаки, но лучше явно category
    for col in STATIC_CAT_COLS:
        train_df[col] = train_df[col].astype("category")

    train_df = optimize_memory(train_df)
    return train_df


def fit_lightgbm_full(train_feat: pd.DataFrame, best_params: dict):
    X = train_feat[FEATURE_COLS].copy()
    y = train_feat["log_target"].copy()

    for col in STATIC_CAT_COLS:
        X[col] = X[col].astype("category")

    params = best_params.copy()
    best_iteration = int(params.pop("best_iteration_", 2000))

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        n_estimators=best_iteration,
        verbosity=-1,
        **params,
    )

    model.fit(
        X,
        y,
        categorical_feature=STATIC_CAT_COLS,
    )
    return model


# =========================
# Прогноз на test
# =========================
def recursive_predict_test(
    model,
    test_prepared: pd.DataFrame,
    final_history: pd.DataFrame,
) -> pd.DataFrame:
    history = build_history_dict(final_history)
    pred_parts = []
    test_dates = sorted(test_prepared["date"].unique())

    for current_date in test_dates:
        print("Прогноз на дату:", pd.Timestamp(current_date).date())

        day_df = test_prepared[test_prepared["date"] == current_date].copy()
        day_df = compute_dynamic_features_for_day(day_df, history)

        for col in DYNAMIC_NUM_COLS:
            day_df[col] = day_df[col].fillna(0.0).astype(np.float32)

        X_day = day_df[FEATURE_COLS].copy()
        for col in STATIC_CAT_COLS:
            X_day[col] = X_day[col].astype("category")

        pred_log = model.predict(X_day)
        pred = inverse_log_target(pred_log)

        day_pred = day_df[["id", "unique_id", "date"]].copy()
        day_pred["unit_sales"] = pred
        pred_parts.append(day_pred)

        for uid, p in zip(day_df["unique_id"].values, pred):
            if uid not in history:
                history[uid] = []
            history[uid].append(float(p))

        del day_df, X_day
        gc.collect()

    pred_df = pd.concat(pred_parts, ignore_index=True)
    pred_df["unit_sales"] = pred_df["unit_sales"].clip(lower=0).astype(np.float32)
    return pred_df


# =========================
# main
# =========================
def main():
    print("N_JOBS =", N_JOBS)
    print("TRAIN_START_DATE =", TRAIN_START_DATE)
    print("BEST_PARAMS_PATH =", BEST_PARAMS_PATH)

    print("Загрузка лучших гиперпараметров ...")
    best_params = load_json(BEST_PARAMS_PATH)
    print("Best params loaded:", best_params)

    train, test, items, stores, transactions, oil, holidays = load_raw_tables()
    cat_maps = build_category_maps(items, stores)

    print("Подготовка full train ...")
    full_train_df = enrich_df(train, items, stores, transactions, oil, holidays, cat_maps)

    print("Подготовка final history ...")
    final_history = full_train_df[["unique_id", "date", "unit_sales_nonneg"]].copy()

    print("Построение лагов full train ...")
    full_train_df = prepare_full_train_features(full_train_df)

    # больше train не нужен
    del train, items, stores, transactions, oil, holidays
    gc.collect()

    print("Подготовка test ...")
    # перечитываем справочники отдельно, чтобы не держать всё вместе в памяти во время train feature engineering
    _, _, items2, stores2, transactions2, oil2, holidays2 = load_raw_tables()
    cat_maps2 = build_category_maps(items2, stores2)

    test_prepared = enrich_df(test, items2, stores2, transactions2, oil2, holidays2, cat_maps2)
    test_prepared = test_prepared.sort_values(["unique_id", "date"]).reset_index(drop=True)

    for col in STATIC_CAT_COLS:
        full_train_df[col] = full_train_df[col].astype("category")
        test_prepared[col] = test_prepared[col].astype("category")

    print("Full-train shape:", full_train_df.shape)
    print("Test shape:", test_prepared.shape)
    print("Train unique_id:", full_train_df["unique_id"].nunique())
    print("Test unique_id:", test_prepared["unique_id"].nunique())
    print(
        "Test unique_id absent in train:",
        len(set(test_prepared["unique_id"].unique()) - set(final_history["unique_id"].unique()))
    )

    print("\nОбучение финальной LightGBM ...")
    final_model = fit_lightgbm_full(full_train_df, best_params)
    final_model.booster_.save_model(str(MODEL_PATH))

    # можно освободить train
    del full_train_df
    gc.collect()

    print("\nРекурсивный прогноз на test ...")
    test_pred_df = recursive_predict_test(
        model=final_model,
        test_prepared=test_prepared,
        final_history=final_history,
    )

    test_pred_df.to_csv(TEST_PRED_PATH, index=False)

    submission = test_pred_df[["id", "unit_sales"]].copy()
    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print("\nГотово.")
    print("Артефакты:")
    print(" -", MODEL_PATH)
    print(" -", TEST_PRED_PATH)
    print(" -", SUBMISSION_PATH)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()