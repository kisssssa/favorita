import os

SAFE_THREADS = "8"
os.environ["OMP_NUM_THREADS"] = SAFE_THREADS
os.environ["MKL_NUM_THREADS"] = SAFE_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = SAFE_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = SAFE_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = SAFE_THREADS
os.environ["BLIS_NUM_THREADS"] = SAFE_THREADS

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, fcluster

N_JOBS = min(8, os.cpu_count() or 1)
torch.set_num_threads(4)

TRAIN_PATH = "train.csv"
ITEMS_PATH = "items.csv"
STORES_PATH = "stores.csv"
TRANSACTIONS_PATH = "transactions.csv"
OIL_PATH = "oil.csv"
HOLIDAYS_PATH = "holidays_events.csv"

OUTPUT_DIR = Path("outputs_lightgbm_ae_hier_cluster")
CACHE_DIR = OUTPUT_DIR / "cache"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_START_DATE = None

HORIZON = 28
INNER_EVAL_DAYS = 28

N_TRIALS_LGBM = 20
OPTUNA_TIMEOUT_LGBM = None
TUNING_UID_SAMPLE_LGBM = 30000

RANDOM_STATE = 42

LAG_COLS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

SERIES_HISTORY_DAYS = 180
LATENT_DIM = 16
AE_EPOCHS = 20
AE_BATCH_SIZE = 4096
AE_LR = 1e-3

N_PROTOTYPES = 1000
N_SERIES_CLUSTERS = 5

SKIP_DONE_MODELS = True

TRAIN_FEAT_PATH = CACHE_DIR / "train_features.parquet"
VALID_KNOWN_PATH = CACHE_DIR / "valid_known.parquet"
VALID_TRUTH_PATH = CACHE_DIR / "valid_truth.parquet"
HISTORY_PREVALID_PATH = CACHE_DIR / "history_prevalid.parquet"
META_PATH = CACHE_DIR / "meta.json"

SERIES_MATRIX_PATH = CACHE_DIR / "series_matrix.npy"
SERIES_UIDS_PATH = CACHE_DIR / "series_uids.json"
CLUSTER_MAP_PATH = CACHE_DIR / "series_cluster_map.parquet"

TRAIN_FEAT_CLUSTERED_PATH = CACHE_DIR / "train_features_clustered.parquet"
VALID_KNOWN_CLUSTERED_PATH = CACHE_DIR / "valid_known_clustered.parquet"
HISTORY_PREVALID_CLUSTERED_PATH = CACHE_DIR / "history_prevalid_clustered.parquet"

LGBM_STUDY_PATH = OUTPUT_DIR / "optuna_lgbm.db"
LGBM_DONE_FLAG = OUTPUT_DIR / "lightgbm_done.flag"

LGBM_MODEL_PATH = OUTPUT_DIR / "lightgbm_model.txt"
LGBM_BEST_PARAMS_PATH = OUTPUT_DIR / "lightgbm_best_params.json"
LGBM_METRICS_PATH = OUTPUT_DIR / "lightgbm_metrics_h28.json"
LGBM_PRED_PATH = OUTPUT_DIR / "lightgbm_valid_predictions_h28.csv"

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
    "series_cluster",
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

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def nwrmsle(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    weights = np.asarray(weights, dtype=float)
    sq = (np.log1p(y_pred) - np.log1p(y_true)) ** 2
    return float(np.sqrt(np.sum(weights * sq) / np.sum(weights)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def make_unique_id(df: pd.DataFrame) -> pd.Series:
    return df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)


def safe_clip_forecast(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    pred = np.clip(pred, 0.0, None)
    return pred


def inverse_log_target(pred_log: np.ndarray) -> np.ndarray:
    pred = np.expm1(pred_log)
    return safe_clip_forecast(pred)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_completed_trials(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def read_train_chunked() -> pd.DataFrame:
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    chunks = []

    start_ts = pd.Timestamp(TRAIN_START_DATE) if TRAIN_START_DATE is not None else None

    for chunk in pd.read_csv(
        TRAIN_PATH,
        usecols=usecols,
        parse_dates=["date"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        if start_ts is not None:
            chunk = chunk[chunk["date"] >= start_ts].copy()
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_and_prepare_base_features(force_rebuild: bool = False):
    if (
        not force_rebuild
        and TRAIN_FEAT_PATH.exists()
        and VALID_KNOWN_PATH.exists()
        and VALID_TRUTH_PATH.exists()
        and HISTORY_PREVALID_PATH.exists()
        and META_PATH.exists()
    ):
        print("Загрузка базового кэша фичей ...")
        train_feat = pd.read_parquet(TRAIN_FEAT_PATH)
        valid_known = pd.read_parquet(VALID_KNOWN_PATH)
        valid_truth = pd.read_parquet(VALID_TRUTH_PATH)
        history_prevalid = pd.read_parquet(HISTORY_PREVALID_PATH)
        meta = load_json(META_PATH)
        return train_feat, valid_known, valid_truth, history_prevalid, meta

    print("Чтение train.csv чанками ...")
    train = read_train_chunked()

    print("Чтение справочников ...")
    items = pd.read_csv(ITEMS_PATH)
    stores = pd.read_csv(STORES_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
    oil = pd.read_csv(OIL_PATH, parse_dates=["date"])
    holidays = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])

    holidays_flag = holidays[["date"]].drop_duplicates().copy()
    holidays_flag["is_holiday_event"] = 1

    print("Merge таблиц ...")
    df = train.merge(transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(items, on="item_nbr", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(holidays_flag, on="date", how="left")

    df["unit_sales_nonneg"] = df["unit_sales"].clip(lower=0)
    df["log_target"] = np.log1p(df["unit_sales_nonneg"])
    df["unique_id"] = make_unique_id(df)

    df["onpromotion"] = df["onpromotion"].astype("boolean").fillna(False).astype("int8")
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype("float32")
    df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill().astype("float32")
    df["is_holiday_event"] = df["is_holiday_event"].fillna(0).astype("int8")

    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    df = df.sort_values(["unique_id", "date"]).reset_index(drop=True)

    print("Построение lag/rolling признаков ...")
    grp = df.groupby("unique_id")["unit_sales_nonneg"]

    for lag in LAG_COLS:
        df[f"lag_{lag}"] = grp.shift(lag).astype("float32")

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .astype("float32")
        )

    df["rolling_std_7"] = (
        grp.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
        .astype("float32")
    )
    df["rolling_std_7"] = df["rolling_std_7"].fillna(0).astype("float32")

    max_date = df["date"].max()
    valid_start = max_date - pd.Timedelta(days=HORIZON - 1)
    inner_eval_start = valid_start - pd.Timedelta(days=INNER_EVAL_DAYS)

    history_prevalid = df[df["date"] < valid_start][["unique_id", "date", "unit_sales_nonneg"]].copy()

    train_feat = df[df["date"] < valid_start].copy()
    essential_lags = [f"lag_{l}" for l in LAG_COLS]
    train_feat = train_feat.dropna(subset=essential_lags).reset_index(drop=True)

    valid_known_cols = ["unique_id", "date"] + [c for c in STATIC_CAT_COLS if c != "series_cluster"] + KNOWN_NUM_COLS
    valid_known = df[df["date"] >= valid_start][valid_known_cols].copy()

    valid_truth = df[df["date"] >= valid_start][["unique_id", "date", "item_nbr", "unit_sales_nonneg", "perishable"]].copy()
    valid_truth = valid_truth.rename(columns={"unit_sales_nonneg": "y_true"})
    valid_truth["weight"] = np.where(valid_truth["perishable"] == 1, 1.25, 1.0)
    valid_truth = valid_truth[["unique_id", "date", "item_nbr", "y_true", "weight"]].copy()

    base_static = [c for c in STATIC_CAT_COLS if c != "series_cluster"]
    for col in base_static:
        train_feat[col] = train_feat[col].astype("category")
        valid_known[col] = valid_known[col].astype("category")

    meta = {
        "valid_start": str(valid_start.date()),
        "inner_eval_start": str(inner_eval_start.date()),
        "max_date": str(max_date.date()),
        "train_start_date": TRAIN_START_DATE,
        "horizon": HORIZON,
    }

    print("Сохранение базового кэша фичей ...")
    train_feat.to_parquet(TRAIN_FEAT_PATH, index=False)
    valid_known.to_parquet(VALID_KNOWN_PATH, index=False)
    valid_truth.to_parquet(VALID_TRUTH_PATH, index=False)
    history_prevalid.to_parquet(HISTORY_PREVALID_PATH, index=False)
    save_json(meta, META_PATH)

    return train_feat, valid_known, valid_truth, history_prevalid, meta


class SeriesAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def build_series_matrix(history_prevalid: pd.DataFrame, valid_known: pd.DataFrame, meta: dict):
    if SERIES_MATRIX_PATH.exists() and SERIES_UIDS_PATH.exists():
        print("Загрузка series matrix из кэша ...")
        X = np.load(SERIES_MATRIX_PATH)
        uids = load_json(SERIES_UIDS_PATH)
        return X, uids

    print("Построение матрицы рядов для AE ...")
    target_uids = sorted(valid_known["unique_id"].unique().tolist())
    valid_start = pd.Timestamp(meta["valid_start"])
    start_date = valid_start - pd.Timedelta(days=SERIES_HISTORY_DAYS)

    dates = pd.date_range(start_date, valid_start - pd.Timedelta(days=1), freq="D")

    hist = history_prevalid[
        (history_prevalid["unique_id"].isin(target_uids)) &
        (history_prevalid["date"] >= start_date)
    ].copy()

    pivot = hist.pivot_table(
        index="unique_id",
        columns="date",
        values="unit_sales_nonneg",
        aggfunc="sum",
        fill_value=0.0,
    )

    pivot = pivot.reindex(index=target_uids, columns=dates, fill_value=0.0)
    X = np.log1p(pivot.to_numpy(dtype=np.float32))

    np.save(SERIES_MATRIX_PATH, X)
    save_json(target_uids, SERIES_UIDS_PATH)

    return X, target_uids


def train_autoencoder_and_encode(X: np.ndarray, latent_dim: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"AE device: {device}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=False)

    model = SeriesAutoencoder(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(AE_EPOCHS):
        total_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(ds)
        print(f"AE epoch {epoch + 1}/{AE_EPOCHS}, loss={avg_loss:.6f}")

    model.eval()
    encodings = []
    with torch.no_grad():
        infer_dl = DataLoader(ds, batch_size=AE_BATCH_SIZE, shuffle=False, drop_last=False)
        for (batch,) in infer_dl:
            batch = batch.to(device)
            z = model.encoder(batch)
            encodings.append(z.cpu().numpy())

    Z = np.vstack(encodings).astype(np.float32)
    return Z


def build_series_cluster_map(history_prevalid: pd.DataFrame, valid_known: pd.DataFrame, meta: dict):
    if CLUSTER_MAP_PATH.exists():
        print("Загрузка cluster_map из кэша ...")
        return pd.read_parquet(CLUSTER_MAP_PATH)

    X, uids = build_series_matrix(history_prevalid, valid_known, meta)
    Z = train_autoencoder_and_encode(X, latent_dim=LATENT_DIM)

    print("MiniBatchKMeans -> prototypes ...")
    n_prototypes = min(N_PROTOTYPES, len(uids))
    proto_kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        batch_size=4096,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    proto_labels = proto_kmeans.fit_predict(Z)
    proto_centers = proto_kmeans.cluster_centers_

    print("Hierarchical clustering on prototypes ...")
    proto_linkage = linkage(proto_centers, method="ward")
    proto_cluster_ids = fcluster(proto_linkage, t=N_SERIES_CLUSTERS, criterion="maxclust") - 1

    series_cluster = proto_cluster_ids[proto_labels].astype(int)

    cluster_map = pd.DataFrame({
        "unique_id": uids,
        "series_cluster": series_cluster,
    })
    cluster_map.to_parquet(CLUSTER_MAP_PATH, index=False)

    print("Распределение по series_cluster:")
    print(cluster_map["series_cluster"].value_counts().sort_index())

    return cluster_map


def attach_series_cluster(
    train_feat: pd.DataFrame,
    valid_known: pd.DataFrame,
    history_prevalid: pd.DataFrame,
    cluster_map: pd.DataFrame,
):
    if (
        TRAIN_FEAT_CLUSTERED_PATH.exists()
        and VALID_KNOWN_CLUSTERED_PATH.exists()
        and HISTORY_PREVALID_CLUSTERED_PATH.exists()
    ):
        print("Загрузка clustered features из кэша ...")
        train_feat_c = pd.read_parquet(TRAIN_FEAT_CLUSTERED_PATH)
        valid_known_c = pd.read_parquet(VALID_KNOWN_CLUSTERED_PATH)
        history_prevalid_c = pd.read_parquet(HISTORY_PREVALID_CLUSTERED_PATH)
        return train_feat_c, valid_known_c, history_prevalid_c

    target_uids = set(cluster_map["unique_id"].tolist())

    train_feat_c = train_feat[train_feat["unique_id"].isin(target_uids)].copy()
    valid_known_c = valid_known[valid_known["unique_id"].isin(target_uids)].copy()
    history_prevalid_c = history_prevalid[history_prevalid["unique_id"].isin(target_uids)].copy()

    train_feat_c = train_feat_c.merge(cluster_map, on="unique_id", how="left")
    valid_known_c = valid_known_c.merge(cluster_map, on="unique_id", how="left")
    history_prevalid_c = history_prevalid_c.merge(cluster_map, on="unique_id", how="left")

    train_feat_c["series_cluster"] = train_feat_c["series_cluster"].astype("int16").astype("category")
    valid_known_c["series_cluster"] = valid_known_c["series_cluster"].astype("int16").astype("category")

    train_feat_c.to_parquet(TRAIN_FEAT_CLUSTERED_PATH, index=False)
    valid_known_c.to_parquet(VALID_KNOWN_CLUSTERED_PATH, index=False)
    history_prevalid_c.to_parquet(HISTORY_PREVALID_CLUSTERED_PATH, index=False)

    return train_feat_c, valid_known_c, history_prevalid_c


def sample_uids_for_tuning(train_feat: pd.DataFrame, valid_known: pd.DataFrame, sample_size: int) -> list[str]:
    all_uids = np.intersect1d(train_feat["unique_id"].unique(), valid_known["unique_id"].unique())
    if sample_size is None or sample_size >= len(all_uids):
        return list(all_uids)

    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(all_uids, size=sample_size, replace=False)
    return list(chosen)


def split_train_inner_eval(train_feat: pd.DataFrame, inner_eval_start: str):
    inner_eval_start = pd.Timestamp(inner_eval_start)
    fit_df = train_feat[train_feat["date"] < inner_eval_start].copy()
    eval_df = train_feat[train_feat["date"] >= inner_eval_start].copy()
    return fit_df, eval_df


def prepare_lgbm_frames(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_fit = fit_df[FEATURE_COLS].copy()
    y_fit = fit_df["log_target"].copy()

    X_eval = eval_df[FEATURE_COLS].copy()
    y_eval = eval_df["log_target"].copy()

    for col in STATIC_CAT_COLS:
        X_fit[col] = X_fit[col].astype("category")
        X_eval[col] = X_eval[col].astype("category")

    return X_fit, y_fit, X_eval, y_eval


def build_history_dict(history_prevalid: pd.DataFrame) -> Dict[str, list[float]]:
    hist_df = history_prevalid.sort_values(["unique_id", "date"])
    history = (
        hist_df.groupby("unique_id")["unit_sales_nonneg"]
        .apply(lambda s: list(map(float, s.tolist())))
        .to_dict()
    )
    return history


def compute_dynamic_features_for_day(day_df: pd.DataFrame, history: Dict[str, list[float]]) -> pd.DataFrame:
    lag_1 = []
    lag_7 = []
    lag_14 = []
    lag_28 = []

    rm_7 = []
    rm_14 = []
    rm_28 = []
    rs_7 = []

    for uid in day_df["unique_id"].values:
        hist = history.get(uid, [])

        def get_lag(k: int):
            return hist[-k] if len(hist) >= k else np.nan

        def get_roll_mean(k: int):
            if len(hist) == 0:
                return np.nan
            arr = hist[-k:] if len(hist) >= k else hist
            return float(np.mean(arr))

        def get_roll_std(k: int):
            if len(hist) == 0:
                return 0.0
            arr = hist[-k:] if len(hist) >= k else hist
            return float(np.std(arr, ddof=0))

        lag_1.append(get_lag(1))
        lag_7.append(get_lag(7))
        lag_14.append(get_lag(14))
        lag_28.append(get_lag(28))

        rm_7.append(get_roll_mean(7))
        rm_14.append(get_roll_mean(14))
        rm_28.append(get_roll_mean(28))
        rs_7.append(get_roll_std(7))

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


def recursive_predict_valid(
    model,
    valid_known: pd.DataFrame,
    history_prevalid: pd.DataFrame,
    valid_truth: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    history = build_history_dict(history_prevalid)
    pred_parts = []
    valid_dates = sorted(valid_known["date"].unique())

    for current_date in valid_dates:
        day_df = valid_known[valid_known["date"] == current_date].copy()
        day_df = compute_dynamic_features_for_day(day_df, history)

        for col in DYNAMIC_NUM_COLS:
            day_df[col] = day_df[col].fillna(0.0)

        X_day = day_df[FEATURE_COLS].copy()
        for col in STATIC_CAT_COLS:
            X_day[col] = X_day[col].astype("category")

        pred_log = model.predict(X_day, num_iteration=model.best_iteration_)
        pred = inverse_log_target(pred_log)

        day_pred = day_df[["unique_id", "date"]].copy()
        day_pred["y_pred"] = pred
        pred_parts.append(day_pred)

        for uid, p in zip(day_df["unique_id"].values, pred):
            if uid not in history:
                history[uid] = []
            history[uid].append(float(p))

    pred_df = pd.concat(pred_parts, ignore_index=True)
    eval_df = valid_truth.merge(pred_df, on=["unique_id", "date"], how="left")
    eval_df["y_pred"] = eval_df["y_pred"].fillna(0.0).clip(lower=0)

    metrics = {
        "NWRMSLE": nwrmsle(eval_df["y_true"].values, eval_df["y_pred"].values, eval_df["weight"].values),
        "WAPE": wape(eval_df["y_true"].values, eval_df["y_pred"].values),
        "RMSLE_unweighted": rmsle(eval_df["y_true"].values, eval_df["y_pred"].values),
        "n_rows_eval": int(len(eval_df)),
        "n_unique_ids_eval": int(eval_df["unique_id"].nunique()),
    }
    return eval_df, metrics


def fit_lightgbm(fit_df: pd.DataFrame, eval_df: pd.DataFrame, params: dict):
    X_fit, y_fit, X_eval, y_eval = prepare_lgbm_frames(fit_df, eval_df)

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        n_estimators=5000,
        verbosity=-1,
        **params,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_eval, y_eval)],
        eval_metric="rmse",
        categorical_feature=STATIC_CAT_COLS,
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def make_lgbm_objective(train_feat, valid_known, valid_truth, history_prevalid, meta):
    tuning_uids = set(sample_uids_for_tuning(train_feat, valid_known, TUNING_UID_SAMPLE_LGBM))
    inner_eval_start = meta["inner_eval_start"]

    train_sub = train_feat[train_feat["unique_id"].isin(tuning_uids)].copy()
    valid_known_sub = valid_known[valid_known["unique_id"].isin(tuning_uids)].copy()
    valid_truth_sub = valid_truth[valid_truth["unique_id"].isin(tuning_uids)].copy()
    history_sub = history_prevalid[history_prevalid["unique_id"].isin(tuning_uids)].copy()

    fit_df, eval_df = split_train_inner_eval(train_sub, inner_eval_start)

    def objective(trial: optuna.Trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = fit_lightgbm(fit_df, eval_df, params)
        _, metrics = recursive_predict_valid(
            model=model,
            valid_known=valid_known_sub,
            history_prevalid=history_sub,
            valid_truth=valid_truth_sub,
        )
        return metrics["NWRMSLE"]

    return objective


def run_lightgbm(train_feat, valid_known, valid_truth, history_prevalid, meta):
    if SKIP_DONE_MODELS and LGBM_DONE_FLAG.exists():
        print("LightGBM уже завершён. Скипаем.")
        return

    print("\n========== LightGBM + series_cluster ==========")
    study = optuna.create_study(
        study_name="favorita_lightgbm_ae_hier_cluster",
        direction="minimize",
        storage=f"sqlite:///{LGBM_STUDY_PATH}",
        load_if_exists=True,
    )

    completed = count_completed_trials(study)
    remaining = max(0, N_TRIALS_LGBM - completed)
    print(f"LightGBM Optuna: completed={completed}, remaining={remaining}")

    if remaining > 0:
        objective = make_lgbm_objective(train_feat, valid_known, valid_truth, history_prevalid, meta)
        study.optimize(objective, n_trials=remaining, timeout=OPTUNA_TIMEOUT_LGBM, show_progress_bar=True)

    best_params = study.best_trial.params
    save_json(best_params, LGBM_BEST_PARAMS_PATH)
    print("LightGBM best params:", best_params)

    fit_df, eval_df = split_train_inner_eval(train_feat, meta["inner_eval_start"])
    model = fit_lightgbm(fit_df, eval_df, best_params)

    pred_df, metrics = recursive_predict_valid(
        model=model,
        valid_known=valid_known,
        history_prevalid=history_prevalid,
        valid_truth=valid_truth,
    )

    save_json(metrics, LGBM_METRICS_PATH)
    pred_df.to_csv(LGBM_PRED_PATH, index=False)
    model.booster_.save_model(str(LGBM_MODEL_PATH))

    print("LightGBM metrics:", metrics)
    LGBM_DONE_FLAG.touch()


def main():
    print(f"N_JOBS = {N_JOBS}")
    print(f"TRAIN_START_DATE = {TRAIN_START_DATE}")
    print(f"HORIZON = {HORIZON}")
    print(f"SERIES_HISTORY_DAYS = {SERIES_HISTORY_DAYS}")
    print(f"LATENT_DIM = {LATENT_DIM}")
    print(f"N_PROTOTYPES = {N_PROTOTYPES}")
    print(f"N_SERIES_CLUSTERS = {N_SERIES_CLUSTERS}")
    print(f"TUNING_UID_SAMPLE_LGBM = {TUNING_UID_SAMPLE_LGBM}")
    print(f"FEATURE_COLS = {FEATURE_COLS}")

    train_feat, valid_known, valid_truth, history_prevalid, meta = load_and_prepare_base_features(force_rebuild=False)

    cluster_map = build_series_cluster_map(history_prevalid, valid_known, meta)

    train_feat_c, valid_known_c, history_prevalid_c = attach_series_cluster(
        train_feat=train_feat,
        valid_known=valid_known,
        history_prevalid=history_prevalid,
        cluster_map=cluster_map,
    )

    valid_truth_c = valid_truth[valid_truth["unique_id"].isin(cluster_map["unique_id"])].copy()

    print("\nКэш загружен/построен:")
    print("train_feat_c:", train_feat_c.shape)
    print("valid_known_c:", valid_known_c.shape)
    print("valid_truth_c:", valid_truth_c.shape)
    print("history_prevalid_c:", history_prevalid_c.shape)
    print("clusters:", cluster_map["series_cluster"].value_counts().sort_index().to_dict())
    print("meta:", meta)

    run_lightgbm(
        train_feat=train_feat_c,
        valid_known=valid_known_c,
        valid_truth=valid_truth_c,
        history_prevalid=history_prevalid_c,
        meta=meta,
    )

    print("\nГотово.")
    print("Артефакты:")
    print(" -", CLUSTER_MAP_PATH)
    print(" -", LGBM_BEST_PARAMS_PATH)
    print(" -", LGBM_METRICS_PATH)
    print(" -", LGBM_PRED_PATH)
    print(" -", LGBM_MODEL_PATH)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
