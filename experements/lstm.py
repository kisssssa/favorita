import os
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


N_JOBS = min(6, os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)

torch.set_num_threads(N_JOBS)

TRAIN_PATH = "train.csv"
ITEMS_PATH = "items.csv"
STORES_PATH = "stores.csv"
TRANSACTIONS_PATH = "transactions.csv"
OIL_PATH = "oil.csv"
HOLIDAYS_PATH = "holidays_events.csv"

OUTPUT_DIR = Path("outputs_lstm_raw_eval_gpu_light")
CACHE_DIR = OUTPUT_DIR / "cache"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_START_DATE = "2016-01-01"

HORIZON = 28
INNER_EVAL_DAYS = 28
SEQ_LEN = 28

BATCH_SIZE = 8192
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.10

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_FEAT_PATH = CACHE_DIR / "train_feat.parquet"
VALID_KNOWN_PATH = CACHE_DIR / "valid_known.parquet"
VALID_TRUTH_PATH = CACHE_DIR / "valid_truth.parquet"
HISTORY_PREVALID_PATH = CACHE_DIR / "history_prevalid.parquet"
META_PATH = CACHE_DIR / "meta.json"
VOCAB_PATH = CACHE_DIR / "vocab.json"

TRAIN_SEQ_PATH = CACHE_DIR / "train_seq.npy"
TRAIN_NUM_PATH = CACHE_DIR / "train_num.npy"
TRAIN_CAT_PATH = CACHE_DIR / "train_cat.npy"
TRAIN_Y_PATH = CACHE_DIR / "train_y.npy"

VAL_SEQ_PATH = CACHE_DIR / "val_seq.npy"
VAL_NUM_PATH = CACHE_DIR / "val_num.npy"
VAL_CAT_PATH = CACHE_DIR / "val_cat.npy"
VAL_Y_PATH = CACHE_DIR / "val_y.npy"

MODEL_PATH = OUTPUT_DIR / "lstm_model.pt"
METRICS_PATH = OUTPUT_DIR / "lstm_metrics_h28.json"
PRED_PATH = OUTPUT_DIR / "lstm_valid_predictions_h28.csv"

CAT_COLS = [
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

NUM_COLS = [
    "onpromotion",
    "transactions",
    "dcoilwtico",
    "is_holiday_event",
    "dayofweek",
    "month",
    "day",
    "is_weekend",
]

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_unique_id(df: pd.DataFrame) -> pd.Series:
    return df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_clip_forecast(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    pred = np.clip(pred, 0.0, None)
    return pred

def inverse_log_target(pred_log: np.ndarray) -> np.ndarray:
    return safe_clip_forecast(np.expm1(pred_log))

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

class SeqDataset(Dataset):
    def __init__(self, seqs, nums, cats, y):
        self.seqs = seqs
        self.nums = nums
        self.cats = cats
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.seqs[idx]),
            torch.from_numpy(self.nums[idx]),
            torch.from_numpy(self.cats[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

class GlobalLSTM(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_numeric: int):
        super().__init__()

        self.emb_layers = nn.ModuleList([
            nn.Embedding(card + 1, min(16, max(4, int(np.sqrt(card + 1)) + 1)))
            for card in cat_cardinalities
        ])
        emb_total = sum(emb.embedding_dim for emb in self.emb_layers)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )

        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + num_numeric + emb_total, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, seq_x, num_x, cat_x):
        _, (h_n, _) = self.lstm(seq_x)
        h = h_n[-1]

        embs = []
        for i, emb in enumerate(self.emb_layers):
            embs.append(emb(cat_x[:, i]))
        emb_cat = torch.cat(embs, dim=1)

        x = torch.cat([h, num_x, emb_cat], dim=1)
        out = self.mlp(x).squeeze(1)
        return out

def build_and_cache_data(force_rebuild: bool = False):
    if (
        not force_rebuild
        and TRAIN_FEAT_PATH.exists()
        and VALID_KNOWN_PATH.exists()
        and VALID_TRUTH_PATH.exists()
        and HISTORY_PREVALID_PATH.exists()
        and META_PATH.exists()
        and VOCAB_PATH.exists()
    ):
        print("Загрузка кэша ...")
        train_feat = pd.read_parquet(TRAIN_FEAT_PATH)
        valid_known = pd.read_parquet(VALID_KNOWN_PATH)
        valid_truth = pd.read_parquet(VALID_TRUTH_PATH)
        history_prevalid = pd.read_parquet(HISTORY_PREVALID_PATH)
        meta = load_json(META_PATH)
        vocab = load_json(VOCAB_PATH)
        return train_feat, valid_known, valid_truth, history_prevalid, meta, vocab

    print("Чтение train.csv ...")
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        parse_dates=["date"],
        low_memory=False,
    )

    if TRAIN_START_DATE is not None:
        train = train[train["date"] >= pd.Timestamp(TRAIN_START_DATE)].copy()

    train["store_nbr"] = train["store_nbr"].astype("int16")
    train["item_nbr"] = train["item_nbr"].astype("int32")
    train["unit_sales"] = train["unit_sales"].astype("float32")

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

    items["item_nbr"] = items["item_nbr"].astype("int32")
    stores["store_nbr"] = stores["store_nbr"].astype("int16")
    transactions["store_nbr"] = transactions["store_nbr"].astype("int16")
    transactions["transactions"] = pd.to_numeric(
        transactions["transactions"], errors="coerce"
    ).fillna(0).astype("float32")

    for col in ["family"]:
        items[col] = items[col].astype("category")
    items["class"] = pd.to_numeric(items["class"], errors="coerce").fillna(-1).astype("int32")
    items["perishable"] = pd.to_numeric(items["perishable"], errors="coerce").fillna(0).astype("int8")

    for col in ["city", "state", "type"]:
        stores[col] = stores[col].astype("category")
    stores["cluster"] = pd.to_numeric(stores["cluster"], errors="coerce").fillna(-1).astype("int16")

    holidays_flag = holidays.drop_duplicates().copy()
    holidays_flag["is_holiday_event"] = np.int8(1)

    print("Merge ...")
    df = train.merge(transactions, on=["date", "store_nbr"], how="left")
    del train, transactions

    df = df.merge(oil, on="date", how="left")
    del oil

    df = df.merge(items, on="item_nbr", how="left")
    del items

    df = df.merge(stores, on="store_nbr", how="left")
    del stores

    df = df.merge(holidays_flag, on="date", how="left")
    del holidays_flag

    df["unique_id"] = make_unique_id(df)
    df["unit_sales_nonneg"] = df["unit_sales"].clip(lower=0).astype("float32")
    df["log_target"] = np.log1p(df["unit_sales_nonneg"]).astype("float32")

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

    max_date = df["date"].max()
    valid_start = max_date - pd.Timedelta(days=HORIZON - 1)
    inner_eval_start = valid_start - pd.Timedelta(days=INNER_EVAL_DAYS)

    mask_prevalid = df["date"] < valid_start
    mask_train = df["date"] < valid_start
    mask_valid = df["date"] >= valid_start

    history_prevalid = df.loc[mask_prevalid, ["unique_id", "date", "log_target"]].copy()
    train_feat = df.loc[mask_train].copy()

    valid_known_cols = ["unique_id", "date"] + CAT_COLS + NUM_COLS
    valid_known = df.loc[mask_valid, valid_known_cols].copy()

    valid_truth = df.loc[mask_valid, ["unique_id", "date", "item_nbr", "unit_sales_nonneg", "perishable"]].copy()
    valid_truth = valid_truth.rename(columns={"unit_sales_nonneg": "y_true"})
    valid_truth["weight"] = np.where(valid_truth["perishable"] == 1, 1.25, 1.0).astype("float32")
    valid_truth = valid_truth[["unique_id", "date", "item_nbr", "y_true", "weight"]].copy()

    vocab = {}
    for col in CAT_COLS:
        vals = train_feat[col].astype(str).fillna("__NA__").unique().tolist()
        mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}
        vocab[col] = mapping
        train_feat[col] = train_feat[col].astype(str).map(mapping).fillna(0).astype("int32")
        valid_known[col] = valid_known[col].astype(str).map(mapping).fillna(0).astype("int32")

    for col in NUM_COLS:
        train_feat[col] = train_feat[col].astype("float32")
        valid_known[col] = valid_known[col].astype("float32")

    meta = {
        "valid_start": str(valid_start.date()),
        "inner_eval_start": str(inner_eval_start.date()),
        "max_date": str(max_date.date()),
        "train_start_date": TRAIN_START_DATE,
        "horizon": HORIZON,
        "seq_len": SEQ_LEN,
    }

    print("Сохранение кэша ...")
    train_feat.to_parquet(TRAIN_FEAT_PATH, index=False)
    valid_known.to_parquet(VALID_KNOWN_PATH, index=False)
    valid_truth.to_parquet(VALID_TRUTH_PATH, index=False)
    history_prevalid.to_parquet(HISTORY_PREVALID_PATH, index=False)
    save_json(meta, META_PATH)
    save_json(vocab, VOCAB_PATH)

    return train_feat, valid_known, valid_truth, history_prevalid, meta, vocab

def build_samples(df: pd.DataFrame, min_seq_len: int = 28):
    seqs, nums, cats, ys = [], [], [], []

    for _, g in df.groupby("unique_id", sort=False):
        g = g.sort_values("date")
        y_hist = g["log_target"].to_numpy(dtype=np.float32)
        num_arr = g[NUM_COLS].to_numpy(dtype=np.float32)
        cat_arr = g[CAT_COLS].to_numpy(dtype=np.int64)

        if len(g) <= min_seq_len:
            continue

        for i in range(min_seq_len, len(g)):
            seqs.append(y_hist[i - min_seq_len:i][:, None].astype(np.float32))
            nums.append(num_arr[i].astype(np.float32))
            cats.append(cat_arr[i].astype(np.int64))
            ys.append(np.float32(y_hist[i]))

    return (
        np.stack(seqs).astype(np.float32),
        np.stack(nums).astype(np.float32),
        np.stack(cats).astype(np.int64),
        np.asarray(ys, dtype=np.float32),
    )

def build_or_load_sample_arrays(train_feat: pd.DataFrame, meta: dict):
    if (
        TRAIN_SEQ_PATH.exists()
        and TRAIN_NUM_PATH.exists()
        and TRAIN_CAT_PATH.exists()
        and TRAIN_Y_PATH.exists()
        and VAL_SEQ_PATH.exists()
        and VAL_NUM_PATH.exists()
        and VAL_CAT_PATH.exists()
        and VAL_Y_PATH.exists()
    ):
        print("Загрузка sample arrays из кэша ...")
        tr_seq = np.load(TRAIN_SEQ_PATH, mmap_mode="r")
        tr_num = np.load(TRAIN_NUM_PATH, mmap_mode="r")
        tr_cat = np.load(TRAIN_CAT_PATH, mmap_mode="r")
        tr_y = np.load(TRAIN_Y_PATH, mmap_mode="r")

        va_seq = np.load(VAL_SEQ_PATH, mmap_mode="r")
        va_num = np.load(VAL_NUM_PATH, mmap_mode="r")
        va_cat = np.load(VAL_CAT_PATH, mmap_mode="r")
        va_y = np.load(VAL_Y_PATH, mmap_mode="r")
        return tr_seq, tr_num, tr_cat, tr_y, va_seq, va_num, va_cat, va_y

    inner_eval_start = pd.Timestamp(meta["inner_eval_start"])
    fit_df = train_feat[train_feat["date"] < inner_eval_start].copy()
    eval_df = train_feat[train_feat["date"] >= inner_eval_start].copy()

    print("Построение train samples ...")
    tr_seq, tr_num, tr_cat, tr_y = build_samples(fit_df, min_seq_len=SEQ_LEN)
    print("train samples:", tr_seq.shape, tr_num.shape, tr_cat.shape, tr_y.shape)

    print("Построение val samples ...")
    va_seq, va_num, va_cat, va_y = build_samples(eval_df, min_seq_len=SEQ_LEN)
    print("val samples:", va_seq.shape, va_num.shape, va_cat.shape, va_y.shape)

    np.save(TRAIN_SEQ_PATH, tr_seq)
    np.save(TRAIN_NUM_PATH, tr_num)
    np.save(TRAIN_CAT_PATH, tr_cat)
    np.save(TRAIN_Y_PATH, tr_y)

    np.save(VAL_SEQ_PATH, va_seq)
    np.save(VAL_NUM_PATH, va_num)
    np.save(VAL_CAT_PATH, va_cat)
    np.save(VAL_Y_PATH, va_y)

    return tr_seq, tr_num, tr_cat, tr_y, va_seq, va_num, va_cat, va_y

def train_model(train_ds: Dataset, val_ds: Dataset, cat_cardinalities: List[int]):
    pin_memory = DEVICE == "cuda"

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=0,
    )

    model = GlobalLSTM(cat_cardinalities=cat_cardinalities, num_numeric=len(NUM_COLS)).to(DEVICE)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 3
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for seq_x, num_x, cat_x, y in train_dl:
            seq_x = seq_x.to(DEVICE, non_blocking=True)
            num_x = num_x.to(DEVICE, non_blocking=True)
            cat_x = cat_x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(seq_x, num_x, cat_x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            n_train += len(y)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for seq_x, num_x, cat_x, y in val_dl:
                seq_x = seq_x.to(DEVICE, non_blocking=True)
                num_x = num_x.to(DEVICE, non_blocking=True)
                cat_x = cat_x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                pred = model(seq_x, num_x, cat_x)
                loss = criterion(pred, y)

                val_loss += loss.item() * len(y)
                n_val += len(y)

        train_loss /= max(n_train, 1)
        val_loss /= max(n_val, 1)

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return model

def build_history_dict(history_prevalid: pd.DataFrame):
    return (
        history_prevalid.sort_values(["unique_id", "date"])
        .groupby("unique_id")["log_target"]
        .apply(list)
        .to_dict()
    )

def recursive_predict_valid(model, valid_known, valid_truth):
    model.eval()
    history_prevalid = pd.read_parquet(HISTORY_PREVALID_PATH)
    history = build_history_dict(history_prevalid)

    pred_parts = []
    valid_dates = sorted(valid_known["date"].unique())

    with torch.no_grad():
        for current_date in valid_dates:
            day_df = valid_known[valid_known["date"] == current_date].copy()

            seqs = []
            nums = []
            cats = []
            keep_idx = []

            for idx, row in day_df.iterrows():
                uid = row["unique_id"]
                hist = history.get(uid, [])
                if len(hist) < SEQ_LEN:
                    continue

                seq = np.asarray(hist[-SEQ_LEN:], dtype=np.float32)[:, None]
                num_x = row[NUM_COLS].to_numpy(dtype=np.float32)
                cat_x = row[CAT_COLS].to_numpy(dtype=np.int64)

                seqs.append(seq)
                nums.append(num_x)
                cats.append(cat_x)
                keep_idx.append(idx)

            if len(keep_idx) > 0:
                seq_x = torch.tensor(np.stack(seqs), dtype=torch.float32, device=DEVICE)
                num_x = torch.tensor(np.stack(nums), dtype=torch.float32, device=DEVICE)
                cat_x = torch.tensor(np.stack(cats), dtype=torch.long, device=DEVICE)

                pred_log = model(seq_x, num_x, cat_x).detach().cpu().numpy()
                pred = inverse_log_target(pred_log)

                sub = day_df.loc[keep_idx, ["unique_id", "date"]].copy()
                sub["y_pred"] = pred
                pred_parts.append(sub)

                for uid, plog in zip(sub["unique_id"].values, pred_log):
                    history.setdefault(uid, []).append(float(plog))

    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame(columns=["unique_id", "date", "y_pred"])
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

def main():
    seed_everything(RANDOM_STATE)

    print(f"DEVICE = {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")
    print(f"N_JOBS = {N_JOBS}")
    print(f"HORIZON = {HORIZON}")
    print(f"SEQ_LEN = {SEQ_LEN}")
    print(f"TRAIN_START_DATE = {TRAIN_START_DATE}")

    train_feat, valid_known, valid_truth, history_prevalid, meta, vocab = build_and_cache_data(force_rebuild=False)

    print("train_feat:", train_feat.shape)
    print("valid_known:", valid_known.shape)
    print("valid_truth:", valid_truth.shape)
    print("history_prevalid:", history_prevalid.shape)
    print("meta:", meta)

    tr_seq, tr_num, tr_cat, tr_y, va_seq, va_num, va_cat, va_y = build_or_load_sample_arrays(train_feat, meta)

    train_ds = SeqDataset(tr_seq, tr_num, tr_cat, tr_y)
    val_ds = SeqDataset(va_seq, va_num, va_cat, va_y)

    cat_cardinalities = [max(vocab[col].values()) if len(vocab[col]) > 0 else 0 for col in CAT_COLS]

    model = train_model(train_ds, val_ds, cat_cardinalities=cat_cardinalities)
    torch.save(model.state_dict(), MODEL_PATH)

    print("Рекурсивный прогноз holdout ...")
    pred_df, metrics = recursive_predict_valid(model, valid_known, valid_truth)

    pred_df.to_csv(PRED_PATH, index=False)
    save_json(metrics, METRICS_PATH)

    print("\nLSTM metrics:")
    for k, v in metrics.items():
        print(k, v)

    print("\nАртефакты:")
    print(" -", MODEL_PATH)
    print(" -", METRICS_PATH)
    print(" -", PRED_PATH)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
