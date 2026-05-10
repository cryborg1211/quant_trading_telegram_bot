"""LSTM baseline – config-driven, Polars IO, vectorized sequence generation."""

from __future__ import annotations

import json
import random
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Dataset

from config.settings import CONFIG

SEED = CONFIG.training.seed
DATA_PATH = CONFIG.paths.alpha360_parquet
MACRO_PATH = CONFIG.paths.macro_parquet
MODEL_DIR = CONFIG.paths.models_dir
SELECTED_FEATURES_PATH = MODEL_DIR / "selected_features.json"
SCALER_ALPHA_PATH = MODEL_DIR / "scaler_alpha.joblib"
SCALER_MACRO_PATH = MODEL_DIR / "scaler_macro.joblib"
MODEL_PATH = MODEL_DIR / "lstm_classification_final.pth"

TARGET_COL = "target_class_5d"
DATE_COL = "date"
TICKER_COL = "ticker"
SPLIT_DATE = pd.Timestamp(CONFIG.training.split_date)

TOP_K_FEATURES = CONFIG.model.lstm_top_k_features
SEQ_LEN = CONFIG.model.lstm_seq_len
HIDDEN_SIZE = CONFIG.model.lstm_hidden_size
NUM_LAYERS = CONFIG.model.lstm_num_layers
DROPOUT = CONFIG.model.lstm_dropout
NUM_CLASSES = CONFIG.model.lstm_num_classes
BATCH_SIZE = CONFIG.training.batch_size
EPOCHS = CONFIG.training.epochs
LEARNING_RATE = CONFIG.training.learning_rate
WEIGHT_DECAY = CONFIG.training.weight_decay
EARLY_STOPPING_PATIENCE = CONFIG.training.early_stopping_patience


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


class AlphaSequenceDataset(Dataset):
    def __init__(self, x_alpha: np.ndarray, x_macro: np.ndarray, y: np.ndarray) -> None:
        self.x_alpha = torch.from_numpy(x_alpha.astype(np.float32, copy=False))
        self.x_macro = torch.from_numpy(x_macro.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_alpha[idx], self.x_macro[idx], self.y[idx]


class QuantLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = TOP_K_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        macro_size: int = 0,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size + macro_size),
            nn.Linear(hidden_size + macro_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x_alpha: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_alpha)
        return self.head(torch.cat([h_n[-1], x_macro], dim=1))


def load_training_frame() -> pd.DataFrame:
    """Fast Polars IO/join; returns Pandas for sklearn/PyTorch compatibility."""
    df = pl.read_parquet(DATA_PATH).with_columns(pl.col(DATE_COL).cast(pl.Date))

    if MACRO_PATH.exists():
        macro_df = pl.read_parquet(MACRO_PATH).with_columns(pl.col(DATE_COL).cast(pl.Date))
        macro_cols = [c for c in macro_df.columns if c != DATE_COL]
        overlapping = [c for c in macro_cols if c in df.columns]
        if overlapping:
            df = df.drop(overlapping)
        df = df.join(macro_df, on=DATE_COL, how="left")

    df = (
        df.drop_nulls(subset=[TARGET_COL, DATE_COL, TICKER_COL])
        .filter(pl.col(TARGET_COL).is_in([0, 1, 2]))
        .with_columns(pl.col(TARGET_COL).cast(pl.Int64))
        .sort([TICKER_COL, DATE_COL])
    )
    return df.to_pandas()


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    macro_cols = [
        c for c in df.columns
        if "macro" in c.lower() or c in {"vix_close", "dxy_close", "vnindex_rsi"}
    ]
    meta_cols = {
        DATE_COL, TICKER_COL, TARGET_COL, "target_1d", "target_3d",
        "target_class_20d", "target_return_5d", "target_return_20d",
        "raw_close", "close",
    }
    alpha_cols = [
        c for c in df.columns
        if c not in meta_cols
        and c not in macro_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(alpha_cols) < TOP_K_FEATURES:
        raise ValueError(f"Only {len(alpha_cols)} alpha columns found; need {TOP_K_FEATURES}.")
    print(f"🔍 Alpha={len(alpha_cols)} Macro={len(macro_cols)}", flush=True)
    return alpha_cols, macro_cols


def clean_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    cleaned = df[columns].replace([np.inf, -np.inf], np.nan)
    medians = cleaned.median(numeric_only=True).fillna(0.0)
    return cleaned.fillna(medians)


def select_alpha_features(train_df: pd.DataFrame, alpha_cols: list[str]) -> list[str]:
    x_train = clean_numeric_frame(train_df, alpha_cols).to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1][:TOP_K_FEATURES]
    selected = [alpha_cols[i] for i in order]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with SELECTED_FEATURES_PATH.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    return selected


def fit_transform_scalers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    macro_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler_alpha = StandardScaler()
    scaler_macro = StandardScaler()

    train_alpha = clean_numeric_frame(train_df, selected_features).to_numpy(dtype=np.float32)
    test_alpha = clean_numeric_frame(test_df, selected_features).to_numpy(dtype=np.float32)

    train_macro = clean_numeric_frame(train_df, macro_cols).to_numpy(dtype=np.float32)
    test_macro = clean_numeric_frame(test_df, macro_cols).to_numpy(dtype=np.float32)
    if not macro_cols:
        train_macro = np.zeros((len(train_df), 0), dtype=np.float32)
        test_macro = np.zeros((len(test_df), 0), dtype=np.float32)

    train_alpha_scaled = scaler_alpha.fit_transform(train_alpha).astype(np.float32)
    test_alpha_scaled = scaler_alpha.transform(test_alpha).astype(np.float32)
    train_macro_scaled = scaler_macro.fit_transform(train_macro).astype(np.float32) if macro_cols else train_macro
    test_macro_scaled = scaler_macro.transform(test_macro).astype(np.float32) if macro_cols else test_macro

    joblib.dump(scaler_alpha, SCALER_ALPHA_PATH)
    joblib.dump(scaler_macro, SCALER_MACRO_PATH)
    return train_alpha_scaled, test_alpha_scaled, train_macro_scaled, test_macro_scaled


def create_rolling_sequences(
    df: pd.DataFrame,
    alpha_values: np.ndarray,
    macro_values: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sequence creation per ticker using sliding_window_view."""
    work = df[[TICKER_COL, DATE_COL, TARGET_COL]].copy().reset_index(drop=True)
    work["_row_pos"] = np.arange(len(work), dtype=np.int64)

    x_alpha_list: list[np.ndarray] = []
    x_macro_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for _, ticker_df in work.groupby(TICKER_COL, sort=False):
        ticker_df = ticker_df.sort_values(DATE_COL)
        positions = ticker_df["_row_pos"].to_numpy(dtype=np.int64)
        targets = ticker_df[TARGET_COL].to_numpy(dtype=np.int64)
        if len(positions) < seq_len:
            continue

        windows = sliding_window_view(positions, seq_len)
        end_positions = positions[seq_len - 1 :]
        x_alpha_list.append(alpha_values[windows])
        x_macro_list.append(macro_values[end_positions])
        y_list.append(targets[seq_len - 1 :])

    if not y_list:
        raise ValueError("No rolling sequences created")

    return (
        np.concatenate(x_alpha_list).astype(np.float32, copy=False),
        np.concatenate(x_macro_list).astype(np.float32, copy=False),
        np.concatenate(y_list).astype(np.int64, copy=False),
    )


def compute_boosted_class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.array([0, 1, 2], dtype=np.int64)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
    print(f"Class Weights: {dict(zip(classes.tolist(), weights.tolist()))}", flush=True)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for x_alpha, x_macro, target in loader:
            x_alpha = x_alpha.to(device)
            x_macro = x_macro.to(device)
            target = target.to(device)

            logits = model(x_alpha, x_macro)
            loss = criterion(logits, target)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * target.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(target.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", labels=[0, 1, 2], zero_division=0))
    return float(avg_loss), macro_f1, all_preds, all_targets


def train_model() -> None:
    seed_everything()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    full_df = load_training_frame()
    alpha_cols, macro_cols = infer_feature_columns(full_df)
    train_df = full_df[full_df[DATE_COL] < SPLIT_DATE].copy().sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    test_df = full_df[full_df[DATE_COL] >= SPLIT_DATE].copy().sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced empty train or test set")

    print(f"Rows | train={len(train_df)} test={len(test_df)}", flush=True)
    print(f"Raw Target Dist | train={dict(Counter(train_df[TARGET_COL]))} test={dict(Counter(test_df[TARGET_COL]))}", flush=True)

    selected_features = select_alpha_features(train_df, alpha_cols)
    print(f"Selected Features: {selected_features}", flush=True)
    print(f"Macro Features: {macro_cols}", flush=True)

    train_alpha_scaled, test_alpha_scaled, train_macro_scaled, test_macro_scaled = fit_transform_scalers(
        train_df, test_df, selected_features, macro_cols,
    )

    x_train_alpha, x_train_macro, y_train = create_rolling_sequences(train_df, train_alpha_scaled, train_macro_scaled)
    x_test_alpha, x_test_macro, y_test = create_rolling_sequences(test_df, test_alpha_scaled, test_macro_scaled)

    print(f"Sequences | train={len(y_train)} test={len(y_test)}", flush=True)
    print(f"Sequence Target Dist | train={dict(Counter(y_train))} test={dict(Counter(y_test))}", flush=True)

    train_loader = DataLoader(
        AlphaSequenceDataset(x_train_alpha, x_train_macro, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        AlphaSequenceDataset(x_test_alpha, x_test_macro, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    class_weights = compute_boosted_class_weights(y_train, device)
    model = QuantLSTM(
        input_size=len(selected_features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        macro_size=len(macro_cols),
        dropout=DROPOUT,
    ).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_macro_f1 = -np.inf
    patience_counter = 0
    best_targets: list[int] = []
    best_preds: list[int] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x_alpha, x_macro, target in train_loader:
            x_alpha = x_alpha.to(device)
            x_macro = x_macro.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_alpha, x_macro)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * target.size(0)

        avg_train_loss = train_loss / max(1, len(y_train))
        val_loss, macro_f1, preds, targets = evaluate(model, test_loader, criterion, device)
        scheduler.step(macro_f1)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Macro F1: {macro_f1:.6f} | "
            f"LR: {lr:.8f} | Pred Dist: {dict(Counter(preds))}",
            flush=True,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            best_targets = targets
            best_preds = preds
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "selected_features": selected_features,
                    "macro_features": macro_cols,
                    "input_size": len(selected_features),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "macro_size": len(macro_cols),
                    "dropout": DROPOUT,
                    "seq_len": SEQ_LEN,
                    "split_date": str(SPLIT_DATE.date()),
                    "best_macro_f1": float(best_macro_f1),
                    "class_weights": class_weights.detach().cpu().numpy().tolist(),
                },
                MODEL_PATH,
            )
            print(f"Saved best model: {MODEL_PATH} | Macro F1: {best_macro_f1:.6f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} | Best Macro F1: {best_macro_f1:.6f}", flush=True)
                break

    if best_targets and best_preds:
        print("Final Best Validation Report", flush=True)
        print(
            classification_report(
                best_targets,
                best_preds,
                labels=[0, 1, 2],
                target_names=["DOWN", "SIDEWAYS", "UP"],
                zero_division=0,
            ),
            flush=True,
        )


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")
    train_model()