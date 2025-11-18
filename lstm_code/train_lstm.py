#!/usr/bin/env python3
"""
train_spy_lstm_fixed.py

Usage example:
python train_spy_lstm_fixed.py \
  --root_dir /N/u/akulshar/Quartz/Documents/new_data \
  --spy_info /N/u/akulshar/Quartz/Documents/spy_targets.csv \
  --cache_dir /N/u/akulshar/Quartz/Documents/LSTM/cache \
  --train_years 2020 2021 2022 \
  --test_years 2024 \
  --last_n 120 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --hidden_size 128 \
  --num_layers 2 \
  --dropout 0.3 \
  --out_dir /N/u/akulshar/Quartz/Documents/LSTM/results/run_new_2 \
  --use_cache \
  --device cuda
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# ---------------------------
# Dataset
# ---------------------------
class SPYIntradayDataset(Dataset):
    def __init__(self, root_dir, spy_info_df, tickers,
                 years=None, normalize=True, last_n=None,
                 cache_dir=None, split_name="train"):
        self.root_dir = root_dir
        self.spy_info = spy_info_df.copy()
        self.tickers = list(tickers)
        self.normalize = normalize
        self.last_n = last_n

        if "label_930_1130" not in self.spy_info.columns and \
           all(c in self.spy_info.columns for c in ["price_11:30:00", "price_13:30:00"]):
            self.spy_info["label_930_1130"] = (
                self.spy_info["price_13:30:00"] > self.spy_info["price_11:30:00"]
            ).astype(int)
        if "label_1130_1330" not in self.spy_info.columns and \
           all(c in self.spy_info.columns for c in ["price_13:30:00", "price_15:30:00"]):
            self.spy_info["label_1130_1330"] = (
                self.spy_info["price_15:30:00"] > self.spy_info["price_13:30:00"]
            ).astype(int)

        if years is not None and len(years) > 0:
            self.spy_info = self.spy_info[self.spy_info["date"].dt.year.isin(years)].reset_index(drop=True)

        self.samples = []
        for _, row in self.spy_info.iterrows():
            date = row["date"]
            self.samples.append((date, "930_1130"))
            self.samples.append((date, "1130_1330"))

        self.cache_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir) / split_name
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, date_obj, segment):
        fname = f"{date_obj.strftime('%Y-%m-%d')}_{segment}.pt"
        return self.cache_dir / fname

    def _load_cached(self, date_obj, segment):
        if not self.cache_dir:
            return None
        p = self._cache_path(date_obj, segment)
        if p.exists():
            obj = torch.load(p)
            return obj["X"], obj["y"]
        return None

    def _save_cached(self, date_obj, segment, X, y):
        if not self.cache_dir:
            return
        p = self._cache_path(date_obj, segment)
        torch.save({"X": X, "y": y}, p)

    def __getitem__(self, idx):
        date_obj, segment = self.samples[idx]

        cached = self._load_cached(date_obj, segment)
        if cached is not None:
            return cached

        date_str = date_obj.strftime("%Y-%m-%d")
        year = date_obj.year
        month = str(date_obj.month).zfill(2)
        filename = f"taq_resampled_{date_str}.csv.gz"
        path = os.path.join(self.root_dir, str(year), month, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        df = pd.read_csv(path, compression="gzip", parse_dates=["datetime"]).set_index("datetime")
        df = df.reindex(columns=self.tickers).ffill().fillna(0)
        df = df.resample("4s").last().ffill().fillna(0)

        if segment == "930_1130":
            features = df.between_time("09:30", "11:30")
            label = int(self.spy_info.loc[self.spy_info["date"] == date_obj, "label_930_1130"].item())
        else:
            features = df.between_time("11:30", "13:30")
            label = int(self.spy_info.loc[self.spy_info["date"] == date_obj, "label_1130_1330"].item())

        if self.last_n is not None:
            features = features.tail(self.last_n)

        arr = features.to_numpy(dtype=np.float32)

        if self.normalize:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            arr = (arr - mean) / std

        X_tensor = torch.tensor(arr, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.float32)  # float for BCE
        self._save_cached(date_obj, segment, X_tensor, y_tensor)
        return X_tensor, y_tensor

# model

class LSTMClassifier(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.out_sz = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(self.out_sz, 1)  # single logit for BCE

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        final_hidden = hn[-1]
        logit = self.fc(final_hidden).squeeze(-1)
        return logit

# utilities

def collate_fn(batch):
    Xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.stack([item[1] for item in batch], dim=0)
    return Xs, ys

def compute_all_metrics(y_true, y_pred_labels, y_pred_probs):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred_labels)
    metrics["precision"] = precision_score(y_true, y_pred_labels, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred_labels, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred_labels, zero_division=0)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_pred_probs)
    except Exception:
        metrics["pr_auc"] = float("nan")
    try:
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred_labels)
    except Exception:
        metrics["mcc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = int(tn), int(fp), int(fn), int(tp)
    return metrics

# evaluation

def evaluate(model, loader, device):
    model.eval()
    ys_all, probs_all, preds_all = [], [], []
    loss_total = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            loss_total += loss.item() * X.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            ys_all.append(y.cpu().numpy())
            probs_all.append(probs)
            preds_all.append(preds)
    if len(ys_all) == 0:
        return {}, np.array([]), np.array([])
    y_true = np.concatenate(ys_all)
    y_probs = np.concatenate(probs_all)
    y_preds = np.concatenate(preds_all)
    avg_loss = loss_total / len(y_true)
    metrics = compute_all_metrics(y_true, y_preds, y_probs)
    metrics["loss"] = float(avg_loss)
    return metrics, y_true, y_probs

# training function

def train_and_evaluate(args):
    spy = pd.read_csv(args.spy_info)
    spy["date"] = pd.to_datetime(spy["date"])

    if args.tickers_path:
        tickers = pd.read_csv(args.tickers_path, header=None).iloc[:,0].tolist()
    else:
        sample_file = None
        for y in sorted(os.listdir(args.root_dir)):
            year_path = os.path.join(args.root_dir, y)
            if not os.path.isdir(year_path): continue
            for m in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, m)
                if not os.path.isdir(month_path): continue
                for fname in sorted(os.listdir(month_path)):
                    if fname.endswith(".csv.gz"):
                        sample_file = os.path.join(month_path, fname)
                        break
                if sample_file: break
            if sample_file: break
        if not sample_file:
            raise FileNotFoundError(f"No sample .csv.gz found under {args.root_dir}")
        cols = pd.read_csv(sample_file, compression="gzip", nrows=0).columns
        tickers = [c for c in cols if c != "datetime"]
    print(f"Using {len(tickers)} tickers")

    train_ds = SPYIntradayDataset(args.root_dir, spy, tickers, years=args.train_years,
                                  normalize=args.normalize, last_n=args.last_n,
                                  cache_dir=args.cache_dir if args.use_cache else None, split_name="train")
    test_ds = SPYIntradayDataset(args.root_dir, spy, tickers, years=args.test_years,
                                 normalize=args.normalize, last_n=args.last_n,
                                 cache_dir=args.cache_dir if args.use_cache else None, split_name="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(args.device=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(args.device=="cuda"))

    device = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    X0, _ = train_ds[0]
    seq_len, num_features = X0.shape
    print(f"Seq len = {seq_len}, num_features = {num_features}")
    model = LSTMClassifier(num_features=num_features, hidden_size=args.hidden_size,
                           num_layers=args.num_layers, dropout=args.dropout,
                           bidirectional=args.bidirectional).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_test_acc = 0.0
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            running_loss += loss.item() * X.size(0)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)
            pbar.set_postfix({"loss": f"{running_loss/running_total:.4f}", "acc": f"{running_correct/running_total:.4f}"})

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        test_metrics, y_test_true, y_test_probs = evaluate(model, test_loader, device)
        test_loss = test_metrics.get("loss", float("nan"))
        test_acc = test_metrics.get("accuracy", float("nan"))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch} summary: Train loss {train_loss:.4f} acc {train_acc:.4f} | Test loss {test_loss:.4f} acc {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model")

        if not np.isnan(test_loss):
            scheduler.step(test_loss)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_train_metrics, _, _ = evaluate(model, train_loader, device)
    final_test_metrics, y_test_true, y_test_probs = evaluate(model, test_loader, device)

    results = {"train": final_train_metrics, "test": final_test_metrics, "args": vars(args)}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame([results["train"], results["test"]]).to_csv(out_dir / "results_table.csv", index=False)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    # --- Plot Train vs Test Accuracy ---
    plt.figure(figsize=(6,4))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "train_vs_test_acc.png")
    plt.close()

    # --- Plot Train vs Test Loss ---
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "train_vs_test_loss.png")
    plt.close()

    # --- ROC Curve ---
    if len(y_test_true) > 0:
        fpr, tpr, _ = roc_curve(y_test_true, y_test_probs)
        auc_score = roc_auc_score(y_test_true, y_test_probs)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})", linewidth=2)
        plt.plot([0,1], [0,1], '--', color='gray', label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png")
        plt.close()

    print("Done. Results saved to", out_dir)
    return results

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--spy_info", type=str, required=True)
    p.add_argument("--tickers_path", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--train_years", type=int, nargs="+", required=True)
    p.add_argument("--test_years", type=int, nargs="+", required=True)
    p.add_argument("--last_n", type=int, default=None)
    p.add_argument("--normalize", type=bool, default=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cache", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    train_and_evaluate(args)
