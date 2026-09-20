import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| cuda name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")

KAGGLE_INPUT_ROOT = "/kaggle/input"
INPUT_DIR = None
if os.path.isdir(KAGGLE_INPUT_ROOT):
    for dirpath, dirnames, filenames in os.walk(KAGGLE_INPUT_ROOT):
        if "train.csv" in filenames and "test.csv" in filenames:
            INPUT_DIR = dirpath
            break
if INPUT_DIR is None:
    # fall back for local smoke-testing outside Kaggle
    INPUT_DIR = "data/raw"
print("using INPUT_DIR:", INPUT_DIR)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5
N_FOLDS_FOR_VALIDATION = 1  # single held-out group-fold for a quick honest sanity check

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
train["text"] = train["text"].fillna("")
test["text"] = test["text"].fillna("")
print("train:", train.shape, "test:", test.shape)

# template-prefix groups so validation never sees a near-duplicate sibling of a training row
groups = train["text"].str[:50].values

y_log = np.log1p(train["listPrice"].values).astype(np.float32)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class PriceTextDataset(Dataset):
    def __init__(self, texts, targets=None):
        self.texts = list(texts)
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx], truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class PriceRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


def mae_raw(y_true_raw, y_pred_log):
    pred_raw = np.clip(np.expm1(y_pred_log), 0, None)
    return float(np.mean(np.abs(y_true_raw - pred_raw)))


def train_one_model(train_idx, val_idx, texts_all, y_all_log, y_all_raw, epochs, tag):
    train_ds = PriceTextDataset(texts_all.iloc[train_idx].values, y_all_log[train_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = PriceRegressor(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps),
                                                 num_training_steps=total_steps)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for step, batch in enumerate(train_dl):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            pred = model(input_ids, attention_mask)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"[{tag}] epoch {epoch + 1}/{epochs} mean SmoothL1 loss: {running / len(train_dl):.4f}")

    if val_idx is not None and len(val_idx) > 0:
        model.eval()
        val_ds = PriceTextDataset(texts_all.iloc[val_idx].values)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        preds = []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                pred = model(input_ids, attention_mask)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds)
        score = mae_raw(train["listPrice"].values[val_idx], preds)
        print(f"[{tag}] GroupKFold holdout MAE (raw scale): {score:,.2f}")

    return model


# --- honest single-fold validation (GroupKFold on template-prefix groups) ---
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train["text"], groups=groups))
tr_idx, va_idx = splits[0]
print(f"validation fold: {len(tr_idx)} train / {len(va_idx)} val rows")
_ = train_one_model(tr_idx, va_idx, train["text"], y_log, train["listPrice"].values,
                     epochs=EPOCHS, tag="validation-fold")

# --- final model: refit on 100% of train for the submission ---
full_idx = np.arange(len(train))
final_model = train_one_model(full_idx, None, train["text"], y_log, train["listPrice"].values,
                               epochs=EPOCHS, tag="final")

final_model.eval()
test_ds = PriceTextDataset(test["text"].values)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_preds = []
with torch.no_grad():
    for batch in test_dl:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pred = final_model(input_ids, attention_mask)
        test_preds.append(pred.cpu().numpy())
test_preds = np.concatenate(test_preds)
test_preds_raw = np.clip(np.expm1(test_preds), 0, None)

submission = pd.DataFrame({"id": test["id"], "listPrice": test_preds_raw})
submission.to_csv("submission.csv", index=False, float_format="%.6f", lineterminator="\n")
print("wrote submission.csv", submission.shape)
print(submission["listPrice"].describe())

torch.save(final_model.state_dict(), "distilbert_price_model.pt")
print("saved model weights")
