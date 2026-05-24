from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from collections import Counter

from config import (
    CLASS_IDS,
    MAX_TEXT_LEN,
    NUM_BINS,
    NUM_SLOTS,
    ROW_CODES_PATH,
    SPLIT_XLS_PATH,
)
from dataset import (
    TableSequenceDataset,
    build_charset,
    collate_table_batch,
)
from metrics import classification_metrics, confusion_matrix_df
from model import TableBiLSTMClassifier
from preprocessing import build_dataset_splits


OUTPUT_DIR = Path("artifacts/table_cls")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 3e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0

CHAR_EMB_DIM = 64
CHAR_HIDDEN_DIM = 64
STEP_HIDDEN_DIM = 128
TABLE_HIDDEN_DIM = 128
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Переносит все tensor-поля батча на device.
    Строковые поля вроде sample_ids оставляет как есть.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Returns:
        avg_loss, avg_acc
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits = model(batch)                           # [B, C]
        loss = criterion(logits, batch["labels"])       # scalar

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = batch["labels"].size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == batch["labels"]).sum().item()
        total_count += batch_size

        running_loss = total_loss / max(total_count, 1)
        running_acc = total_correct / max(total_count, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """
    Returns:
        avg_loss, avg_acc, preds, targets
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds: list[int] = []
    all_targets: list[int] = []

    pbar = tqdm(loader, desc="valid", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        logits = model(batch)
        loss = criterion(logits, batch["labels"])

        preds = logits.argmax(dim=1)
        batch_size = batch["labels"].size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == batch["labels"]).sum().item()
        total_count += batch_size

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(batch["labels"].cpu().tolist())

        running_loss = total_loss / max(total_count, 1)
        running_acc = total_correct / max(total_count, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc, all_preds, all_targets


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_training() -> None:
    """
    Полный training pipeline.

    1. Загружает train/val/test samples.
    2. Строит charset по train.
    3. Создает Dataset/DataLoader.
    4. Инициализирует model/optimizer/scheduler.
    5. Тренирует модель и сохраняет лучший checkpoint.
    6. Сохраняет vocab/config/history/test metrics.
    """
    set_seed(SEED)

    print(f"Device: {DEVICE}")

    train_samples, val_samples, test_samples = build_dataset_splits(
        split_xls_path=SPLIT_XLS_PATH,
        row_codes_path=ROW_CODES_PATH,
        num_bins=NUM_BINS,
        num_slots=NUM_SLOTS,
    )

    print(
        f"Samples: train={len(train_samples)} "
        f"val={len(val_samples)} "
        f"test={len(test_samples)}"
    )

    if len(train_samples) == 0:
        raise ValueError("Train split is empty")

    itos, stoi = build_charset(train_samples)

    train_ds = TableSequenceDataset(
        samples=train_samples,
        stoi=stoi,
        max_text_len=MAX_TEXT_LEN,
    )
    val_ds = TableSequenceDataset(
        samples=val_samples,
        stoi=stoi,
        max_text_len=MAX_TEXT_LEN,
    )
    test_ds = TableSequenceDataset(
        samples=test_samples,
        stoi=stoi,
        max_text_len=MAX_TEXT_LEN,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_table_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_table_batch,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_table_batch,
        num_workers=0,
    )

    num_numeric_features = train_ds.num_numeric_features
    pad_idx = stoi["<PAD>"] if "<PAD>" in stoi else stoi.get("[PAD]", 0)

    model = TableBiLSTMClassifier(
        vocab_size=len(stoi),
        num_numeric_features=num_numeric_features,
        num_classes=len(CLASS_IDS),
        num_slots=NUM_SLOTS,
        step_hidden_dim=STEP_HIDDEN_DIM,
        table_hidden_dim=TABLE_HIDDEN_DIM,
        char_emb_dim=CHAR_EMB_DIM,
        char_hidden_dim=CHAR_HIDDEN_DIM,
        pad_idx=pad_idx,
        dropout=DROPOUT,
    ).to(DEVICE)


    # посчитать частоты классов по train_samples
    label_counts = Counter(s.class_id for s in train_samples)
    # гарантируем порядок по CLASS_IDS
    counts = torch.tensor([label_counts[c] for c in CLASS_IDS], dtype=torch.float32)

    # инверсия частот: редкий класс → больший вес
    weights = 1.0 / counts
    # нормализуем, чтобы средний вес ≈ 1
    weights = weights / weights.mean()

    print(f"Веса классов в функции потерь: {list(weights)}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_metric = -1.0
    best_model_path = OUTPUT_DIR / "best_model.pt"
    history: list[dict] = []

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
        )

        val_loss, val_acc, val_preds, val_targets = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        val_metrics = classification_metrics(val_targets, val_preds)
        val_macro_f1 = float(val_metrics["macro_f1"])

        scheduler.step(val_macro_f1)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_macro_f1 > best_val_metric:
            best_val_metric = val_macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model was not saved: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, test_acc, test_preds, test_targets = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
    )
    test_metrics = classification_metrics(test_targets, test_preds)
    cm_df = confusion_matrix_df(test_targets, test_preds, CLASS_IDS)
    cm_path = OUTPUT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    print(
        f"TEST | loss={test_loss:.4f} acc={test_acc:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}"
    )

    vocab_path = OUTPUT_DIR / "vocab.json"
    config_path = OUTPUT_DIR / "train_config.json"
    history_path = OUTPUT_DIR / "history.json"
    test_metrics_path = OUTPUT_DIR / "test_metrics.json"

    save_json(
        {
            "itos": itos,
            "stoi": stoi,
        },
        vocab_path,
    )

    save_json(
        {
            "seed": SEED,
            "device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "max_text_len": MAX_TEXT_LEN,
            "num_bins": NUM_BINS,
            "num_slots": NUM_SLOTS,
            "class_ids": CLASS_IDS,
            "vocab_size": len(stoi),
            "num_numeric_features": num_numeric_features,
            "char_emb_dim": CHAR_EMB_DIM,
            "char_hidden_dim": CHAR_HIDDEN_DIM,
            "step_hidden_dim": STEP_HIDDEN_DIM,
            "table_hidden_dim": TABLE_HIDDEN_DIM,
            "dropout": DROPOUT,
            "split_xls_path": SPLIT_XLS_PATH,
            "row_codes_path": ROW_CODES_PATH,
            "best_model_path": str(best_model_path),
        },
        config_path,
    )

    save_json({"history": history}, history_path)
    save_json(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            **test_metrics,
        },
        test_metrics_path,
    )

    print("Artifacts saved:")
    print(f"  - {best_model_path}")
    print(f"  - {vocab_path}")
    print(f"  - {config_path}")
    print(f"  - {history_path}")
    print(f"  - {test_metrics_path}")
    print(f"  - {cm_path}") 


if __name__ == "__main__":
    run_training()