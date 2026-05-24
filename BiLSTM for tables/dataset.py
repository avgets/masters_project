from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import Dataset

from config import (
    CHAR_PAD_TOKEN,
    CHAR_UNK_TOKEN,
    GLOBAL_BIN_FEATURES,
    SLOT_FEATURES,
)
from schemas import BinStep, TableSample


def build_charset(samples: list[TableSample]) -> tuple[list[str], dict[str, int]]:
    """
    Строит символьный словарь только по train samples.

    Возвращает:
        itos: index-to-string
        stoi: string-to-index
    """
    charset: set[str] = set()

    for sample in samples:
        for step in sample.steps:
            for slot in step.slots:
                if slot.text:
                    charset.update(slot.text)

    special_tokens = [CHAR_PAD_TOKEN, CHAR_UNK_TOKEN]
    itos = special_tokens + sorted(charset)
    stoi = {ch: idx for idx, ch in enumerate(itos)}
    return itos, stoi


def encode_text(text: str, stoi: dict[str, int], max_len: int) -> tuple[list[int], int]:
    """
    Символьное кодирование текста.

    Возвращает:
        ids: список длины max_len
        length: реальная длина до padding/truncation, но не больше max_len
    """
    text = text or ""

    pad_idx = stoi[CHAR_PAD_TOKEN]
    unk_idx = stoi[CHAR_UNK_TOKEN]

    ids = [stoi.get(ch, unk_idx) for ch in text[:max_len]]
    length = len(ids)

    if length < max_len:
        ids.extend([pad_idx] * (max_len - length))

    return ids, length


def get_numeric_feature_names(num_slots: int) -> list[str]:
    """
    Возвращает порядок numeric features на timestep:
    GLOBAL_BIN_FEATURES + slot0_* + slot1_* + ...
    """
    feature_names = list(GLOBAL_BIN_FEATURES)
    for slot_idx in range(num_slots):
        for feat_name in SLOT_FEATURES:
            feature_names.append(f"slot{slot_idx}_{feat_name}")
    return feature_names


def flatten_step_numeric(step: BinStep) -> list[float]:
    """
    Преобразует BinStep в плоский numeric vector.

    Порядок:
        seq_pos, seq_len, empty_bins_from_prev,
        slot0_*,
        slot1_*,
        ...
    """
    numeric: list[float] = [
        float(step.seq_pos),
        float(step.seq_len),
        float(step.empty_bins_from_prev),
    ]

    for slot in step.slots:
        numeric.extend(float(x) for x in slot.numeric_features)

    return numeric


class TableSequenceDataset(Dataset):
    """
    Dataset для table classification.

    На выходе __getitem__:
        {
            'sample_id': str,
            'label': int,
            'seqlen': int,
            'numeric': FloatTensor[T, D],
            'text_ids': LongTensor[T, S, L],
            'text_len': LongTensor[T, S],
        }
    """

    def __init__(
        self,
        samples: list[TableSample],
        stoi: dict[str, int],
        max_text_len: int,
    ) -> None:
        self.samples = samples
        self.stoi = stoi
        self.max_text_len = max_text_len

        if len(self.samples) == 0:
            raise ValueError("samples must not be empty")

        first_sample = self.samples[0]
        if len(first_sample.steps) == 0:
            raise ValueError("sample.steps must not be empty")

        self.num_slots = len(first_sample.steps[0].slots)
        self.numeric_feature_names = get_numeric_feature_names(self.num_slots)
        self.num_numeric_features = len(self.numeric_feature_names)

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_step_texts(self, step: BinStep) -> tuple[list[list[int]], list[int]]:
        """
        Кодирует все слоты одного timestep.

        Returns:
            slot_text_ids: [S, L]
            slot_text_len: [S]
        """
        slot_text_ids: list[list[int]] = []
        slot_text_len: list[int] = []

        for slot in step.slots:
            ids, length = encode_text(
                text=slot.text,
                stoi=self.stoi,
                max_len=self.max_text_len,
            )
            slot_text_ids.append(ids)
            slot_text_len.append(length)

        return slot_text_ids, slot_text_len

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if len(sample.steps) == 0:
            raise ValueError(f"sample {sample.sample_id} has no steps")

        numeric_rows: list[list[float]] = []
        text_ids_rows: list[list[list[int]]] = []
        text_len_rows: list[list[int]] = []

        for step in sample.steps:
            if len(step.slots) != self.num_slots:
                raise ValueError(
                    f"sample {sample.sample_id}: inconsistent num_slots, "
                    f"expected {self.num_slots}, got {len(step.slots)}"
                )

            step_numeric = flatten_step_numeric(step)
            if len(step_numeric) != self.num_numeric_features:
                raise ValueError(
                    f"sample {sample.sample_id}: numeric dim mismatch, "
                    f"expected {self.num_numeric_features}, got {len(step_numeric)}"
                )

            step_text_ids, step_text_len = self._encode_step_texts(step)

            numeric_rows.append(step_numeric)
            text_ids_rows.append(step_text_ids)
            text_len_rows.append(step_text_len)

        return {
            "sample_id": sample.sample_id,
            "label": int(sample.class_id),
            "seqlen": int(len(sample.steps)),
            "numeric": torch.tensor(numeric_rows, dtype=torch.float32),       # [T, D]
            "text_ids": torch.tensor(text_ids_rows, dtype=torch.long),        # [T, S, L]
            "text_len": torch.tensor(text_len_rows, dtype=torch.long),        # [T, S]
        }


def collate_table_batch(batch: list[dict]) -> dict:
    """
    Collate function для DataLoader.

    Возвращает:
        {
            'sample_ids': list[str],
            'labels': LongTensor[B],
            'seqlens': LongTensor[B],
            'numeric': FloatTensor[B, T_max, D],
            'text_ids': LongTensor[B, T_max, S, L],
            'text_len': LongTensor[B, T_max, S],
        }
    """
    if len(batch) == 0:
        raise ValueError("batch must not be empty")

    batch = sorted(batch, key=lambda x: x["seqlen"], reverse=True)

    batch_size = len(batch)
    t_max = max(item["seqlen"] for item in batch)

    sample0 = batch[0]
    _, num_slots, max_text_len = sample0["text_ids"].shape
    num_numeric_features = sample0["numeric"].shape[-1]

    sample_ids: list[str] = []
    labels = torch.zeros(batch_size, dtype=torch.long)
    seqlens = torch.zeros(batch_size, dtype=torch.long)

    numeric = torch.zeros(batch_size, t_max, num_numeric_features, dtype=torch.float32)
    text_ids = torch.zeros(batch_size, t_max, num_slots, max_text_len, dtype=torch.long)
    text_len = torch.zeros(batch_size, t_max, num_slots, dtype=torch.long)

    for i, item in enumerate(batch):
        t = int(item["seqlen"])

        sample_ids.append(item["sample_id"])
        labels[i] = int(item["label"])
        seqlens[i] = t

        numeric[i, :t] = item["numeric"]
        text_ids[i, :t] = item["text_ids"]
        text_len[i, :t] = item["text_len"]

    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "seqlens": seqlens,
        "numeric": numeric,
        "text_ids": text_ids,
        "text_len": text_len,
    }