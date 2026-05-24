from __future__ import annotations

import torch

from config import MAX_TEXT_LEN
from dataset import collate_table_batch, encode_text, flatten_step_numeric
from features import build_slot_numeric_features, normalize_bbox_to_table
from schemas import (
    BinSlot,
    BinStep,
    InferenceTableInput,
    TableOcrRow,
    TableSample,
)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _x_center(bbox: list[float]) -> float:
    return (_safe_float(bbox[0]) + _safe_float(bbox[2])) / 2.0


def _y_center(bbox: list[float]) -> float:
    return (_safe_float(bbox[1]) + _safe_float(bbox[3])) / 2.0


def _empty_slot() -> BinSlot:
    return BinSlot(
        text="",
        bbox=[0.0, 0.0, 0.0, 0.0],
        numeric_features=[],
    )


def _assign_row_to_bin(norm_bbox: list[float], num_bins: int) -> int:
    """
    Бин определяется по y_center в координатах таблицы [0, 1].
    """
    yc = _y_center(norm_bbox)
    yc = max(0.0, min(1.0, yc))

    if num_bins <= 1:
        return 0

    bin_idx = int(yc * num_bins)
    if bin_idx >= num_bins:
        bin_idx = num_bins - 1
    return bin_idx


def _build_bin_groups(rows: list[TableOcrRow], num_bins: int) -> dict[int, list[TableOcrRow]]:
    groups: dict[int, list[TableOcrRow]] = {}
    for row in rows:
        bin_idx = _assign_row_to_bin(row.table_bbox_norm, num_bins)
        groups.setdefault(bin_idx, []).append(row)
    return groups


def _sort_bin_rows_left_to_right(rows: list[TableOcrRow]) -> list[TableOcrRow]:
    return sorted(rows, key=lambda r: _x_center(r.table_bbox_norm))


def _make_step(
    bin_index: int,
    rows_in_bin: list[TableOcrRow],
    seq_pos: int,
    seq_len: int,
    prev_nonempty_bin_index: int | None,
    row_codes: set[str],
    num_slots: int,
) -> BinStep:
    rows_sorted = _sort_bin_rows_left_to_right(rows_in_bin)
    rows_sorted = rows_sorted[:num_slots]

    slots: list[BinSlot] = []
    for row in rows_sorted:
        numeric = build_slot_numeric_features(
            text=row.text,
            norm_bbox=row.table_bbox_norm,
            row_codes=row_codes,
        )
        slots.append(
            BinSlot(
                text=row.text,
                bbox=row.table_bbox_norm,
                numeric_features=numeric,
            )
        )

    while len(slots) < num_slots:
        slots.append(_empty_slot())

    if prev_nonempty_bin_index is None:
        empty_bins_from_prev = 0
    else:
        empty_bins_from_prev = max(0, bin_index - prev_nonempty_bin_index - 1)

    return BinStep(
        bin_index=bin_index,
        seq_pos=seq_pos,
        seq_len=seq_len,
        empty_bins_from_prev=empty_bins_from_prev,
        slots=slots,
    )


def build_inference_sample(
    inp: InferenceTableInput,
    row_codes: set[str],
    num_bins: int,
    num_slots: int,
) -> TableSample:
    """
    На инференсе table_id / table_id_prelabel не используются.
    Берем все OCR-строки, бинним по bbox таблицы, строим steps.
    """
    table_bbox = inp.table_bbox

    table_rows: list[TableOcrRow] = []
    for row in inp.ocr_rows:
        row_idx = int(row["row_idx"])
        row_bbox = [float(v) for v in row["bbox"]]
        text = str(row.get("text", "") or "")
        text_conf = row.get("text_conf", None)
        text_conf = None if text_conf is None else float(text_conf)

        norm_bbox = normalize_bbox_to_table(row_bbox, table_bbox)

        table_rows.append(
            TableOcrRow(
                row_idx=row_idx,
                text=text,
                text_conf=text_conf,
                page_bbox=row_bbox,
                table_bbox_norm=norm_bbox,
            )
        )

    bin_groups = _build_bin_groups(table_rows, num_bins)
    nonempty_bin_indices = sorted(bin_groups.keys())

    if len(nonempty_bin_indices) == 0:
        raise ValueError("No OCR rows fell into table bins; cannot build inference sample")

    seq_len = len(nonempty_bin_indices)
    steps: list[BinStep] = []

    prev_nonempty_bin_index: int | None = None
    for seq_pos, bin_index in enumerate(nonempty_bin_indices):
        step = _make_step(
            bin_index=bin_index,
            rows_in_bin=bin_groups[bin_index],
            seq_pos=seq_pos,
            seq_len=seq_len,
            prev_nonempty_bin_index=prev_nonempty_bin_index,
            row_codes=row_codes,
            num_slots=num_slots,
        )
        steps.append(step)
        prev_nonempty_bin_index = bin_index

    sample_id = getattr(inp, "sample_id", None) or "inference_sample"
    document_path = getattr(inp, "document_path", None) or "inference_document"
    page_index = int(getattr(inp, "page_index", 0))
    table_id = int(getattr(inp, "table_id", -1))

    return TableSample(
        sample_id=sample_id,
        document_path=document_path,
        page_index=page_index,
        table_id=table_id,
        class_id=-1,
        table_bbox=[float(v) for v in table_bbox],
        steps=steps,
    )


@torch.no_grad()
def predict_one_sample(model, sample, stoi, device) -> dict:
    """
    return {
        'pred_class_id': int,
        'probs': list[float],
    }
    """
    text_ids_steps = []
    text_len_steps = []
    numeric_steps = []

    for step in sample.steps:
        slot_text_ids = []
        slot_text_len = []

        for slot in step.slots:
            ids, length = encode_text(slot.text, stoi=stoi, max_len=MAX_TEXT_LEN)
            slot_text_ids.append(ids)
            slot_text_len.append(length)

        text_ids_steps.append(slot_text_ids)
        text_len_steps.append(slot_text_len)
        numeric_steps.append(flatten_step_numeric(step))

    item = {
        "sample_id": sample.sample_id,
        "label": int(getattr(sample, "class_id", -1)),
        "seqlen": len(sample.steps),
        "numeric": torch.tensor(numeric_steps, dtype=torch.float32),
        "text_ids": torch.tensor(text_ids_steps, dtype=torch.long),
        "text_len": torch.tensor(text_len_steps, dtype=torch.long),
    }

    batch = collate_table_batch([item])

    batch_on_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_on_device[k] = v.to(device)
        else:
            batch_on_device[k] = v

    model.eval()
    logits = model(batch_on_device)                  # [1, C]
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    pred_class_id = int(torch.argmax(logits, dim=1).item())

    return {
        "pred_class_id": pred_class_id,
        "probs": probs,
    }