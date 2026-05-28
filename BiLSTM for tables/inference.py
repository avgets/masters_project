from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    CLASS_IDS,
    MAX_TEXT_LEN,
    NUM_BINS,
    NUM_SLOTS,
    ROW_CODES_PATH,
    SPLIT_XLS_PATH,
)

from dataset import TableSequenceDataset, collate_table_batch

from io_utils import (
    SPLIT_PDF_PATH_COLUMN,
    SPLIT_STATUS_COLUMN,
    load_document_annotation,
    load_row_codes_txt,
    pdf_path_to_json_path,
)

from metrics import classification_metrics, confusion_matrix_df
from model import TableBiLSTMClassifier
from preprocessing import build_table_sample


from pathlib import Path

# Абсолютный путь к директории репозитория с классификатором
PROJECT_ROOT = Path(__file__).resolve().parent

# Путь к весам относительно inference.py:
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "table_cls" 

MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
OUTPUT_DIR = Path("artifacts/table_cls_inference")
MISCLS_DIR = OUTPUT_DIR / "misclassified"

IMAGES_ROOT = Path(r"C:\Img_Dataset\eval\images")
#BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEP_HIDDEN_DIM = 128
TABLE_HIDDEN_DIM = 128
CHAR_EMB_DIM = 64
CHAR_HIDDEN_DIM = 64
DROPOUT = 0.2


def load_vocab(vocab_path: str | Path) -> tuple[list[str], dict[str, int]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["itos"], {str(k): int(v) for k, v in raw["stoi"].items()}


def load_test_pdf_paths(xls_path: str | Path) -> list[str]:
    df = pd.read_excel(xls_path)

    if SPLIT_PDF_PATH_COLUMN not in df.columns:
        raise ValueError(
            f"В split-файле нет колонки '{SPLIT_PDF_PATH_COLUMN}'. "
            f"Доступные колонки: {list(df.columns)}"
        )
    if SPLIT_STATUS_COLUMN not in df.columns:
        raise ValueError(
            f"В split-файле нет колонки '{SPLIT_STATUS_COLUMN}'. "
            f"Доступные колонки: {list(df.columns)}"
        )

    out: list[str] = []
    for _, row in df.iterrows():
        pdf_path = str(row[SPLIT_PDF_PATH_COLUMN]).strip()
        split = str(row[SPLIT_STATUS_COLUMN]).strip().lower()
        if not pdf_path:
            continue
        if split == "test":
            out.append(pdf_path)
    return out


def extract_codes_from_pdf_path(pdf_path: str) -> tuple[str, str]:
    folder_name = Path(pdf_path).parent.name
    codes = re.findall(r"\[(\d+)\]", folder_name)
    if len(codes) < 2:
        raise ValueError(
            f"Не удалось извлечь два числовых кода из имени папки: '{folder_name}'"
        )
    return codes[-2], codes[-1]


def pdf_page_to_image_path(pdf_path: str, page_index_zero_based: int) -> Path:
    code1, code2 = extract_codes_from_pdf_path(pdf_path)
    page_num = page_index_zero_based + 1
    filename = f"{code1}_{code2}_p{page_num:03d}.jpg"
    return IMAGES_ROOT / filename


def build_model(stoi: dict[str, int], num_numeric_features: int) -> TableBiLSTMClassifier:
    pad_idx = stoi.get("<PAD>", 0)
    print(f"Модель классификатора загружаем на {DEVICE}")
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
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def predict_document_samples(
    model: TableBiLSTMClassifier,
    samples,
    stoi: dict[str, int],
) -> list[dict]:
    ds = TableSequenceDataset(samples=samples, stoi=stoi, max_text_len=MAX_TEXT_LEN)

    doc_batch_size = max(1, len(samples))
    loader = DataLoader(
        ds,
        batch_size=doc_batch_size,
        shuffle=False,
        collate_fn=collate_table_batch,
        num_workers=0,
    )

    results: list[dict] = []
    sample_iter_idx = 0

    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        preds = preds.cpu().tolist()
        confs = confs.cpu().tolist()
        labels = batch["labels"].cpu().tolist()
        sample_ids = batch["sample_ids"]

        for sample_id, pred, conf, label in zip(sample_ids, preds, confs, labels):
            sample = samples[sample_iter_idx]
            results.append(
                {
                    "sample_id": sample_id,
                    "pred_class": int(pred),
                    "pred_confidence": float(conf),
                    "true_class": int(label),
                    "page_index": int(sample.page_index),
                    "table_id": int(sample.table_id),
                    "table_bbox": list(sample.table_bbox),
                    "document_path": sample.document_path,
                    "num_steps": int(len(sample.steps)),
                }
            )
            sample_iter_idx += 1

    return results


@torch.no_grad()
def predict_document_samples(
    model: TableBiLSTMClassifier,
    samples,
    stoi: dict[str, int],
) -> list[dict]:
    ds = TableSequenceDataset(samples=samples, stoi=stoi, max_text_len=MAX_TEXT_LEN)

    doc_batch_size = max(1, len(samples))

    loader = DataLoader(
        ds,
        batch_size=doc_batch_size,
        shuffle=False,
        collate_fn=collate_table_batch,
        num_workers=0,
    )

    sample_meta = {
        s.sample_id: {
            "page_index": int(s.page_index),
            "table_id": int(s.table_id),
            "table_bbox": list(s.table_bbox),
            "document_path": s.document_path,
            "num_steps": int(len(s.steps)),
            "class_id": int(s.class_id),
        }
        for s in samples
    }

    results: list[dict] = []

    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        preds = preds.cpu().tolist()
        confs = confs.cpu().tolist()
        labels = batch["labels"].cpu().tolist()
        sample_ids = batch["sample_ids"]

        for sample_id, pred, conf, label in zip(sample_ids, preds, confs, labels):
            meta = sample_meta[sample_id]
            assert int(label) == int(meta["class_id"]), (
                label,
                meta["class_id"],
                sample_id,
            )

            results.append(
                {
                    "sample_id": sample_id,
                    "pred_class": int(pred),
                    "pred_confidence": float(conf),
                    "true_class": int(label),
                    "page_index": meta["page_index"],
                    "table_id": meta["table_id"],
                    "table_bbox": meta["table_bbox"],
                    "document_path": meta["document_path"],
                    "num_steps": meta["num_steps"],
                }
            )

    return results


def norm_bbox_to_pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


def draw_multiline_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill, bg_fill, font) -> None:
    x, y = xy
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
    pad = 4
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(rect, fill=bg_fill)
    draw.multiline_text((x, y), text, fill=fill, font=font, spacing=2)


def save_misclassified_image(pdf_path: str, item: dict, save_dir: Path) -> Path:
    image_path = pdf_page_to_image_path(pdf_path, item["page_index"])
    if not image_path.exists():
        raise FileNotFoundError(f"Не найден рендер страницы: {image_path}")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x0, y0, x1, y1 = norm_bbox_to_pixels(item["table_bbox"], image.width, image.height)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)

    label_text = (
        f"gt={item['true_class']}\n"
        f"pred={item['pred_class']}\n"
        f"conf={item['pred_confidence']:.3f}"
    )
    label_x = max(0, x0)
    label_y = max(0, y0 - 42)
    draw_multiline_label(
        draw,
        (label_x, label_y),
        label_text,
        fill=(255, 255, 255),
        bg_fill=(255, 0, 0),
        font=font,
    )

    out_name = (
        f"{Path(pdf_path).stem}__page{item['page_index'] + 1:03d}__"
        f"table{item['table_id']}__gt{item['true_class']}__pred{item['pred_class']}.jpg"
    )
    out_path = save_dir / out_name
    image.save(out_path, quality=95)
    return out_path


def run_inference() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MISCLS_DIR.mkdir(parents=True, exist_ok=True)

    _, stoi = load_vocab(VOCAB_PATH)
    row_codes = load_row_codes_txt(ROW_CODES_PATH)
    test_pdf_paths = load_test_pdf_paths(SPLIT_XLS_PATH)

    print(f"Device: {DEVICE}")
    print(f"Test documents: {len(test_pdf_paths)}")

    model: TableBiLSTMClassifier | None = None
    all_results: list[dict] = []
    total_tables = 0

    for pdf_path in tqdm(test_pdf_paths, desc="documents"):
        json_path = pdf_path_to_json_path(pdf_path)
        doc = load_document_annotation(json_path)

        samples = []
        for table in doc.tables:
            sample = build_table_sample(
                doc=doc,
                table=table,
                row_codes=row_codes,
                num_bins=NUM_BINS,
                num_slots=NUM_SLOTS,
            )
            if sample is not None:
                samples.append(sample)

        if not samples:
            continue

        if model is None:
            probe_ds = TableSequenceDataset(samples=samples, stoi=stoi, max_text_len=MAX_TEXT_LEN)
            model = build_model(stoi=stoi, num_numeric_features=probe_ds.num_numeric_features)

        doc_results = predict_document_samples(model=model, samples=samples, stoi=stoi)
        total_tables += len(doc_results)

        for item in doc_results:
            item["pdf_path"] = pdf_path
            item["json_path"] = json_path
            item["image_path"] = str(pdf_page_to_image_path(pdf_path, item["page_index"]))
            item["is_error"] = bool(item["pred_class"] != item["true_class"])

            if item["is_error"]:
                saved_path = save_misclassified_image(pdf_path, item, MISCLS_DIR)
                item["misclassified_image"] = str(saved_path)
            else:
                item["misclassified_image"] = None

            all_results.append(item)

    y_true = [int(x["true_class"]) for x in all_results]
    y_pred = [int(x["pred_class"]) for x in all_results]

    cls_metrics = classification_metrics(y_true, y_pred)
    cm_df = confusion_matrix_df(y_true=y_true, y_pred=y_pred, class_ids=list(CLASS_IDS))

    results_df = pd.DataFrame(all_results)
    results_csv_path = OUTPUT_DIR / "inference_results.csv"
    results_json_path = OUTPUT_DIR / "inference_results.json"
    summary_json_path = OUTPUT_DIR / "inference_summary.json"
    cm_csv_path = OUTPUT_DIR / "confusion_matrix.csv"

    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    summary = {
        "device": str(DEVICE),
        "num_test_documents": len(test_pdf_paths),
        "num_tables": total_tables,
        "num_errors": int(sum(int(x["is_error"]) for x in all_results)),
        "model_path": str(MODEL_PATH),
        "vocab_path": str(VOCAB_PATH),
        "images_root": str(IMAGES_ROOT),
        **cls_metrics,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== INFERENCE SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== CONFUSION MATRIX ===")
    print(cm_df)
    print("\nArtifacts saved:")
    print(f"  - {results_json_path}")
    print(f"  - {results_csv_path}")
    print(f"  - {summary_json_path}")
    print(f"  - {cm_csv_path}")
    print(f"  - {MISCLS_DIR}")


if __name__ == "__main__":
    run_inference()
