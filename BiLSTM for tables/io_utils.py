from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from schemas import DocumentAnnotation, OcrRow, PhysicalTable


SPLIT_PDF_PATH_COLUMN = "Путь к файлу"
SPLIT_STATUS_COLUMN = "status"


def load_document_annotation(json_path: str) -> DocumentAnnotation:
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tables_raw = raw.get("tables", [])
    pages_raw = raw.get("pages", [])
    num_pages = int(raw.get("num_pages", len(pages_raw)))

    tables: list[PhysicalTable] = []
    for t in tables_raw:
        tables.append(
            PhysicalTable(
                table_id=int(t["table_id"]),
                page_index=int(t["page_index"])-1,
                class_id=int(t["class_id"]),
                rotated_90=bool(t.get("rotated_90", False)),
                bbox=[float(v) for v in t["bbox"]],
            )
        )

    pages: list[dict] = []
    for page_idx, page in enumerate(pages_raw):
        ocr_rows_raw = page.get("ocr_rows", [])
        ocr_rows: list[OcrRow] = []

        for row in ocr_rows_raw:
            text_conf = row.get("text_conf", None)
            ocr_rows.append(
                OcrRow(
                    row_idx=int(row["row_idx"]),
                    bbox=[float(v) for v in row["bbox"]],
                    text=str(row.get("text", "") or ""),
                    text_conf=None if text_conf is None else float(text_conf),
                    table_id=None if row.get("table_id") is None else int(row["table_id"]),
                    table_id_prelabel=None
                    if row.get("table_id_prelabel") is None
                    else int(row["table_id_prelabel"]),
                )
            )

        page_copy = dict(page)
        page_copy["page_index"] = int(page.get("page_index", page_idx))
        page_copy["ocr_rows"] = ocr_rows
        pages.append(page_copy)

    return DocumentAnnotation(
        document_path=str(path),
        num_pages=num_pages,
        tables=tables,
        pages=pages,
    )


def load_split_mapping(xls_path: str) -> dict[str, str]:
    """
    pdf_path -> split, где split in {'train', 'val', 'test'}
    Берем только колонки:
    - "Путь к файлу"
    - "status"
    """
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

    out: dict[str, str] = {}

    for _, row in df.iterrows():
        pdf_path = str(row[SPLIT_PDF_PATH_COLUMN]).strip()
        split = str(row[SPLIT_STATUS_COLUMN]).strip().lower()

        if not pdf_path:
            continue

        if split == "valid":
            split = "val"

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Некорректный split '{split}' для файла '{pdf_path}'. "
                f"Ожидается one of: train, val, test"
            )

        out[pdf_path] = split

    return out


def pdf_path_to_json_path(pdf_path: str) -> str:
    p = Path(pdf_path)
    return str(p.with_suffix(".annotation.json"))


def load_row_codes_txt(txt_path: str) -> set[str]:
    """
    Один код в строке.
    """
    codes: set[str] = set()

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.strip()
            if code:
                codes.add(code)

    return codes
