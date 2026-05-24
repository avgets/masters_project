from __future__ import annotations

from schemas import BinSlot, BinStep, DocumentAnnotation, OcrRow, PhysicalTable, TableSample

from features import build_slot_numeric_features, empty_slot_numeric_features
from io_utils import (
    load_document_annotation,
    load_row_codes_txt,
    load_split_mapping,
    pdf_path_to_json_path,
)

from statistics import median


def resolve_row_table_id(row: OcrRow) -> int | None:
    """
    Сначала row.table_id, если не None; иначе row.table_id_prelabel.
    """
    if row.table_id is not None:
        return row.table_id
    return row.table_id_prelabel


def compute_page_median_row_height(doc: DocumentAnnotation, page_index: int) -> float:
    """
    Медианная высота OCR-строк на странице в нормированных координатах [0, 1].
    """
    if page_index < 0 or page_index >= len(doc.pages):
        return 0.0

    page = doc.pages[page_index]
    page_rows = page.get("ocr_rows", [])

    heights: list[float] = []
    for row in page_rows:
        if row.bbox is None or len(row.bbox) != 4:
            continue
        _, y0, _, y1 = row.bbox
        h = max(0.0, y1 - y0)
        if h > 0:
            heights.append(h)

    if not heights:
        return 0.0

    return float(median(heights))

'''
def extract_table_rows(doc: DocumentAnnotation, table: PhysicalTable) -> list[OcrRow]:
    """
    Берет OCR-строки нужной страницы, относящиеся к table.table_id.
    Источник истины: resolve_row_table_id(row).
    """
    if table.page_index < 0 or table.page_index >= len(doc.pages):
        return []

    page = doc.pages[table.page_index]
    page_rows = page.get("ocr_rows", [])

    out: list[OcrRow] = []
    for row in page_rows:
        if resolve_row_table_id(row) == table.table_id:
            out.append(row)

    return out
'''

def extract_table_rows(doc: DocumentAnnotation, table: PhysicalTable) -> list[OcrRow]:
    """
    Берет OCR-строки нужной страницы, относящиеся к table.table_id,
    и дополнительно добирает строки из верхней расширенной зоны таблицы.

    Логика:
    1. Всегда включаем строки, у которых resolve_row_table_id(row) == table.table_id.
    2. Считаем медианную высоту OCR-строк на странице.
    3. Поднимаем верхнюю границу table.bbox вверх на 3 * median_height.
    4. Дополнительно включаем строки той же страницы, если их bbox
       пересекается с расширенной рамкой по Y.
    5. Убираем дубликаты.
    """
    if table.page_index < 0 or table.page_index >= len(doc.pages):
        return []

    page = doc.pages[table.page_index]
    page_rows = page.get("ocr_rows", [])

    if not page_rows:
        return []

    page_median_row_height = compute_page_median_row_height(doc, table.page_index)

    if table.bbox is None or len(table.bbox) != 4:
        return []

    x0, y0, x1, y1 = table.bbox
    expanded_y0 = max(0.0, y0 - 3.0 * page_median_row_height)

    out: list[OcrRow] = []
    seen_keys: set[tuple] = set()

    def row_key(row: OcrRow) -> tuple:
        return (
            getattr(row, "text", ""),
            tuple(row.bbox) if row.bbox is not None else (),
            resolve_row_table_id(row),
        )

    def add_row(row: OcrRow) -> None:
        key = row_key(row)
        if key not in seen_keys:
            seen_keys.add(key)
            out.append(row)

    for row in page_rows:
        if row.bbox is None or len(row.bbox) != 4:
            continue

        row_table_id = resolve_row_table_id(row)
        _, ry0, _, ry1 = row.bbox

        # 1) Явные строки таблицы
        if row_table_id == table.table_id:
            add_row(row)
            continue

        # 2) Строки, попавшие в расширенную верхнюю область таблицы
        y_overlaps = (ry1 > expanded_y0) and (ry0 < y1)
        if y_overlaps:
            add_row(row)

    return out



def assign_row_to_page_bin(row_bbox: list[float], num_bins: int) -> int:
    """
    Бин определяется по y_center строки в координатах всей страницы.
    Предполагается, что bbox уже в page-normalized координатах [0, 1].
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be > 0, got {num_bins}")

    _, ry0, _, ry1 = row_bbox
    row_y_center = (ry0 + ry1) / 2.0
    row_y_center = max(0.0, min(0.999999, row_y_center))

    bin_index = int(row_y_center * num_bins)
    return min(max(bin_index, 0), num_bins - 1)


def build_bin_groups(
    rows: list[OcrRow],
    table_bbox: list[float],
    num_bins: int,
) -> dict[int, list[OcrRow]]:
    """
    Возвращает только непустые бины: {bin_index: [rows...]}
    """
    groups: dict[int, list[OcrRow]] = {}

    for row in rows:
        bin_index = assign_row_to_page_bin(row.bbox, num_bins)
        groups.setdefault(bin_index, []).append(row)

    return groups


def sort_bin_rows_left_to_right(rows: list[OcrRow]) -> list[OcrRow]:
    """
    Сортировка по x_center в координатах страницы.
    """
    return sorted(rows, key=lambda row: (row.bbox[0] + row.bbox[2]) / 2.0)


def make_empty_slot() -> BinSlot:
    return BinSlot(
        text="",
        bbox=[0.0, 0.0, 0.0, 0.0],
        numeric_features=empty_slot_numeric_features(),
    )


def build_bin_step(
    bin_index: int,
    rows_in_bin: list[OcrRow],
    table_bbox: list[float],
    seq_pos: int,
    seq_len: int,
    prev_nonempty_bin_index: int | None,
    row_codes: set[str],
    num_slots: int,
) -> BinStep:
    """
    1. Сортирует rows_in_bin по x_center
    2. Берет только первые num_slots строк
    3. Остальные игнорирует
    4. Пустые слоты дополняет
    5. Считает empty_bins_from_prev
    """
    sorted_rows = sort_bin_rows_left_to_right(rows_in_bin)
    used_rows = sorted_rows[:num_slots]

    slots: list[BinSlot] = []
    for row in used_rows:
        numeric_features = build_slot_numeric_features(
            text=row.text,
            bbox=row.bbox, #norm_bbox,
            row_codes=row_codes,
        )
        slots.append(
            BinSlot(
                text=row.text,
                bbox=row.bbox, #norm_bbox,
                numeric_features=numeric_features,
            )
        )

    while len(slots) < num_slots:
        slots.append(make_empty_slot())

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


def build_table_sample(
    doc: DocumentAnnotation,
    table: PhysicalTable,
    row_codes: set[str],
    num_bins: int,
    num_slots: int,
) -> TableSample | None:
    """
    rotated_90=True -> None
    нет непустых бинов -> None
    """
    if table.rotated_90:
        return None

    rows = extract_table_rows(doc, table)
    if not rows:
        return None

    bin_groups = build_bin_groups(rows, table.bbox, num_bins)
    if not bin_groups:
        return None

    nonempty_bins = sorted(bin_groups.keys())
    seq_len = len(nonempty_bins)

    steps: list[BinStep] = []
    prev_nonempty_bin_index: int | None = None

    for seq_pos, bin_index in enumerate(nonempty_bins):
        step = build_bin_step(
            bin_index=bin_index,
            rows_in_bin=bin_groups[bin_index],
            table_bbox=table.bbox,
            seq_pos=seq_pos,
            seq_len=seq_len,
            prev_nonempty_bin_index=prev_nonempty_bin_index,
            row_codes=row_codes,
            num_slots=num_slots,
        )
        steps.append(step)
        prev_nonempty_bin_index = bin_index

    if not steps:
        return None

    sample_id = f"{doc.document_path}::page{table.page_index}::table{table.table_id}"

    return TableSample(
        sample_id=sample_id,
        document_path=doc.document_path,
        page_index=table.page_index,
        table_id=table.table_id,
        class_id=table.class_id,
        table_bbox=table.bbox,
        steps=steps,
    )


def build_samples_from_document(
    json_path: str,
    row_codes: set[str],
    num_bins: int,
    num_slots: int,
) -> list[TableSample]:
    doc = load_document_annotation(json_path)

    samples: list[TableSample] = []
    for table in doc.tables:
        sample = build_table_sample(
            doc=doc,
            table=table,
            row_codes=row_codes,
            num_bins=num_bins,
            num_slots=num_slots,
        )
        if sample is not None:
            samples.append(sample)

    return samples


def build_dataset_splits(
    split_xls_path: str,
    row_codes_path: str,
    num_bins: int,
    num_slots: int,
) -> tuple[list[TableSample], list[TableSample], list[TableSample]]:
    """
    split по документам из XLS.
    json лежит рядом с pdf, имя то же, расширение .json
    """
    split_mapping = load_split_mapping(split_xls_path)
    row_codes = load_row_codes_txt(row_codes_path)

    train_samples: list[TableSample] = []
    val_samples: list[TableSample] = []
    test_samples: list[TableSample] = []

    for pdf_path, split in split_mapping.items():
        json_path = pdf_path_to_json_path(pdf_path)
        samples = build_samples_from_document(
            json_path=json_path,
            row_codes=row_codes,
            num_bins=num_bins,
            num_slots=num_slots,
        )

        if split == "train":
            train_samples.extend(samples)
        elif split == "val":
            val_samples.extend(samples)
        elif split == "test":
            test_samples.extend(samples)
        else:
            raise ValueError(f"Unknown split '{split}' for file '{pdf_path}'")

    return train_samples, val_samples, test_samples