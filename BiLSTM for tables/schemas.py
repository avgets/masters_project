from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBox = list[float]

@dataclass(slots=True)
class OcrRow:
    """
    OCR-строка на странице документа.

    Поля table_id / table_id_prelabel используются только на этапе
    формирования train samples и не должны идти в модель как признаки.
    """
    row_idx: int
    bbox: BBox                      # [x0, y0, x1, y1] в координатах страницы
    text: str
    text_conf: float | None = None
    table_id: int | None = None
    table_id_prelabel: int | None = None


@dataclass(slots=True)
class PhysicalTable:
    """
    Физическая таблица из разметки документа.
    Одна запись = одна таблица на конкретной странице.
    """
    table_id: int
    page_index: int
    class_id: int
    rotated_90: bool
    bbox: BBox                      # [x0, y0, x1, y1] в координатах страницы


@dataclass(slots=True)
class DocumentPage:
    """
    Одна страница документа с OCR-строками.
    Остальные поля JSON можно хранить в meta при необходимости.
    """
    page_index: int
    ocr_rows: list[OcrRow] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentAnnotation:
    """
    Полный объект документа после чтения JSON.
    """
    document_path: str
    num_pages: int
    tables: list[PhysicalTable] = field(default_factory=list)
    pages: list[DocumentPage] = field(default_factory=list)


@dataclass(slots=True)
class TableOcrRow:
    """
    OCR-строка, уже отнесенная к конкретной таблице.
    bbox страницы сохраняется отдельно, а bbox в координатах таблицы
    нужен для биннинга и вычисления numeric features.
    """
    row_idx: int
    text: str
    text_conf: float | None
    page_bbox: BBox
    table_bbox_norm: BBox           # bbox строки в координатах таблицы


@dataclass(slots=True)
class BinSlot:
    """
    Один слот внутри бина.
    Обычно соответствует одной OCR-строке внутри bin-а после сортировки
    слева направо; если слот пустой, text='' и numeric_features — нули.
    """
    text: str
    bbox: BBox                      # bbox в координатах таблицы
    numeric_features: list[float]


@dataclass(slots=True)
class BinStep:
    """
    Один timestep последовательности для table classifier.
    Соответствует одному непустому vertical bin.
    """
    bin_index: int
    seq_pos: int
    seq_len: int
    empty_bins_from_prev: int
    slots: list[BinSlot]            # ожидаемая длина: NUM_SLOTS


@dataclass(slots=True)
class TableSample:
    """
    Финальный sample для классификации физической таблицы.
    Одна физическая таблица = один sample.
    """
    sample_id: str
    document_path: str
    page_index: int
    table_id: int
    class_id: int | None
    table_bbox: BBox
    steps: list[BinStep] = field(default_factory=list)


@dataclass
class InferenceTableInput:
    table_bbox: list[float]
    ocr_rows: list[dict[str, Any]]