from __future__ import annotations

import re


TOTAL_PATTERNS = ("итого", "итог", "всего", "баланс")

def _safe_text(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _num_digits(text: str) -> int:
    return len(re.findall(r"\d", text))


def _letters_only(text: str) -> list[str]:
    return [ch for ch in text if ch.isalpha()]


def is_upper_like(text: str) -> int:
    letters = _letters_only(text)
    if not letters:
        return 0
    upper = sum(ch.isupper() for ch in letters)
    return int(upper / len(letters) >= 0.8)


def has_digits(text: str) -> int:
    return int(bool(re.search(r"\d", text)))


def is_total_like(text: str) -> int:
    low = _safe_text(text).lower()
    return int(any(pattern in low for pattern in TOTAL_PATTERNS))


def get_bbox_features(norm_bbox: list[float]) -> dict[str, float]:
    """
    x_left, x_right, width, y_center, row_height
    """
    x0, y0, x1, y1 = norm_bbox
    width = max(0.0, x1 - x0)
    row_height = max(0.0, y1 - y0)
    y_center = (y0 + y1) / 2.0

    return {
        "x_left": float(x0),
        "x_right": float(x1),
        "width": float(width),
        "y_center": float(y_center),
        "row_height": float(row_height),
    }


def basic_text_features(text: str) -> dict[str, float]:
    """
    text_len, num_words, num_digits_text, has_parentheses, has_slash,
    is_upper_like, is_total_like, has_digits
    """
    text = _safe_text(text)

    return {
        "text_len": float(len(text)),
        "num_words": float(len(text.split())) if text else 0.0,
        "num_digits_text": float(_num_digits(text)),
        "has_parentheses": float(int(("(" in text) or (")" in text))),
        "has_slash": float(int("/" in text or "\\" in text)),
        "is_upper_like": float(is_upper_like(text)),
        "is_total_like": float(is_total_like(text)),
        "has_digits": float(has_digits(text)),
    }


def looks_like_row_code(text: str, row_codes: set[str]) -> int:
    text = _safe_text(text)
    if not text:
        return 0

    compact = re.sub(r"\s+", "", text)
    compact_low = compact.lower()

    normalized_codes = {str(code).strip().lower() for code in row_codes if str(code).strip()}
    if compact_low in normalized_codes:
        return 1

    #if re.fullmatch(r"[A-Za-zА-Яа-я]?\d{2,6}[A-Za-zА-Яа-я]?", compact):
    #    return 1

    #if re.fullmatch(r"\d{1,4}([./-]\d{1,4}){1,2}", compact):
    #    return 1

    return 0


def looks_like_date(text: str) -> int:
    """
    Эвристика под даты/шапки:
    - годы 2015..2030
    - 30 / 31 только рядом с месяцем текстом или числом
    - месяцы текстом
    - даты только с минимум двумя разделителями: dd.mm.yy / dd.mm.yyyy и аналоги
    - месяцы числом: mm.yyyy / mm.yy
    """
    text = _safe_text(text)
    if not text:
        return 0

    low = text.lower()

    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", low)]
    if any(2015 <= y <= 2030 for y in years):
        return 1
    
    long_month_stems = ["январ", "март", "июн","декабр"]
    short_months = ["янв",  "мар", "июн",  "дек"]
    
    has_long_text_month = any(stem in low for stem in long_month_stems)

    short_month_pat = r"(?:%s)" % "|".join(map(re.escape, short_months))
    has_short_text_month_with_day = bool(
        re.search(
            rf"(?<!\d)(0?[1-9]|[12]\d|3[01])\s*[-./]?\s*{short_month_pat}[а-яё]*",
            low,
            flags=re.IGNORECASE,
        )
    )

    has_text_month = has_long_text_month or has_short_text_month_with_day
    if has_text_month:
        return 1

    if re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", low):
        return 1

    if re.search(r"\b(0?[1-9]|1[0-2])[./-](20\d{2}|\d{2})\b", low):
        return 1

    has_day_30_31 = bool(re.search(r"\b(30|31)\b", low))
    has_numeric_month = bool(
        re.search(r"\b(0?[1-9]|1[0-2])\b", low)
    )

    if has_day_30_31 and (has_text_month or has_numeric_month):
        return 1

    return 0


def build_slot_numeric_features(
    text: str,
    bbox: list[float],
    row_codes: set[str],
) -> list[float]:
    
    bbox_feats = get_bbox_features(bbox)
    text_feats = basic_text_features(text)

    return [
        bbox_feats["x_left"],
        bbox_feats["x_right"],
        bbox_feats["width"],
        bbox_feats["y_center"],
        bbox_feats["row_height"],
        text_feats["text_len"],
        text_feats["num_words"],
        text_feats["num_digits_text"],
        text_feats["has_parentheses"],
        text_feats["has_slash"],
        text_feats["is_upper_like"],
        text_feats["is_total_like"],
        text_feats["has_digits"],
        float(looks_like_row_code(text, row_codes)),
        float(looks_like_date(text)),
    ]


def empty_slot_numeric_features() -> list[float]:
  
    return [0.0] * 15