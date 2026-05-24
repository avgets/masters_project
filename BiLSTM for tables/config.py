from __future__ import annotations

CLASS_IDS = [0, 1, 2, 3, 4]
OTHER_TABLE_CLASS_ID = 4

NUM_BINS: int = 64
NUM_SLOTS: int = 5
MAX_TEXT_LEN: int = 256

CHAR_PAD_TOKEN: str = "<PAD>"
CHAR_UNK_TOKEN: str = "UNK"

GLOBAL_BIN_FEATURES = [
    "seq_pos",
    "seq_len",
    "empty_bins_from_prev",
]

SLOT_FEATURES = [
    "x_left",
    "x_right",
    "width",
    "y_center",
    "row_height",
    "text_len",
    "num_words",
    "num_digits_text",
    "has_parentheses",
    "has_slash",
    "is_upper_like",
    "is_total_like",
    "has_digits",
    "looks_like_row_code",
    "looks_like_date",
]

ROW_CODES_PATH: str = "resources/row_codes.txt"
SPLIT_XLS_PATH: str = r"C:\Users\GaV\Desktop\FinRepDatasetFileList.xlsx"

