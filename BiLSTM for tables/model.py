from __future__ import annotations

import torch
import torch.nn as nn

class CharBiLSTMEncoder(nn.Module):
    """
    Символьный BiLSTM-энкодер одной текстовой последовательности.

    Input:
        x:       [N, L]
        lengths: [N]

    Output:
        vec:     [N, 2 * hidden_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, L] long
            lengths: [N] long

        Returns:
            [N, 2 * hidden_dim]
        """
        lengths = lengths.clamp_min(1)

        emb = self.embedding(x)              # [N, L, E]
        emb = self.dropout(emb)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hn, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            out = torch.cat([hn[-2], hn[-1]], dim=-1)   # [N, 2H]
        else:
            out = hn[-1]                                # [N, H]

        return out


class BinStepEncoder(nn.Module):
    """
    Кодирует один timestep последовательности таблицы.

    На входе timestep содержит:
    - до num_slots текстовых слотов;
    - numeric features всего бина.

    Логика:
    1. Все слоты прогоняются через один и тот же CharBiLSTMEncoder.
    2. Эмбеддинги слотов конкатенируются.
    3. Добавляются numeric features.
    4. MLP строит step representation.

    Input:
        text_ids: [B, T, S, L]
        text_len: [B, T, S]
        numeric:  [B, T, D]

    Output:
        step_vec: [B, T, step_hidden_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        num_numeric_features: int,
        num_slots: int = 5,
        char_emb_dim: int = 64,
        char_hidden_dim: int = 64,
        step_hidden_dim: int = 128,
        pad_idx: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_slots = num_slots
        self.num_numeric_features = num_numeric_features

        self.text_encoder = CharBiLSTMEncoder(
            vocab_size=vocab_size,
            emb_dim=char_emb_dim,
            hidden_dim=char_hidden_dim,
            num_layers=1,
            pad_idx=pad_idx,
            dropout=dropout,
        )

        step_input_dim = self.text_encoder.output_dim * num_slots + num_numeric_features

        self.step_mlp = nn.Sequential(
            nn.Linear(step_input_dim, step_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(step_hidden_dim, step_hidden_dim),
            nn.ReLU(),
        )

        self.output_dim = step_hidden_dim

    def forward(
        self,
        text_ids: torch.Tensor,
        text_len: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_ids: [B, T, S, L]
            text_len: [B, T, S]
            numeric:  [B, T, D]

        Returns:
            [B, T, step_hidden_dim]
        """
        if text_ids.ndim != 4:
            raise ValueError(f"text_ids must have shape [B, T, S, L], got {tuple(text_ids.shape)}")
        if text_len.ndim != 3:
            raise ValueError(f"text_len must have shape [B, T, S], got {tuple(text_len.shape)}")
        if numeric.ndim != 3:
            raise ValueError(f"numeric must have shape [B, T, D], got {tuple(numeric.shape)}")

        bsz, tmax, num_slots, max_text_len = text_ids.shape
        if num_slots != self.num_slots:
            raise ValueError(f"expected num_slots={self.num_slots}, got {num_slots}")

        if text_len.shape != (bsz, tmax, num_slots):
            raise ValueError(
                f"text_len shape mismatch: expected {(bsz, tmax, num_slots)}, got {tuple(text_len.shape)}"
            )

        if numeric.shape[:2] != (bsz, tmax):
            raise ValueError(
                f"numeric first dims mismatch: expected {(bsz, tmax, 'D')}, got {tuple(numeric.shape)}"
            )

        num_dim = numeric.shape[-1]
        if num_dim != self.num_numeric_features:
            raise ValueError(
                f"expected num_numeric_features={self.num_numeric_features}, got {num_dim}"
            )

        text_ids_flat = text_ids.reshape(bsz * tmax * num_slots, max_text_len)   # [B*T*S, L]
        text_len_flat = text_len.reshape(bsz * tmax * num_slots)                 # [B*T*S]

        slot_vec_flat = self.text_encoder(text_ids_flat, text_len_flat)          # [B*T*S, C]
        slot_vec = slot_vec_flat.reshape(
            bsz, tmax, num_slots * self.text_encoder.output_dim
        )                                                                        # [B, T, S*C]

        step_input = torch.cat([slot_vec, numeric], dim=-1)                      # [B, T, S*C + D]
        step_input_flat = step_input.reshape(bsz * tmax, -1)

        step_vec_flat = self.step_mlp(step_input_flat)                           # [B*T, H]
        step_vec = step_vec_flat.reshape(bsz, tmax, self.output_dim)            # [B, T, H]

        return step_vec


class TableBiLSTMClassifier(nn.Module):
    """
    Классификатор физической таблицы.

    Pipeline:
        BinStepEncoder -> table-level BiLSTM -> MLP classifier

    Batch contract:
        batch = {
            'seqlens':  LongTensor[B],
            'numeric':  FloatTensor[B, T, D],
            'text_ids': LongTensor[B, T, S, L],
            'text_len': LongTensor[B, T, S],
        }

    Output:
        logits [B, num_classes]
    """

    def __init__(
        self,
        vocab_size: int,
        num_numeric_features: int,
        num_classes: int = 5,
        num_slots: int = 5,
        step_hidden_dim: int = 128,
        table_hidden_dim: int = 128,
        char_emb_dim: int = 64,
        char_hidden_dim: int = 64,
        pad_idx: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.step_encoder = BinStepEncoder(
            vocab_size=vocab_size,
            num_numeric_features=num_numeric_features,
            num_slots=num_slots,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
            step_hidden_dim=step_hidden_dim,
            pad_idx=pad_idx,
            dropout=dropout,
        )

        self.table_lstm = nn.LSTM(
            input_size=self.step_encoder.output_dim,
            hidden_size=table_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(table_hidden_dim * 2, table_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(table_hidden_dim, num_classes),
        )

        self.output_dim = num_classes

    def encode_steps(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Возвращает step embeddings [B, T, H].
        """
        required_keys = ("text_ids", "text_len", "numeric")
        missing = [k for k in required_keys if k not in batch]
        if missing:
            raise KeyError(f"batch is missing required keys: {missing}")

        step_vec = self.step_encoder(
            text_ids=batch["text_ids"],
            text_len=batch["text_len"],
            numeric=batch["numeric"],
        )
        return step_vec

    def encode_table(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Возвращает table representation [B, 2 * table_hidden_dim].
        """
        if "seqlens" not in batch:
            raise KeyError("batch is missing required key: 'seqlens'")

        step_vec = self.encode_steps(batch)          # [B, T, H]
        seqlens = batch["seqlens"].clamp_min(1)      # [B]

        packed = nn.utils.rnn.pack_padded_sequence(
            step_vec,
            lengths=seqlens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hn, _) = self.table_lstm(packed)

        if self.table_lstm.bidirectional:
            table_vec = torch.cat([hn[-2], hn[-1]], dim=-1)   # [B, 2H]
        else:
            table_vec = hn[-1]                                # [B, H]

        return table_vec

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                - text_ids: [B, T, S, L]
                - text_len: [B, T, S]
                - numeric:  [B, T, D]
                - seqlens:  [B]

        Returns:
            logits: [B, num_classes]
        """
        table_vec = self.encode_table(batch)
        logits = self.classifier(self.dropout(table_vec))
        return logits