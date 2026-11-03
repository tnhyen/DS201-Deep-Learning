import json
import string

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_json(path: str) -> list[dict]:
    """Đọc file UIT-VSFC dạng JSON (một list các mẫu)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} phải chứa một JSON list.")

    return data


class Vocab:
    def __init__(self, train_path: str, label_key: str = "topic"):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.label_key = label_key

        data = load_json(train_path)

        all_words = set()
        labels = set()

        for item in data:
            if "sentence" not in item or label_key not in item:
                raise KeyError(
                    f"Mỗi mẫu trong {train_path} cần có 'sentence' "
                    f"và '{label_key}'."
                )

            sentence = self.preprocess_sentence(item["sentence"])
            all_words.update(sentence.split())
            labels.add(item[label_key])

        # 0 dành cho padding, 1 dành cho từ chưa gặp.
        self.w2i = {
            self.pad: 0,
            self.unk: 1
        }

        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx

        self.i2w = {idx: word for word, idx in self.w2i.items()}

        # Tạo label mapping chỉ từ tập train.
        self.l2i = {
            label: idx for idx, label in enumerate(sorted(labels))
        }
        self.i2l = {idx: label for label, idx in self.l2i.items()}

    @property
    def n_labels(self) -> int:
        return len(self.l2i)

    @property
    def len(self) -> int:
        return len(self.w2i)

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def preprocess_sentence(self, sentence: str) -> str:
        # Giữ đúng cách tiền xử lý trong sample: lowercase và bỏ dấu câu.
        translator = str.maketrans("", "", string.punctuation)
        sentence = sentence.lower()
        sentence = sentence.translate(translator)
        return sentence

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        sentence = self.preprocess_sentence(sentence)
        words = sentence.split()

        # Tránh câu rỗng sau khi loại bỏ dấu câu.
        if len(words) == 0:
            words = [self.unk]

        unk_idx = self.w2i[self.unk]
        word_ids = [self.w2i.get(word, unk_idx) for word in words]

        return torch.tensor(word_ids, dtype=torch.long)

    def encode_label(self, label: str) -> torch.Tensor:
        if label not in self.l2i:
            raise ValueError(
                f"Nhãn '{label}' không có trong label mapping được tạo từ train. "
                f"Hãy kiểm tra lại dữ liệu hoặc label_key='{self.label_key}'."
            )

        return torch.tensor(self.l2i[label], dtype=torch.long)

    def decode_label(self, label_ids: torch.Tensor) -> list[str]:
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.tolist()

        return [self.i2l[idx] for idx in label_ids]


class UIT_VSFC(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        self._data = load_json(path)
        self.vocab = vocab

        # Kiểm tra nhãn trước để báo lỗi rõ ràng nếu dev/test có nhãn lạ.
        for idx, item in enumerate(self._data):
            if "sentence" not in item or vocab.label_key not in item:
                raise KeyError(
                    f"Mẫu thứ {idx} trong {path} thiếu 'sentence' "
                    f"hoặc '{vocab.label_key}'."
                )

            if item[vocab.label_key] not in vocab.l2i:
                raise ValueError(
                    f"Nhãn '{item[vocab.label_key]}' ở mẫu thứ {idx} "
                    f"trong {path} không xuất hiện trong train."
                )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]

        input_ids = self.vocab.encode_sentence(item["sentence"])
        label = self.vocab.encode_label(item[self.vocab.label_key])

        return {
            "input_ids": input_ids,
            "label": label
        }


def collate_fn(items: list[dict]) -> dict:
    """Padding các câu trong cùng batch về độ dài bằng nhau."""
    if len(items) == 0:
        raise ValueError("Batch rỗng, không thể thực hiện collate.")

    input_ids = [item["input_ids"] for item in items]
    lengths = torch.tensor(
        [ids.size(0) for ids in input_ids],
        dtype=torch.long
    )

    max_len = int(lengths.max().item())

    padded_input_ids = torch.stack([
        F.pad(
            ids,
            pad=(0, max_len - ids.size(0)),
            mode="constant",
            value=0
        )
        for ids in input_ids
    ])

    labels = torch.stack([item["label"] for item in items])

    return {
        "input_ids": padded_input_ids,  # (batch_size, max_len)
        "lengths": lengths,             # (batch_size,)
        "label": labels                 # (batch_size,)
    }
