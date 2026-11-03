import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

# Nhãn của vị trí padding: CrossEntropyLoss(ignore_index=-100) sẽ bỏ qua
PAD_TAG_ID = -100


def load_jsonl(path: str) -> list[dict]:
    """PhoNER lưu dạng JSONL: mỗi dòng là một câu {"words": [...], "tags": [...]}."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class Vocab:
    def __init__(self, train_path: str):
        self.pad = "<pad>"
        self.unk = "<unk>"

        data = load_jsonl(train_path)
        all_words = set()
        tags = set()

        for item in data:
            # giữ nguyên token của dataset (word-level, đã nối bằng "_"),
            # không lowercase vì chữ hoa là tín hiệu quan trọng cho NER
            all_words.update(item["words"])
            tags.update(item["tags"])

        # 0 = <pad>, 1 = <unk>, các từ khác bắt đầu từ 2
        self.w2i = {
            self.pad: 0,
            self.unk: 1
        }
        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx
        self.i2w = {idx: word for word, idx in self.w2i.items()}

        # tag mapping lấy từ train (đủ 20 nhãn, kể cả I-GENDER không có trong dev)
        self.t2i = {tag: idx for idx, tag in enumerate(sorted(tags))}
        self.i2t = {idx: tag for tag, idx in self.t2i.items()}

    @property
    def n_tags(self) -> int:
        return len(self.t2i)

    @property
    def len(self) -> int:
        return len(self.w2i)

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def encode_words(self, words: list[str]) -> torch.Tensor:
        unk_idx = self.w2i[self.unk]
        word_ids = [self.w2i.get(word, unk_idx) for word in words]
        return torch.tensor(word_ids, dtype=torch.long)

    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        tag_ids = [self.t2i[tag] for tag in tags]
        return torch.tensor(tag_ids, dtype=torch.long)

    def decode_tags(self, tag_ids: list[int]) -> list[str]:
        return [self.i2t[idx] for idx in tag_ids]


class PhoNER(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        self._data = load_jsonl(path)
        self.vocab = vocab

        # kiểm tra dữ liệu trước để báo lỗi rõ ràng
        for idx, item in enumerate(self._data):
            if len(item["words"]) != len(item["tags"]):
                raise ValueError(f"Câu thứ {idx} trong {path}: số từ khác số nhãn.")
            for tag in item["tags"]:
                if tag not in vocab.t2i:
                    raise ValueError(f"Nhãn '{tag}' trong {path} không xuất hiện trong train.")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]

        return {
            "input_ids": self.vocab.encode_words(item["words"]),   # (len, )
            "tags": self.vocab.encode_tags(item["tags"])           # (len, )
        }


def collate_fn(items: list[dict]) -> dict:
    """Padding cả từ (bằng 0) lẫn nhãn (bằng -100) về cùng độ dài trong batch."""
    lengths = torch.tensor([item["input_ids"].size(0) for item in items], dtype=torch.long)
    max_len = int(lengths.max())

    input_ids = torch.stack([
        F.pad(item["input_ids"], pad=(0, max_len - item["input_ids"].size(0)), value=0)
        for item in items
    ])
    tags = torch.stack([
        F.pad(item["tags"], pad=(0, max_len - item["tags"].size(0)), value=PAD_TAG_ID)
        for item in items
    ])

    return {
        "input_ids": input_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, )
        "tags": tags                # (bs, max_len)
    }
