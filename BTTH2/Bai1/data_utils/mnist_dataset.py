import numpy as np
import torch
from torch.utils.data import Dataset


def read_idx(path) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([item["image"] for item in items]),  # (bs, 28, 28)
        "label": torch.tensor([item["label"] for item in items])  # (bs,)
    }


class MnistDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.images = torch.from_numpy(read_idx(image_path).astype(np.float32) / 255.0)  # (N, 28, 28)
        self.labels = read_idx(label_path).astype(np.int64)                             # (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "label": self.labels[idx]
        }
