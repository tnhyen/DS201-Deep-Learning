from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([item["image"] for item in items]),  # (bs, 3, 224, 224)
        "label": torch.tensor([item["label"] for item in items])  # (bs,)
    }


class VinaFoodDataset(Dataset):
    def __init__(self, root, image_size: int = 224):
        root = Path(root)
        self.classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        self.samples = [
            (path, label)
            for label, name in enumerate(self.classes)
            for path in sorted((root / name).glob("*.jpg"))
        ]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        image = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD  # (224, 224, 3)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),  # (3, 224, 224)
            "label": label
        }
