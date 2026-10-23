from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

from data_utils.vinafood_dataset import VinaFoodDataset, collate_fn
from models.pretrained_resnet50 import PretrainedResNet50

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "Data" / "VinaFood21 dataset"
CKPT_PATH = ROOT / "checkpoints" / "best_model.pth"
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 4

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, optimizer, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, item in enumerate(pbar, 1):
        images, labels = item["image"].to(device), item["label"].to(device)
        loss = loss_fn(model(images), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / step:.4f}")
    return total_loss / len(dataloader)


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader):
    model.eval()
    preds, trues = [], []
    for item in dataloader:
        preds.append(model(item["image"].to(device)).argmax(dim=1).cpu())
        trues.append(item["label"])
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }


def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    return compute_metrics(*predict(model, dataloader))


def main():
    torch.manual_seed(42)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = VinaFoodDataset(DATA_DIR / "train")
    test_dataset = VinaFoodDataset(DATA_DIR / "test")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
                                 num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = PretrainedResNet50(num_classes=len(train_dataset.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, epoch)
        scores = evaluate(test_dataloader, model)
        print(f"Epoch {epoch} | loss {train_loss:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in scores.items()))

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            torch.save(model.state_dict(), CKPT_PATH)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    y_true, y_pred = predict(model, test_dataloader)
    print("\nBest model on test set:")
    for k, v in compute_metrics(y_true, y_pred).items():
        print(f"  {k:<9}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4))


if __name__ == "__main__":
    main()
