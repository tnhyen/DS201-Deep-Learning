import os
import json
import argparse

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from tqdm import tqdm
import numpy as np

from data_utils.uit_vsfc import UIT_VSFC, Vocab, collate_fn
from models.lstm import LstmModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:

    running_loss = []
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["label"].to(device)

            # forward pass
            logits = model(input_ids, lengths)      # (bs, n_labels)
            loss = loss_fn(logits, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({
                "loss": np.array(running_loss).mean()
            })
    return float(np.mean(running_loss))


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch
    ) -> float:
    model.eval()

    gts = []        # ground truths
    pds = []        # predictions
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["label"].to(device)   # (bs, )

            # forward pass
            logits: torch.Tensor = model(input_ids, lengths)
            predicted_labels = logits.argmax(dim=-1)    # (bs, )

            gts.extend(labels.tolist())
            pds.extend(predicted_labels.tolist())

    f1 = float(f1_score(gts, pds, average="macro"))
    print(f"F1: {f1:.4f}")
    return f1


def get_rng_state() -> dict:
    # Shuffle của DataLoader, dropout và khởi tạo trọng số đều dùng RNG của PyTorch
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def fit(model, train_dataloader, dev_dataloader, loss_fn, optimizer, train_fn, evaluate_fn,
        checkpoint_dir: str, name: str, config: dict, max_epochs: int,
        resume: bool = True, max_patience: int = 5):
    '''
        Vòng lặp huấn luyện chung cho cả 3 bài:
          - lưu <name>_best.pt mỗi khi F1 trên dev cải thiện
          - lưu <name>_last.pt cuối mỗi epoch để chạy tiếp nếu Kaggle bị ngắt
          - kết thúc: nạp lại trọng số tốt nhất để đem đi test
        config: thông tin cấu hình (vocab size, số nhãn, hidden size, ...) lưu kèm checkpoint
                để không nạp nhầm checkpoint của bài khác / cấu hình khác.
    '''
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
    last_path = os.path.join(checkpoint_dir, f"{name}_last.pt")

    epoch = 0
    best_f1 = -1.0
    best_epoch = 0
    patience = 0

    has_last = os.path.exists(last_path)
    has_best = os.path.exists(best_path)

    if resume and (has_last or has_best):
        # Chạy tiếp cần đủ cả 2 file: last để train tiếp, best để test
        if not (has_last and has_best):
            raise FileNotFoundError(
                f"Checkpoint không đầy đủ trong {checkpoint_dir}: cần cả {name}_last.pt và {name}_best.pt. "
                f"Hãy copy đủ 2 file, hoặc chạy với --no_resume để train lại từ đầu."
            )

        ckpt = torch.load(last_path, map_location=device)
        if ckpt["config"] != config:
            raise ValueError(
                f"Checkpoint {last_path} không cùng cấu hình.\n"
                f"  checkpoint: {ckpt['config']}\n  hiện tại:   {config}\n"
                f"Hãy dùng --checkpoint_dir khác hoặc --no_resume."
            )

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng_state"])
        epoch = ckpt["epoch"]
        best_f1 = ckpt["best_f1"]
        best_epoch = ckpt["best_epoch"]
        patience = ckpt["patience"]
        print(f"Resume từ epoch {epoch} (best dev F1 = {best_f1:.4f} ở epoch {best_epoch})")
    elif has_last or has_best:
        print(f"--no_resume: bỏ qua checkpoint cũ của {name}, train lại từ đầu (file cũ sẽ bị ghi đè)")

    while epoch < max_epochs and patience < max_patience:
        epoch += 1
        train_loss = train_fn(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = float(evaluate_fn(model, dev_dataloader, epoch))

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            patience = 0
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "dev_f1": f1
            }, best_path)
            print(f"Đã lưu best checkpoint (epoch {epoch})")
        else:
            patience += 1

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "rng_state": get_rng_state(),
            "epoch": epoch,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "patience": patience
        }, last_path)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f} | dev F1 = {f1:.4f} | patience = {patience}")

    if patience >= max_patience:
        print("Patience reached! Training completed")
    else:
        print(f"Reached max_epochs = {max_epochs}")

    # Test bằng mô hình tốt nhất trên dev, không phải mô hình ở epoch cuối
    ckpt = torch.load(best_path, map_location=device)
    if ckpt["config"] != config:
        raise ValueError(f"{best_path} không cùng cấu hình với mô hình hiện tại.")
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model: epoch {ckpt['epoch']}, dev F1 = {ckpt['dev_f1']:.4f}")

    return best_epoch, best_f1


def save_results(checkpoint_dir: str, name: str, results: dict):
    path = os.path.join(checkpoint_dir, f"{name}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Đã lưu kết quả vào {path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/kaggle/working/UIT-VSFC")
    parser.add_argument("--checkpoint_dir", default="/kaggle/working/checkpoints")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--no_resume", action="store_true", help="bỏ qua checkpoint cũ, train lại từ đầu")
    return parser.parse_args()


def build_dataloaders(data_dir: str):
    print("Loading vocab ...")
    vocab = Vocab(train_path=os.path.join(data_dir, "UIT-VSFC-train.json"))

    print("Creating datasets ...")
    train_dataset = UIT_VSFC(path=os.path.join(data_dir, "UIT-VSFC-train.json"), vocab=vocab)
    dev_dataset = UIT_VSFC(path=os.path.join(data_dir, "UIT-VSFC-dev.json"), vocab=vocab)
    test_dataset = UIT_VSFC(path=os.path.join(data_dir, "UIT-VSFC-test.json"), vocab=vocab)
    print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")
    print(f"Vocab size: {vocab.len} | Labels: {vocab.l2i}")

    print("Creating dataloaders ...")
    # chỉ shuffle tập train; dev/test giữ nguyên thứ tự
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return vocab, train_dataloader, dev_dataloader, test_dataloader


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(42)

    vocab, train_dataloader, dev_dataloader, test_dataloader = build_dataloaders(args.data_dir)

    print("Initializing model and utilities ...")
    # cấu hình được lưu kèm checkpoint để kiểm tra khi resume
    config = {
        "model": "lstm",
        "vocab_size": vocab.len,
        "n_labels": vocab.n_labels,
        "hidden_size": 256,
        "n_layer": 5
    }
    model = LstmModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        n_layer=config["n_layer"],
        n_labels=config["n_labels"],
        padding_idx=vocab.pad_idx
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_epoch, best_f1 = fit(
        model, train_dataloader, dev_dataloader, loss_fn, optimizer, train, evaluate,
        checkpoint_dir=args.checkpoint_dir, name="lstm", config=config,
        max_epochs=args.max_epochs, resume=not args.no_resume
    )

    test_f1 = evaluate(model, test_dataloader, "Test")
    print(f"F1-score on test set: {test_f1:.4f}")

    save_results(args.checkpoint_dir, "lstm", {
        "model": "LSTM 5 layers, hidden 256",
        "dataset": "UIT-VSFC (topic)",
        "best_epoch": best_epoch,
        "dev_macro_f1": best_f1,
        "test_macro_f1": test_f1
    })
