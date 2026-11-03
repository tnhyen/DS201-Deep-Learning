import os
import argparse

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, classification_report

from tqdm import tqdm
import numpy as np

from data_utils.phoner import PhoNER, Vocab, collate_fn, PAD_TAG_ID
from models.bilstm import BiLstmNER
# Dùng lại vòng lặp huấn luyện (checkpoint, resume, early stopping) của Bài 1
from assignment_1 import device, fit, save_results


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
            tags = item["tags"].to(device)          # (bs, len)

            # forward pass
            logits = model(input_ids, lengths)      # (bs, len, n_tags)

            # mỗi token là một mẫu phân loại -> duỗi về (bs * len, n_tags)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tags.reshape(-1))

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
def predict(model, dataloader, vocab: Vocab, desc: str):
    '''Trả về danh sách nhãn thật và nhãn dự đoán (dạng chuỗi BIO) cho từng câu.'''
    model.eval()

    gts = []
    pds = []
    for item in tqdm(dataloader, desc=desc):
        input_ids = item["input_ids"].to(device)
        lengths = item["lengths"]
        tags = item["tags"]

        logits = model(input_ids, lengths)          # (bs, len, n_tags)
        predicted = logits.argmax(dim=-1).cpu()     # (bs, len)

        # bỏ phần padding: chỉ lấy `length` token đầu của mỗi câu
        for i, length in enumerate(lengths.tolist()):
            gts.append(vocab.decode_tags(tags[i, :length].tolist()))
            pds.append(vocab.decode_tags(predicted[i, :length].tolist()))

    return gts, pds


def make_evaluate(vocab: Vocab):
    def evaluate(model, dataloader, epoch) -> float:
        gts, pds = predict(model, dataloader, vocab, desc=f"Epoch {epoch} - Evaluating")
        # F1 mức thực thể (entity-level, micro), như bài báo PhoNER-COVID19
        f1 = float(f1_score(gts, pds))
        print(f"Entity F1: {f1:.4f}")
        return f1
    return evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/kaggle/working/PhoNER_COVID19/data/word")
    parser.add_argument("--checkpoint_dir", default="/kaggle/working/checkpoints")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--no_resume", action="store_true", help="bỏ qua checkpoint cũ, train lại từ đầu")
    args = parser.parse_args()
    torch.manual_seed(42)

    print("Loading vocab ...")
    vocab = Vocab(train_path=os.path.join(args.data_dir, "train_word.json"))

    print("Creating datasets ...")
    train_dataset = PhoNER(os.path.join(args.data_dir, "train_word.json"), vocab)
    dev_dataset = PhoNER(os.path.join(args.data_dir, "dev_word.json"), vocab)
    test_dataset = PhoNER(os.path.join(args.data_dir, "test_word.json"), vocab)
    print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")
    print(f"Vocab size: {vocab.len} | Số nhãn: {vocab.n_tags}")

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print("Initializing model and utilities ...")
    # cấu hình được lưu kèm checkpoint để kiểm tra khi resume
    config = {
        "model": "bilstm",
        "vocab_size": vocab.len,
        "n_tags": vocab.n_tags,
        "hidden_size": 256,
        "n_layer": 5
    }
    model = BiLstmNER(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        n_layer=config["n_layer"],
        n_tags=config["n_tags"],
        padding_idx=vocab.pad_idx
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # bỏ qua vị trí padding khi tính loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TAG_ID).to(device)

    evaluate = make_evaluate(vocab)
    best_epoch, best_f1 = fit(
        model, train_dataloader, dev_dataloader, loss_fn, optimizer, train, evaluate,
        checkpoint_dir=args.checkpoint_dir, name="bilstm", config=config,
        max_epochs=args.max_epochs, resume=not args.no_resume
    )

    gts, pds = predict(model, test_dataloader, vocab, desc="Test")
    test_f1 = float(f1_score(gts, pds))
    test_macro_f1 = float(f1_score(gts, pds, average="macro"))
    print(f"Entity F1 (micro) on test set: {test_f1:.4f}")
    print(f"Entity F1 (macro) on test set: {test_macro_f1:.4f}")
    print(classification_report(gts, pds, digits=4, zero_division=0))

    save_results(args.checkpoint_dir, "bilstm", {
        "model": "Encoder 5 BiLSTM layers, hidden 256",
        "dataset": "PhoNER-COVID19 (word-level)",
        "best_epoch": best_epoch,
        "dev_entity_f1": best_f1,
        "test_entity_micro_f1": test_f1,
        "test_entity_macro_f1": test_macro_f1
    })
