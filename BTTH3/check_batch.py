'''
    Kiểm tra nhanh trước khi train thật (chạy vài giây):
    đọc dữ liệu -> lấy 1 batch -> kiểm tra shape, nhãn -> forward -> tính loss -> backward.
'''
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils import uit_vsfc, phoner
from models.lstm import LstmModel
from models.gru import GruModel
from models.bilstm import BiLstmNER

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def check_uit_vsfc(data_dir: str):
    print("=" * 20, "UIT-VSFC (Bài 1, 2)", "=" * 20)
    vocab = uit_vsfc.Vocab(os.path.join(data_dir, "UIT-VSFC-train.json"))
    datasets = {}
    for split in ["train", "dev", "test"]:
        datasets[split] = uit_vsfc.UIT_VSFC(os.path.join(data_dir, f"UIT-VSFC-{split}.json"), vocab)
        print(f"{split}: {len(datasets[split])} mẫu")
    print(f"Vocab size: {vocab.len} | Labels ({vocab.n_labels}): {vocab.l2i}")

    # lấy batch đầu của tập train
    dataloader = DataLoader(datasets["train"], batch_size=32, shuffle=False, collate_fn=uit_vsfc.collate_fn)
    batch = next(iter(dataloader))
    input_ids, lengths, labels = batch["input_ids"], batch["lengths"], batch["label"]
    print(f"input_ids: {tuple(input_ids.shape)} | lengths: {tuple(lengths.shape)} | label: {tuple(labels.shape)}")

    assert input_ids.size(0) == lengths.size(0) == labels.size(0)
    assert int(lengths.max()) == input_ids.size(1), "max_len phải bằng độ dài câu dài nhất"
    assert 0 <= labels.min() and labels.max() < vocab.n_labels, "label nằm ngoài khoảng hợp lệ"

    loss_fn = nn.CrossEntropyLoss()
    for model_cls in [LstmModel, GruModel]:
        model = model_cls(vocab_size=vocab.len, hidden_size=256, n_layer=5,
                          n_labels=vocab.n_labels, padding_idx=vocab.pad_idx).to(device)
        logits = model(input_ids.to(device), lengths)
        loss = loss_fn(logits, labels.to(device))
        loss.backward()
        assert logits.shape == (input_ids.size(0), vocab.n_labels)
        print(f"{model_cls.__name__}: logits {tuple(logits.shape)} | loss = {loss.item():.4f} -> OK")


def check_phoner(data_dir: str):
    print("=" * 20, "PhoNER-COVID19 (Bài 3)", "=" * 20)
    vocab = phoner.Vocab(os.path.join(data_dir, "train_word.json"))
    datasets = {}
    for split in ["train", "dev", "test"]:
        datasets[split] = phoner.PhoNER(os.path.join(data_dir, f"{split}_word.json"), vocab)
        print(f"{split}: {len(datasets[split])} câu")
    print(f"Vocab size: {vocab.len} | Số nhãn: {vocab.n_tags}")
    print(f"Nhãn: {list(vocab.t2i)}")
    # PhoNER word-level: 10 nhãn B-, 9 nhãn I- (không có I-GENDER) và O
    if vocab.n_tags != 20:
        print(f"CẢNH BÁO: kỳ vọng 20 nhãn, đang có {vocab.n_tags}. Hãy kiểm tra lại dữ liệu.")

    # lấy batch đầu của tập train
    dataloader = DataLoader(datasets["train"], batch_size=32, shuffle=False, collate_fn=phoner.collate_fn)
    batch = next(iter(dataloader))
    input_ids, lengths, tags = batch["input_ids"], batch["lengths"], batch["tags"]
    print(f"input_ids: {tuple(input_ids.shape)} | lengths: {tuple(lengths.shape)} | tags: {tuple(tags.shape)}")

    assert input_ids.shape == tags.shape
    # vị trí padding: token = 0 và tag = -100; vị trí thật: tag nằm trong [0, n_tags)
    real = tags != phoner.PAD_TAG_ID
    assert int(real.sum()) == int(lengths.sum()), "số token thật phải bằng tổng lengths"
    assert (input_ids[~real] == vocab.pad_idx).all()
    assert tags[real].min() >= 0 and tags[real].max() < vocab.n_tags

    model = BiLstmNER(vocab_size=vocab.len, hidden_size=256, n_layer=5,
                      n_tags=vocab.n_tags, padding_idx=vocab.pad_idx).to(device)
    logits = model(input_ids.to(device), lengths)
    loss_fn = nn.CrossEntropyLoss(ignore_index=phoner.PAD_TAG_ID)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), tags.to(device).reshape(-1))
    loss.backward()
    assert logits.shape == (*input_ids.shape, vocab.n_tags)
    print(f"BiLstmNER: logits {tuple(logits.shape)} | loss = {loss.item():.4f} -> OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uit_dir", default="/kaggle/working/UIT-VSFC")
    parser.add_argument("--ner_dir", default="/kaggle/working/PhoNER_COVID19/data/word")
    args = parser.parse_args()

    print(f"Device: {device}")
    check_uit_vsfc(args.uit_dir)
    check_phoner(args.ner_dir)
    print("Tất cả kiểm tra đều OK, có thể bắt đầu train.")
