import torch
from torch import optim
from torch import nn

from models.gru import GruModel
# Dùng lại pipeline dữ liệu, train/evaluate và vòng lặp huấn luyện của Bài 1
from assignment_1 import device, train, evaluate, fit, save_results, get_args, build_dataloaders


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(42)

    vocab, train_dataloader, dev_dataloader, test_dataloader = build_dataloaders(args.data_dir)

    print("Initializing model and utilities ...")
    # cấu hình được lưu kèm checkpoint để kiểm tra khi resume
    config = {
        "model": "gru",
        "vocab_size": vocab.len,
        "n_labels": vocab.n_labels,
        "hidden_size": 256,
        "n_layer": 5
    }
    model = GruModel(
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
        checkpoint_dir=args.checkpoint_dir, name="gru", config=config,
        max_epochs=args.max_epochs, resume=not args.no_resume
    )

    test_f1 = evaluate(model, test_dataloader, "Test")
    print(f"F1-score on test set: {test_f1:.4f}")

    save_results(args.checkpoint_dir, "gru", {
        "model": "GRU 5 layers, hidden 256",
        "dataset": "UIT-VSFC (topic)",
        "best_epoch": best_epoch,
        "dev_macro_f1": best_f1,
        "test_macro_f1": test_f1
    })
