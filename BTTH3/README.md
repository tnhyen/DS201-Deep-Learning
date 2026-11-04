# DS201 - Lab 3: Mạng neural hồi quy (RNN)

| Bài | Mô hình | Dataset | Nhiệm vụ | Độ đo |
| --- | --- | --- | --- | --- |
| 1 | LSTM 5 lớp, hidden 256 | UIT-VSFC | Phân loại văn bản (topic, 4 lớp) | Macro F1 |
| 2 | GRU 5 lớp, hidden 256 | UIT-VSFC | Phân loại văn bản (topic, 4 lớp) | Macro F1 |
| 3 | Encoder 5 lớp BiLSTM, hidden 256 | PhoNER-COVID19 (word) | Nhận dạng thực thể (NER) | Entity-level F1 (seqeval) |

Cả 3 bài: Embedding 256, dropout 0.2, Adam (lr = 0.001), batch size 32, CrossEntropyLoss, early stopping patience 5 trên dev.

## Cấu trúc

```
BTTH3/
├── data_utils/
│   ├── uit_vsfc.py      # Vocab, Dataset, collate_fn cho UIT-VSFC (nhãn: "topic")
│   └── phoner.py        # Vocab, Dataset, collate_fn cho PhoNER (JSONL, nhãn BIO)
├── models/
│   ├── lstm.py          # LstmModel
│   ├── gru.py           # GruModel
│   └── bilstm.py        # BiLstmEncoder + BiLstmNER
├── assignment_1.py      # train/evaluate + vòng lặp fit() dùng chung
├── assignment_2.py      # dùng lại pipeline Bài 1, thay LSTM bằng GRU
├── assignment_3.py      # train/evaluate theo token, F1 mức thực thể
├── check_batch.py       # kiểm tra 1 batch (shape, nhãn, padding, forward, loss) trước khi train
└── run_kaggle.ipynb     # notebook chạy trên Kaggle
```

## Chạy

```bash
pip install seqeval   # chỉ cần cho Bài 3
python check_batch.py --uit_dir /kaggle/working/UIT-VSFC --ner_dir /kaggle/working/PhoNER_COVID19/data/word
python assignment_1.py --data_dir /kaggle/working/UIT-VSFC
python assignment_2.py --data_dir /kaggle/working/UIT-VSFC
python assignment_3.py --data_dir /kaggle/working/PhoNER_COVID19/data/word
```

Tham số chung: `--checkpoint_dir` (mặc định `/kaggle/working/checkpoints`), `--max_epochs` (mặc định 50), `--no_resume` (bỏ qua checkpoint cũ, train lại từ đầu).

## Những điểm sửa so với sample

1. **Lưu checkpoint tốt nhất**: sample chỉ ghi nhớ `best_f1` nên khi dừng, mô hình đem đi test là mô hình ở epoch cuối. Code này lưu `<model>_best.pt` mỗi khi dev F1 tăng và nạp lại trước khi test.
2. **Dev/test không shuffle** (sample để `shuffle=True` cho cả 3 tập).
3. **Giới hạn `max_epochs`** để vòng `while True` không chạy vô hạn trên Kaggle.
4. **Chạy tiếp khi bị ngắt**: cuối mỗi epoch lưu `<model>_last.pt` (model, optimizer, trạng thái random của PyTorch/CUDA, epoch, best F1, patience). Chạy lại script với cùng `--checkpoint_dir` sẽ tự train tiếp.
   - Chỉ resume khi có **đủ** `<model>_last.pt` và `<model>_best.pt`; thiếu một file thì báo lỗi ngay thay vì lỗi ở cuối lúc test.
   - Mỗi checkpoint lưu kèm `config` (tên model, vocab size, số nhãn, hidden size, số layer); khác cấu hình hiện tại thì báo lỗi, tránh nạp nhầm checkpoint của bài khác.
   - Lưu trạng thái random nên train 3 epoch liên tục và train 2 epoch + resume 1 epoch cho kết quả giống nhau (đã kiểm tra trên CPU; trên GPU cuDNN có thể lệch rất nhỏ).
   - NumPy/`random` không cần lưu vì quá trình train không dùng đến (shuffle, dropout, khởi tạo đều dùng RNG của PyTorch).

## Ghi chú

- `hidden_size=256` của BiLSTM hiểu theo PyTorch là 256 **cho mỗi chiều**, nên output nối 2 chiều là 512. Nếu giảng viên định nghĩa 256 là tổng 2 chiều thì đổi thành 128 mỗi chiều.
- Tiền xử lý UIT-VSFC giữ như sample (`string.punctuation` chỉ bỏ dấu câu ASCII, không bỏ hết dấu câu Unicode như `“ ” …`).
- `Sample/` là code gốc của giảng viên, giữ để đối chiếu.

## Bài 3: khác gì so với Bài 1, 2

- Mỗi token có một nhãn → model trả về `(bs, len, n_tags)` thay vì `(bs, n_labels)`; dùng `pad_packed_sequence` để lấy output ở mọi bước.
- BiLSTM nối 2 chiều nên lớp Linear nhận `2 * 256 = 512` chiều.
- Nhãn ở vị trí padding = `-100`, `CrossEntropyLoss(ignore_index=-100)` bỏ qua khi tính loss.
- Token giữ nguyên như dataset (không lowercase, không bỏ dấu câu) vì chữ hoa và dấu câu có ích cho NER.
- Label mapping lấy từ train: 20 nhãn (10 `B-`, 9 `I-`, `O`; dataset không có `I-GENDER`). Dev thiếu `I-AGE` nhưng mapping vẫn giữ vì lấy từ train.
- Đánh giá bằng F1 mức thực thể (micro, như bài báo PhoNER-COVID19), kèm macro F1 và báo cáo theo từng loại thực thể.

## Kết quả thực nghiệm

Chạy trên Kaggle GPU T4 (version 2, 03/11/2026). Best epoch ghi dạng `best / số epoch đã chạy` (dừng do early stopping, patience 5). Thời gian train lấy từ log Kaggle, tính từ lúc khởi tạo model đến lúc nạp best model.

| Thông số | LSTM | GRU | BiLSTM |
| --- | --- | --- | --- |
| Dataset | UIT-VSFC (topic) | UIT-VSFC (topic) | PhoNER-COVID19 (word) |
| Layers | 5 | 5 | 5 |
| Hidden size | 256 | 256 | 256 mỗi chiều |
| Batch size | 32 | 32 | 32 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Độ đo | Macro F1 | Macro F1 | Entity F1 (micro) |
| Best epoch | 3 / 8 | 14 / 19 | 10 / 15 |
| Best dev F1 | 0.7428 | 0.7642 | 0.8851 |
| Test F1 | 0.7404 | 0.7407 | 0.8737 (macro 0.7958) |
| Training time | ~54 s | ~108 s | ~99 s |
