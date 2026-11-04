# DS201 - Deep Learning trong Khoa học Dữ liệu

Bài thực hành môn DS201, UIT. Mỗi bài thực hành nằm trong một thư mục riêng, có README mô tả chi tiết.

| Thư mục | Nội dung |
|---|---|
| [BTTH2](BTTH2) | Mạng neural tích chập: LeNet (MNIST), GoogLeNet, ResNet-18, fine-tune ResNet-50 pretrained (VinaFood21) |
| [BTTH3](BTTH3) | Mạng neural hồi quy: LSTM, GRU phân loại văn bản (UIT-VSFC), BiLSTM nhận dạng thực thể (PhoNER-COVID19) |

## Cấu trúc một bài

```
BTTHx/
  BaiN/
    data_utils/       # đọc dữ liệu (Dataset, collate_fn)
    models/           # định nghĩa mô hình
    assignment_N.py   # huấn luyện và đánh giá
    requirements.txt
  README.md
```

BTTH3 để 3 bài chung một thư mục (`data_utils/`, `models/`, `assignment_1.py` … `assignment_3.py`) vì Bài 1 và Bài 2 dùng chung dữ liệu và vòng lặp huấn luyện.

## Môi trường

Python 3.11+, PyTorch. Mỗi bài cài riêng:

```
cd BTTH2/Bai1
pip install -r requirements.txt
python assignment_1.py
```

Code tự chọn thiết bị theo thứ tự CUDA, MPS (Apple Silicon), CPU.

BTTH3 chạy trên Kaggle (GPU T4) bằng notebook `BTTH3/run_kaggle.ipynb`.

## Dữ liệu

Dữ liệu không đưa lên repo. BTTH2 đọc từ thư mục `Data/` nằm trong thư mục bài thực hành (ví dụ `BTTH2/Data/`), cấu trúc mô tả trong README của từng bài. BTTH3 tự tải dữ liệu trong notebook Kaggle (UIT-VSFC từ Google Drive của đề bài, PhoNER-COVID19 từ GitHub của VinAI).
