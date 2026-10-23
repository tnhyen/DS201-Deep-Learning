# DS201 - Deep Learning trong Khoa học Dữ liệu

Bài thực hành môn DS201, UIT. Mỗi bài thực hành nằm trong một thư mục riêng, có README mô tả chi tiết.

| Thư mục | Nội dung |
|---|---|
| [BTTH2](BTTH2) | Mạng neural tích chập: LeNet (MNIST), GoogLeNet, ResNet-18, fine-tune ResNet-50 pretrained (VinaFood21) |

## Cấu trúc một bài

```
BTTHx/
  BaiN/
    data_utils/       # đọc dữ liệu (Dataset, collate_fn)
    models/           # định nghĩa mô hình
    assignment_N.py   # huấn luyện và đánh giá
    requirements.txt
    results.txt       # log kết quả
  README.md
```

## Môi trường

Python 3.11+, PyTorch. Mỗi bài cài riêng:

```
cd BTTH2/Bai1
pip install -r requirements.txt
python assignment_1.py
```

Code tự chọn thiết bị theo thứ tự CUDA, MPS (Apple Silicon), CPU.

## Dữ liệu

Dữ liệu không đưa lên repo. Mỗi bài đọc từ thư mục `Data/` nằm trong thư mục bài thực hành (ví dụ `BTTH2/Data/`), cấu trúc mô tả trong README của từng bài.
