# DS201 - Bài thực hành 2: Mạng neural tích chập

| Bài | Mô hình | Dữ liệu |
|---|---|---|
| Bai1 | LeNet | MNIST |
| Bai2 | GoogLeNet | VinaFood21 |
| Bai3 | ResNet-18 | VinaFood21 |
| Bai4 | ResNet-50 pretrained (HuggingFace `microsoft/resnet-50`) | VinaFood21 |

Tất cả dùng Adam, đánh giá trên tập test bằng accuracy, precision, recall, F1 (macro).

## Dữ liệu

Dữ liệu không đưa lên repo. Đặt đúng cấu trúc sau:

```
Data/
  MNIST dataset/        train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
  VinaFood21 dataset/   train/<ten-mon>/*.jpg, test/<ten-mon>/*.jpg
Bai1/ ... Bai4/
```

## Chạy

Python 3.11+.

```
cd Bai2
pip install -r requirements.txt
python assignment_2.py
```

Model có F1 tốt nhất được lưu ở `BaiN/checkpoints/best_model.pth`.

## Ghi chú cài đặt

- GoogLeNet: conv đầu padding 3, mọi MaxPool `ceil_mode=True` theo đề. Có thêm BatchNorm sau mỗi conv vì bản không BatchNorm không hội tụ với Adam trên VinaFood21.
- ResNet-18: giữa các nhóm residual block dùng MaxPool (kernel 3, stride 2, padding 0) theo đề để giảm kích thước; block dùng conv 1x1 ở nhánh tắt khi số kênh thay đổi.
- Seed cố định `torch.manual_seed(42)`.

## Kết quả

| Bài | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Bai1 | 0.9800 | 0.9803 | 0.9796 | 0.9798 |
| Bai2 | | | | |
| Bai3 | | | | |
| Bai4 | | | | |
