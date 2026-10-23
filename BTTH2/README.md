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

Python 3.13+ (đã chạy trên Kaggle, GPU T4).

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

Chạy trên Kaggle: [DS201-Lab02](https://www.kaggle.com/code/hoanggyen/ds201-lab02). Số liệu trên tập test, lấy theo epoch có F1 cao nhất.

| Bài | Mô hình | Epoch | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Bai1 | LeNet | 10/10 | 0.9800 | 0.9803 | 0.9796 | 0.9798 |
| Bai2 | GoogLeNet | 10/10 | 0.4677 | 0.5201 | 0.4426 | 0.4494 |
| Bai3 | ResNet-18 | 7/10 | 0.4753 | 0.5062 | 0.4616 | 0.4520 |
| Bai4 | ResNet-50 pretrained | 5/5 | 0.8183 | 0.8173 | 0.8152 | 0.8138 |

## Nhận xét

- LeNet đủ cho MNIST (ảnh xám 28x28, 10 lớp), F1 gần 0.98 sau 10 epoch.
- Trên VinaFood21, GoogLeNet và ResNet-18 train từ đầu chỉ đạt F1 khoảng 0.45. Với khoảng 10.000 ảnh cho 21 lớp, dữ liệu không đủ để học đặc trưng tốt từ đầu. ResNet-18 đạt đỉnh ở epoch 7 rồi dao động, GoogLeNet vẫn còn tăng ở epoch 10.
- GoogLeNet không có BatchNorm thì không học được: loss đứng ở khoảng 2.98 và mô hình chỉ đoán một lớp. Thêm BatchNorm thì hội tụ bình thường. ResNet-18 có sẵn BatchNorm và kết nối tắt nên train ổn định ngay.
- ResNet-50 pretrained trên ImageNet đạt F1 0.81 chỉ sau 5 epoch, cao gần gấp đôi hai mô hình train từ đầu. Với tập dữ liệu nhỏ, fine-tune mô hình đã huấn luyện sẵn hiệu quả hơn nhiều so với train từ đầu.
- Lớp khó nhất với ResNet-18 và ResNet-50 là `ca-ri` (F1 lần lượt 0.25 và 0.41), cũng là một trong các lớp ít ảnh nhất trong tập test (171 ảnh).
