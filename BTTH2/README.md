# BTTH2 - Mạng neural tích chập

- Bai1: LeNet trên MNIST
- Bai2: GoogLeNet trên VinaFood21
- Bai3: ResNet-18 trên VinaFood21
- Bai4: fine-tune ResNet-50 pretrained (`microsoft/resnet-50`) trên VinaFood21

Optimizer Adam, đánh giá bằng accuracy, precision, recall, F1 (macro) trên tập test.

## Chạy

Dữ liệu không có trong repo, để vào thư mục `Data/`:

```
Data/
  MNIST dataset/
  VinaFood21 dataset/train, test
```

```
cd Bai2
pip install -r requirements.txt
python assignment_2.py
```

## Lưu ý

- GoogLeNet có thêm BatchNorm sau mỗi conv, không có thì model không học được (loss đứng ở ~2.98).
- ResNet-18 giảm kích thước bằng MaxPool (kernel 3, stride 2, padding 0) giữa các block theo đề, không dùng conv stride 2.

## Kết quả

Notebook chạy trên Kaggle: https://www.kaggle.com/code/hoanggyen/ds201-lab02

| Bài | Mô hình | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Bai1 | LeNet | 0.9832 | 0.9830 | 0.9833 | 0.9831 |
| Bai2 | GoogLeNet | 0.4805 | 0.4891 | 0.4530 | 0.4409 |
| Bai3 | ResNet-18 | 0.4720 | 0.5231 | 0.4341 | 0.4388 |
| Bai4 | ResNet-50 pretrained | 0.8135 | 0.8163 | 0.8084 | 0.8081 |
