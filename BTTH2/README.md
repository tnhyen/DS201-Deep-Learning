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
| Bai1 | LeNet | 0.9800 | 0.9803 | 0.9796 | 0.9798 |
| Bai2 | GoogLeNet | 0.4677 | 0.5201 | 0.4426 | 0.4494 |
| Bai3 | ResNet-18 | 0.4753 | 0.5062 | 0.4616 | 0.4520 |
| Bai4 | ResNet-50 pretrained | 0.8183 | 0.8173 | 0.8152 | 0.8138 |
