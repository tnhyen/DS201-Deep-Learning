import torch
from torch import nn
from transformers import ResNetForImageClassification


class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes: int = 21, checkpoint: str = "microsoft/resnet-50"):
        super().__init__()
        self.backbone = ResNetForImageClassification.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=images).logits  # (bs, num_classes)
