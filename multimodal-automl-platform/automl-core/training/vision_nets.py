import torch.nn as nn
import torchvision.models as models

class VisionModelFactory:
    @staticmethod
    def build_model(arch: str = "resnet50", num_classes: int = 2) -> nn.Module:
        if arch.lower() == "vit":
            model = models.vit_b_16(weights=None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
